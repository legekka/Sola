import torch
import random
import phonemizer
import numpy as np
import yaml
from munch import Munch
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
from modules.styletts2.utils import *
from modules.styletts2.models import *
from modules.styletts2.text_utils import TextCleaner
from modules.styletts2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from modules.styletts2.Utils.PLBERT.util import load_plbert
import pyaudio
import queue
import time
import threading

class TextToSpeech:
    def __init__(self, ref_audio_path="Samples/Ganyu2.wav"):
        self.ref_audio_path = ref_audio_path
        self.new_event = None
        self.textcleaner = TextCleaner()
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean, self.std = -4, 4

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if os.name == 'nt':
            os.environ["PHONEMIZER_ESPEAK_PATH"] = "C:\\Program Files\\eSpeak\\command_line\\espeak.exe"
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll"
        
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
        config = yaml.safe_load(open("Models/LibriTTS/config.yml"))

        self.text_aligner = load_ASR_models(config.get('ASR_path', False), config.get('ASR_config', False))
        self.pitch_extractor = load_F0_models(config.get('F0_path', False))
        self.plbert = load_plbert(config.get('PLBERT_dir', False))

        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, self.text_aligner, self.pitch_extractor, self.plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        params_whole = torch.load("models/LibriTTS/epochs_2nd_00020.pth", map_location='cpu')
        params = params_whole['net']

        for key in self.model:
            if key in params:
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)

        _ = [self.model[key].eval() for key in self.model]

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
            clamp=False
        )

        torch.cuda.empty_cache()

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask

    def preprocess(self, wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, path):
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)
    
    def inference(self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self.length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                  embedding=bert_dur,
                                  embedding_scale=embedding_scale,
                                  features=ref_s,  # reference from the same speaker as the embedding
                                  num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,
                                                  s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,
                                     F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50]  # weird pulse at the end of the model, need to be fixed later

    def callback(self, in_data, frame_count, time_info, status):
        # we have to get the data from the queue, but it's important that the data can be longer than the frame_count
        # so we don't want to completely remove the data from the queue, but only the amount that we need, after that we truncate the data in the queue, and we put it back to the same place in the queue
        # also if it's the end, we pad the data with zeros
        
        # check if there's anything in the queue
        if len(self.audio_list) == 0:
            return (np.zeros(frame_count).astype(np.float32).tostring(), pyaudio.paContinue)
        
        data = self.audio_list[0]
        status = self.audio_list_meta[0]["status"]

        if status == "new":
            self.audio_list_meta[0]["status"] = "playing"
            ## here we want to send a message to the main thread to update the gui
            self.new_event = {
                "text": self.audio_list_meta[0]["text"],
                "audio_length": self.audio_list_meta[0]["audio_length"]
            }

        if len(data) > frame_count:
            self.audio_list[0] = data[frame_count:]
            data = data[:frame_count]
        elif len(data) < frame_count:
            data = np.pad(data, (0, frame_count - len(data)), 'constant')
            self.audio_list.pop(0)
            self.audio_list_meta.pop(0)
        data = data.astype(np.float32).tostring()
        return (data, pyaudio.paContinue)

    def init(self):
        self.ref_s = self.compute_style(self.ref_audio_path)
        self.channels = 1
        self.sr = 24000
        self.audio = pyaudio.PyAudio()
        self.device_index = 8
        self.audio_list = []
        self.audio_list_meta = []
        # open the audio stream
        self.stream = self.audio.open(format=pyaudio.paFloat32,
                                        channels=self.channels,
                                        rate=self.sr,
                                        output=True,
                                        stream_callback=self.callback,
                                        frames_per_buffer=12000,
                                        output_device_index=self.device_index)
        self.stream.start_stream()
    
    def play_audio(self, wav, text, audio_length): 
        meta = {
            "text": text,
            "audio_length": audio_length,
            "status": "new"
        }
        self.audio_list.append(wav)
        self.audio_list_meta.append(meta)
    
    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
    
    def synthetize(self, texts="You've forgot the text!", alpha=0.1, beta=0.5, diffusion_steps=7, embedding_scale=2, gui=None):
        # first we need to check if the text is a string or a list of strings
        if isinstance(texts, str):
            texts = [texts]

        start = time.time()
        wavs = []
        full_text = ""
        for text in texts:
            if text.strip() == "":
                continue
            wav = self.inference(text, self.ref_s, alpha, beta, diffusion_steps, embedding_scale)
            wavs.append(wav)
            full_text += text + " "
        full_text = full_text.strip()
        print("StyleTTS2 time: " + str(round(time.time() - start, 2)) + "s")
        wavs = np.concatenate(wavs)
        audio_length = len(wavs) / self.sr * 1000
        self.play_audio(wavs, full_text, audio_length)
    
    def run(self, texts):
        self.init()
        for text in texts:
            self.synthetize(text)
        self.close()


class TextToSpeechThreaded(TextToSpeech):
    def __init__(self, ref_audio_path="Samples/Ganyu2.wav"):
        super().__init__(ref_audio_path)
        self.thread = None

    def run(self, texts):
        self.thread = threading.Thread(target=super().run, args=(texts,))
        self.thread.start()

    def synthetize_async(self, text="You've forgot the text!", alpha=0.1, beta=0.5, diffusion_steps=15, embedding_scale=2):
        self.thread = threading.Thread(target=self.synthetize, args=(text, alpha, beta, diffusion_steps, embedding_scale))
        self.thread.start()

    def join(self):
        if self.thread is not None:
            self.thread.join()

class TextToSpeechQueue(TextToSpeechThreaded):
    def __init__(self, ref_audio_path="Samples/Ganyu2.wav"):
        super().__init__(ref_audio_path)
        self.sentences = queue.Queue()
        self.closed = False


    def run(self):
        def loop():
            while not self.closed:
                if not self.sentences.empty():
                    sentence = self.sentences.get()
                    self.synthetize(sentence)
                time.sleep(0.5)
                    
        threading.Thread(target=loop).start()

    def add_sentence(self, sentence):
        self.sentences.put(sentence)

    def add_sentences(self, sentences):
        self.sentences.put(sentences)

    def close(self):
        self.closed = True

    def wait(self):
        # first we wait for all sentences to be synthetized
        while not self.sentences.empty():
            time.sleep(0.5)
        # after that we wait for the audio list to be empty - so no audio is being played
        while len(self.audio_list) > 0:
            time.sleep(0.5)
        