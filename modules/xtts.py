from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import time
import pyaudio
import queue
import threading
import numpy as np

class TextToSpeech:
    def __init__(self, model_path, ref_audio_path):
        self.model_path = model_path
        self.ref_audio_path = ref_audio_path
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 22050
        self.CHUNK_SIZE = int(22272)
        self.q = queue.Queue()
        self.audio = pyaudio.PyAudio()
        self.device_index = 7
        self.stream = None
        self.model = None
        self.config = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None
        self.start_time = None

    def callback(self, in_data, frame_count, time_info, status):
        audio_data = b''
        while len(audio_data) < frame_count * self.audio.get_sample_size(self.FORMAT):
            try:
                chunk = self.q.get_nowait().tobytes()
                audio_data += chunk
            except queue.Empty:
                audio_data += b'\x00' * (frame_count * self.audio.get_sample_size(self.FORMAT) - len(audio_data))
                break
        return (audio_data, pyaudio.paContinue)

    def init(self):
        self.config = XttsConfig()
        self.config.load_json(self.model_path + "config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=self.model_path, eval=True)
        self.model.cuda()
        self.stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, output=True, stream_callback=self.callback, frames_per_buffer=self.CHUNK_SIZE, output_device_index=self.device_index)
        self.stream.start_stream()

    def prepare(self):
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=self.ref_audio_path, gpt_cond_len=7)

    def get_stream(self, text, language):
        output_generator = self.model.inference_stream(text=text, language=language, gpt_cond_latent=self.gpt_cond_latent, speaker_embedding=self.speaker_embedding)
        return output_generator

    def play_audio(self, generator):
        complete_audio = None
        for audio_chunk in generator:
            self.start_time = time.time()
            audio_array = audio_chunk.cpu().numpy()
            self.q.put(audio_array)
            if complete_audio is None:
                complete_audio = audio_array
            else:
                complete_audio = np.concatenate((complete_audio, audio_array))
        while not self.q.empty():
            time.sleep(1)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def synthetize(self, text="You've forgot the text!", language="en"):
        self.start_time = time.time()
        output_generator = self.get_stream(text, language)
        self.play_audio(output_generator)

    def run(self, texts, language):
        self.init()
        self.prepare()
        print("Starting inference_stream()...")
        for text in texts:
            self.synthetize(text, language)
        self.close()


class TextToSpeechThreaded(TextToSpeech):
    def __init__(self, model_path, ref_audio_path):
        super().__init__(model_path, ref_audio_path)
        self.thread = None

    def run(self, texts, language):
        self.thread = threading.Thread(target=super().run, args=(texts, language))
        self.thread.start()

    def synthetize_async(self, text="You've forgot the text!", language="en"):
        self.thread = threading.Thread(target=self.synthetize, args=(text, language))
        self.thread.start()

    def join(self):
        if self.thread is not None:
            self.thread.join()



class TextToSpeechQueue(TextToSpeechThreaded):
    def __init__(self, model_path, ref_audio_path):
        super().__init__(model_path, ref_audio_path)
        self.sentences = queue.Queue()
        self.closed = False

    def run(self):
        def loop():
            while not self.closed:
                if not self.sentences.empty():
                    sentence = self.sentences.get()
                    self.synthetize(sentence, "en")
        threading.Thread(target=loop).start()

    def add_sentence(self, sentence):
        self.sentences.put(sentence)

    def close(self):
        self.closed = True

    def wait(self):
        while not self.sentences.empty():
            time.sleep(0.5)  # sleep for a short period of time to avoid busy waiting


if __name__ == "__main__":
    tts = TextToSpeech("C:/Users/legek/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2/", "Samples/Ganyu2.wav")
    texts = [
        "The Solaris has successfully completed the exploration process in 18 systems. The total amount earned from these sales is now over 77 million credits!",
        "This means that Commander can purchase various resources and upgrades for their spaceship, including bonus credits for first discoveries.",
        "You've also made a profit of 174,000 credits on this adventure!"
    ]
    tts.run(texts, "en")