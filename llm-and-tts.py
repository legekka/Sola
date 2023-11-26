from modules.stts import TextToSpeechQueue
from modules.textgenerator import TextGeneratorAPIClient


if __name__ == "__main__":
    print("Initializing TextGenerator...")
    tg = TextGeneratorAPIClient()
    tts = TextToSpeechQueue(ref_audio_path="Samples/female_reference2.wav")

    print("Preparing TTS...")
    tts.init()
    #tts.prepare()

    print("Starting TTS...")
    tts.run()

    print("Start!")

    sentences = tg.rephrase("Full control re-established.  Reminder: you have a limpet controller but are not carrying any limpets.")
    print(sentences)
    for sentence in sentences:
        tts.add_sentence(sentence)

    sentences = tg.rephrase("2 Biological Surface Signals detected.")
    print(sentences)
    for sentence in sentences:
        tts.add_sentence(sentence)

    sentences = tg.rephrase("Planet A 6 is notable, with an unusually oblong orbit.")
    print(sentences)
    for sentence in sentences:
        tts.add_sentence(sentence)

    tts.wait()
    tts.close()