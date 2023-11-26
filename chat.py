from modules.xtts import TextToSpeechQueue
from modules.textgenerator import TextGenerator
import time

# we want to create a Queue for the TTS to use
import queue

if __name__ == "__main__":
    print("Initializing TextGenerator...")
    tg = TextGenerator()
    tts = TextToSpeechQueue("C:/Users/legek/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2/", "Samples/Ganyu2.wav")

    print("Preparing TTS...")
    tts.init()
    tts.prepare()

    print("Starting TTS...")
    tts.run()

    print("Start!")


    start = time.time()
    sentences = tg.chat("Hey Sola! Have you ever heard about the Cosmic Bonjour.")
    # print time taken with 2 decimal places
    print("Time taken: {:.2f}s".format(time.time() - start))
    print(sentences)
    for sentence in sentences:
        tts.add_sentence(sentence)
    
    tts.wait()

    start = time.time()
    sentences = tg.chat("Well, it's a meme in my group, actually a pretty funny one.")
    # print time taken with 2 decimal places
    print("Time taken: {:.2f}s".format(time.time() - start))
    print(sentences)
    for sentence in sentences:
        tts.add_sentence(sentence)

    tts.wait()
    tts.close()