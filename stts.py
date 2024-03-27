from modules.stts import TextToSpeechQueue

tts = TextToSpeechQueue(ref_audio_path="Samples/Ganyu3.wav", model_path="Models/LJSpeech_Ganyu/epoch_2nd_00004.pth", config_path="Models/LJSpeech_Ganyu/config.yml")
print("__init__ done.")
tts.init()
print("init done.")
tts.run()
print("run done.")

tts.add_sentence("Sola is getting all the spaceship systems ready, and preparing for a big, exciting launch!")
print("synthetize done.")
tts.wait()
import time
time.sleep(10)
tts.close()