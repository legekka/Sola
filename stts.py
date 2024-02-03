from modules.stts import TextToSpeechQueue

tts = TextToSpeechQueue(ref_audio_path="Samples/Ganyu.wav")
print("__init__ done.")
tts.init()
print("init done.")
tts.run()
print("run done.")

tts.add_sentence("They had me no choice but to get my Tatoo.")
print("synthetize done.")
tts.wait()
tts.close()