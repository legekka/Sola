from modules.xtts import TextToSpeechThreaded

if __name__ == "__main__":
    tts = TextToSpeechThreaded("C:/Users/legek/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2/", "Samples/Ganyu2.wav")
    tts.init()
    tts.prepare()

    # Async mode
    tts.synthetize_async("Hey Commander, Sola heard that the exploration data for 18 systems was quite valuable! It sold for more than 77 million credits. ", "en")
    # do something else
    tts.join()  # wait for the synthesis to finish

    # Sync mode
    tts.synthetize("In addition to this, there was a special reward of 174,000 credits given to those who made their first discoveries in these systems.", "en")
    tts.synthetize("Isn't it exciting?", "en")

    tts.close()