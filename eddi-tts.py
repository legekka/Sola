from modules.stts import TextToSpeechQueue
from modules.textgenerator import TextGeneratorAPIClient
from modules.gui import Gui
import time
import keyboard
import asyncio

watchfile = "C:/Users/legek/AppData/Roaming/EDDI/speechresponder.out"
lines = []


def checkForKeypress():
    global tts
    global gui
    if keyboard.is_pressed("ctrl+*"):
        print("Ctrl+* pressed, exiting...")
        return True
# this function monitors the speechresponder.out file. Another process is updating it with new lines (separated by newlines)
# when a new line is detected, it is added to rephase and then to the tts queue
# if lines is empty, we only add the last line, else every new line if there's multiple
async def mainloop():
    global lines
    global gui
    global tts
    while True:
        if checkForKeypress():
            break
        with open(watchfile, "r") as f:
            newlines = f.readlines()
            if len(newlines) > 0:
                if len(lines) == 0:
                    lines = newlines
                else:
                    # only add the new lines, not everything
                    newlines = newlines[len(lines):]
                    for line in newlines:
                        # make sure we don't add empty lines
                        if line.strip() == "":
                            continue
                        lines.append(line)
                        print("New line: " + line)
                        sentences = await tg.chat(line)
                        print("Rephrased: " + str(sentences))
                        tts.add_sentences(sentences)
        if gui is not None:
            if tts.new_event is not None:
                gui.reset()
                gui.sleep(500)
                gui.display_message(tts.new_event["text"], tts.new_event["audio_length"])
                tts.new_event = None
            else:
                gui.sleep(250)
        else:
            time.sleep(0.25)
        
def initGui():
    global gui
    global app
    from PySide6.QtWidgets import QApplication
    import sys
    global tts
    
    app = QApplication(sys.argv)
    gui = Gui()
    gui.init_ui()
    gui.show()

async def main():
    global tg
    global gui
    global tts

    print("Initializing TextGenerator API Client...")
    tg = TextGeneratorAPIClient(host="192.168.1.20")
    
    print("Initializing GUI...")
    initGui()

    tts = TextToSpeechQueue("Samples/female_reference2.wav")
    tts.init()

    print("Starting TTS Queue...")
    tts.run()

    print("Warming up...")
    sentences = await tg.rephrase("Initializing ship systems, preparing to launch.")
    print(sentences)
    for sentence in sentences:
        tts.add_sentence(sentence)

    tts.wait()

    print("Start!")
    await mainloop()

    tts.close()
    if gui is not None:
        gui.close()

    print("End.")
    exit()

if __name__ == "__main__":
    asyncio.run(main())