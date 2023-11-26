import time

from modules.gui import Gui
from PySide6.QtWidgets import QApplication
import sys


if __name__ == "__main__":
    app = QApplication(sys.argv)

    gui = Gui()
    gui.show()

    gui.display_message("New message", 3000)

    print("this is still async")
    # exec loop
    gui.wait()
    print("this is after the animations")

    gui.display_message("Hey Commander! I've just received some new information about brown dwarf 5 - it seems to be quite uncommon and even more remarkable because it has a rocky ring around it!", 13000)
    gui.wait()
    print("end")
    exit()
    sys.exit(app.exec())
