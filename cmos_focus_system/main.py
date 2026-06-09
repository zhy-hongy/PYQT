import sys
from PyQt5.QtWidgets import QApplication

from camera import Camera
from ui import MainUI

app = QApplication(sys.argv)

win = MainUI()
win.show()

cam = Camera(0)

cam.frame.connect(win.update)

cam.start()

sys.exit(app.exec_())