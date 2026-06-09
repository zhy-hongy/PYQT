import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class Camera(QThread):
    frame = pyqtSignal(object)

    def __init__(self, idx=0):
        super().__init__()
        self.cap = cv2.VideoCapture(idx)

    def run(self):
        while True:
            ret, img = self.cap.read()
            if ret:
                self.frame.emit(img)