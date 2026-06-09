import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap

from vision import detect_cross_center, compute_orientation, focus_analysis


class MainUI(QWidget):

    def __init__(self):
        super().__init__()

        self.label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def update(self, frame):

        img = frame.copy()

        # =========================
        # CMOS中心检测
        # =========================
        center, lines = detect_cross_center(img)

        if center is not None:

            cx, cy = center
            h, w = img.shape[:2]

            dx = cx - w/2
            dy = cy - h/2

            cv2.circle(img, (int(cx),int(cy)), 8, (0,255,0), 2)
            cv2.circle(img, (w//2, h//2), 6, (255,0,0), 2)

            cv2.line(img, (w//2,h//2), (int(cx),int(cy)), (0,255,255), 2)

            cv2.putText(img,
                        f"CMOS dx:{dx:.2f} dy:{dy:.2f}",
                        (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,255),2)

            # =========================
            # 旋转误差
            # =========================
            if lines:
                angle = compute_orientation(lines[0])
                cv2.putText(img,
                            f"tilt:{angle:.2f} deg",
                            (30,60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,(0,200,255),2)

        # =========================
        # 对焦分析
        # =========================
        focus = focus_analysis(img)

        y = 90
        ok = True

        for k,v in focus.items():

            color = (0,255,0) if v > 50 else (0,0,255)

            if v < 50:
                ok = False

            cv2.putText(img,
                        f"{k}:{v:.1f}",
                        (30,y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,2)

            y += 25

        status = "FOCUS OK" if ok else "FOCUS BAD"

        cv2.putText(img,
                    status,
                    (30,y+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0) if ok else (0,0,255),
                    2)

        # =========================
        # 显示
        # =========================
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h,w,ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)

        self.label.setPixmap(QPixmap.fromImage(qimg))