# -*- coding: utf-8 -*-
import os
import sys
import time
import cv2

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
)


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UVC相机控制")
        self.resize(1000, 650)

        self.save_dir = "./saved_images"
        os.makedirs(self.save_dir, exist_ok=True)

        self.capture = None
        self.current_frame = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)

        # ==========================
        # 图像显示区
        # ==========================
        self.image_label = QLabel("点击打开相机")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background:black;
            color:white;
            min-width:640px;
            min-height:480px;
        """)

        main_layout.addWidget(self.image_label, stretch=4)

        # ==========================
        # 控制区
        # ==========================
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        main_layout.addWidget(control_widget, stretch=1)

        # 打开关闭相机
        self.btn_camera = QPushButton("打开相机")
        self.btn_camera.setFixedHeight(40)
        self.btn_camera.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.btn_camera)

        # 保存图片
        self.btn_save = QPushButton("保存图片")
        self.btn_save.setFixedHeight(40)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        control_layout.addWidget(self.btn_save)

        control_layout.addSpacing(20)

        grid = QGridLayout()

        # ==========================
        # 亮度
        # ==========================
        grid.addWidget(QLabel("亮度"), 0, 0)

        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(0, 2000)
        self.slider_brightness.valueChanged.connect(
            self.change_brightness
        )

        grid.addWidget(self.slider_brightness, 0, 1)

        self.lbl_brightness = QLabel("0")
        grid.addWidget(self.lbl_brightness, 0, 2)

        # ==========================
        # 增益
        # ==========================
        grid.addWidget(QLabel("增益"), 1, 0)

        self.slider_gain = QSlider(Qt.Horizontal)
        self.slider_gain.setRange(0, 255)
        self.slider_gain.valueChanged.connect(
            self.change_gain
        )

        grid.addWidget(self.slider_gain, 1, 1)

        self.lbl_gain = QLabel("0")
        grid.addWidget(self.lbl_gain, 1, 2)

        control_layout.addLayout(grid)

        # 调驱动设置页
        self.btn_setting = QPushButton("相机驱动设置")
        self.btn_setting.clicked.connect(
            self.open_camera_setting
        )
        control_layout.addWidget(self.btn_setting)

        control_layout.addStretch()

    # =====================================================
    # 打开关闭相机
    # =====================================================
    def toggle_camera(self):

        if not self.timer.isActive():

            self.capture = cv2.VideoCapture(
                0,
                cv2.CAP_DSHOW
            )

            if not self.capture.isOpened():
                QMessageBox.critical(
                    self,
                    "错误",
                    "无法打开相机"
                )
                return

            # 读取当前硬件参数
            brightness = int(
                self.capture.get(
                    cv2.CAP_PROP_BRIGHTNESS
                )
            )

            gain = int(
                self.capture.get(
                    cv2.CAP_PROP_GAIN
                )
            )

            print("Brightness =", brightness)
            print("Gain =", gain)

            self.slider_brightness.setValue(brightness)
            self.slider_gain.setValue(gain)

            self.timer.start(30)

            self.btn_camera.setText("关闭相机")
            self.btn_save.setEnabled(True)

        else:

            self.timer.stop()

            if self.capture:
                self.capture.release()

            self.image_label.clear()
            self.image_label.setText("相机已关闭")

            self.btn_camera.setText("打开相机")
            self.btn_save.setEnabled(False)

    # =====================================================
    # 实时显示
    # =====================================================
    def update_frame(self):

        if self.capture is None:
            return

        ret, frame = self.capture.read()

        if not ret:
            return

        self.current_frame = frame.copy()

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        h, w, ch = rgb.shape

        img = QImage(
            rgb.data,
            w,
            h,
            ch * w,
            QImage.Format_RGB888
        )

        pix = QPixmap.fromImage(img)

        pix = pix.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(pix)

    # =====================================================
    # 亮度
    # =====================================================
    def change_brightness(self, value):

        self.lbl_brightness.setText(str(value))

        if self.capture and self.capture.isOpened():

            self.capture.set(
                cv2.CAP_PROP_BRIGHTNESS,
                value
            )

    # =====================================================
    # 增益
    # =====================================================
    def change_gain(self, value):

        self.lbl_gain.setText(str(value))

        if self.capture and self.capture.isOpened():

            self.capture.set(
                cv2.CAP_PROP_GAIN,
                value
            )

    # =====================================================
    # 保存图片
    # =====================================================
    def save_image(self):

        if self.current_frame is None:
            return

        filename = time.strftime(
            "IMG_%Y%m%d_%H%M%S.png"
        )

        full_path = os.path.join(
            self.save_dir,
            filename
        )

        ok = cv2.imwrite(
            full_path,
            self.current_frame
        )

        if ok:
            QMessageBox.information(
                self,
                "成功",
                f"已保存:\n{full_path}"
            )
        else:
            QMessageBox.critical(
                self,
                "错误",
                "保存失败"
            )

    # =====================================================
    # 打开驱动属性页
    # =====================================================
    def open_camera_setting(self):

        if self.capture and self.capture.isOpened():

            self.capture.set(
                cv2.CAP_PROP_SETTINGS,
                0
            )

    # =====================================================
    # 退出
    # =====================================================
    def closeEvent(self, event):

        self.timer.stop()

        if self.capture:
            self.capture.release()

        event.accept()


if __name__ == "__main__":

    app = QApplication(sys.argv)

    win = CameraApp()
    win.show()

    sys.exit(app.exec_())