#-*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QSlider, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QMessageBox)

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("USB 相机控制系统 (无损亮度调节版)")
        self.setGeometry(100, 100, 1000, 650)

        # 确保创建保存路径
        self.save_dir = "./saved_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 初始化变量
        self.current_frame = None  
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.initUI()
        self.init_camera_values()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧：视频显示区域
        self.image_label = QLabel("点击“打开相机”开始采集")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white; min-width: 640px; min-height: 480px;")
        main_layout.addWidget(self.image_label, stretch=4)

        # 右侧：控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel, stretch=1)

        # 1. 开关相机按钮
        self.btn_toggle = QPushButton("打开相机")
        self.btn_toggle.setStyleSheet("height: 40px; font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_toggle.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.btn_toggle)

        # 2. 保存图片按钮
        self.btn_capture = QPushButton("📸 保存当前图片")
        self.btn_capture.setStyleSheet("height: 40px; font-weight: bold; background-color: #008CBA; color: white;")
        self.btn_capture.clicked.connect(self.save_image)
        self.btn_capture.setEnabled(False) 
        control_layout.addWidget(self.btn_capture)

        control_layout.addSpacing(20)

        # 参数调节滑块布局
        slider_layout = QGridLayout()
        
        # 1. 亮度（改为相对调节：-150 到 +150，默认 0 表示不改变原有清晰度）
        slider_layout.addWidget(QLabel("亮度增减:"), 0, 0)
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(0, 9999)  
        self.slider_brightness.setValue(0) # 默认为0
        self.slider_brightness.valueChanged.connect(self.change_brightness)
        slider_layout.addWidget(self.slider_brightness, 0, 1)
        self.lbl_brightness = QLabel("0")
        slider_layout.addWidget(self.lbl_brightness, 0, 2)

        # 2. 曝光时间 Exposure
        slider_layout.addWidget(QLabel("曝光时间:"), 1, 0)
        self.slider_exposure = QSlider(Qt.Horizontal)
        self.slider_exposure.setRange(-13, -1) 
        self.slider_exposure.valueChanged.connect(self.change_exposure)
        slider_layout.addWidget(self.slider_exposure, 1, 1)
        self.lbl_exposure = QLabel("0")
        slider_layout.addWidget(self.lbl_exposure, 1, 2)

        # 3. 增益 Gain
        slider_layout.addWidget(QLabel("曝光增益:"), 2, 0)
        self.slider_gain = QSlider(Qt.Horizontal)
        self.slider_gain.setRange(0, 255)
        self.slider_gain.valueChanged.connect(self.change_gain)
        slider_layout.addWidget(self.slider_gain, 2, 1)
        self.lbl_gain = QLabel("0")
        slider_layout.addWidget(self.lbl_gain, 2, 2)

        control_layout.addLayout(slider_layout)
        control_layout.addStretch() 

    def init_camera_values(self):
        """让硬件保持其原本的、最清晰的默认值"""
        if self.capture.isOpened():
            # 仅在必要时关闭硬件级别的自动曝光，防止硬件冲突
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 

            e = int(self.capture.get(cv2.CAP_PROP_EXPOSURE))
            g = int(self.capture.get(cv2.CAP_PROP_GAIN))

            if self.slider_exposure.minimum() <= e <= self.slider_exposure.maximum():
                self.slider_exposure.setValue(e)
            if self.slider_gain.minimum() <= g <= self.slider_gain.maximum():
                self.slider_gain.setValue(g)

    def toggle_camera(self):
        if not self.timer.isActive():
            if not self.capture.isOpened():
                self.capture.open(0, cv2.CAP_DSHOW)
            self.timer.start(30) 
            self.btn_toggle.setText("关闭相机")
            self.btn_toggle.setStyleSheet("height: 40px; font-weight: bold; background-color: #f44336; color: white;")
            self.btn_capture.setEnabled(True) 
        else:
            self.timer.stop()
            self.capture.release()
            self.image_label.clear()
            self.image_label.setText("相机已关闭")
            self.btn_toggle.setText("打开相机")
            self.btn_toggle.setStyleSheet("height: 40px; font-weight: bold; background-color: #4CAF50; color: white;")
            self.btn_capture.setEnabled(False) 

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # 【核心修改点】：通过数字图像处理算法进行无损软件级亮度增减
            brightness_offset = self.slider_brightness.value()
            if brightness_offset != 0:
                # 使用 convertScaleAbs 动态调整图像亮度偏移量 beta，同时自动防止像素越界(0-255)
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness_offset)

            # 备份修改后的清晰画面用于保存
            self.current_frame = frame.copy()

            # 转换为 PyQt 格式并在界面展示
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(QPixmap.fromImage(p))

    def save_image(self):
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"IMG_{timestamp}.jpg"
            full_path = os.path.join(self.save_dir, filename)
            
            success = cv2.imwrite(full_path, self.current_frame)
            if success:
                QMessageBox.information(self, "成功", f"图片已成功保存至:\n{full_path}")
            else:
                QMessageBox.critical(self, "错误", "图片保存失败，请检查文件夹权限！")

    def change_brightness(self, value):
        # 仅更新界面数字标签，不再写入底层硬件，避免模糊
        self.lbl_brightness.setText(f"{'+' if value > 0 else ''}{value}")

    def change_exposure(self, value):
        self.lbl_exposure.setText(str(value))
        if self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
            self.capture.set(cv2.CAP_PROP_EXPOSURE, value)

    def change_gain(self, value):
        self.lbl_gain.setText(str(value))
        if self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_GAIN, value)

    def closeEvent(self, event):
        self.timer.stop()
        if self.capture.isOpened():
            self.capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    th = CameraApp()
    th.show()
    sys.exit(app.exec_())