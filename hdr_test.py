# -*- coding: utf-8 -*-
import os
import sys
import time
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout,
    QCheckBox, QDoubleSpinBox, QGroupBox
)


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UVC相机控制（自动亮度/增益 + HDR）")
        self.resize(1200, 700)

        self.save_dir = "./saved_hdr_images"
        os.makedirs(self.save_dir, exist_ok=True)

        # ---------- 相机相关 ----------
        self.capture = None
        self.current_frame = None
        self.camera_ok = False

        # ---------- 自动调节参数 ----------
        self.auto_adjust = False          # 是否启用自动
        self.target_peak = 220            # 目标峰值
        self.target_sat = 0.001           # 饱和比例上限
        self.alpha = 0.2                  # 平滑系数
        self.roi_start = 0.0
        self.roi_end = 1.0

        # 亮度与增益的调节步长（增量式调节用）
        self.brightness_step = 5
        self.gain_step = 2

        # ---------- 定时器 ----------
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.reconnect_timer = QTimer()
        self.reconnect_timer.timeout.connect(self.try_reconnect)

        self.init_ui()

    # ---------- UI 构建 ----------
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)

        # ==========================
        # 图像显示区
        # ==========================
        self.image_label = QLabel("点击「打开相机」")
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

        # ---- 相机控制 ----
        self.btn_camera = QPushButton("打开相机")
        self.btn_camera.setFixedHeight(40)
        self.btn_camera.clicked.connect(self.toggle_camera)
        control_layout.addWidget(self.btn_camera)

        # ---- 保存图片 ----
        self.btn_save = QPushButton("保存图片")
        self.btn_save.setFixedHeight(40)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False)
        control_layout.addWidget(self.btn_save)

        # ---- 【新增】HDR 拍摄按钮 ----
        self.btn_hdr = QPushButton("HDR 拍摄 (融合3帧)")
        self.btn_hdr.setFixedHeight(40)
        self.btn_hdr.clicked.connect(self.capture_hdr)
        self.btn_hdr.setEnabled(False)
        control_layout.addWidget(self.btn_hdr)

        control_layout.addSpacing(10)

        # ---- 手动控制分组 ----
        group_manual = QGroupBox("手动控制")
        manual_grid = QGridLayout()

        # 亮度
        manual_grid.addWidget(QLabel("亮度"), 0, 0)
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(0, 2000)
        self.slider_brightness.valueChanged.connect(self.change_brightness)
        manual_grid.addWidget(self.slider_brightness, 0, 1)
        self.lbl_brightness = QLabel("0")
        manual_grid.addWidget(self.lbl_brightness, 0, 2)

        # 增益
        manual_grid.addWidget(QLabel("增益"), 1, 0)
        self.slider_gain = QSlider(Qt.Horizontal)
        self.slider_gain.setRange(0, 255)
        self.slider_gain.valueChanged.connect(self.change_gain)
        manual_grid.addWidget(self.slider_gain, 1, 1)
        self.lbl_gain = QLabel("0")
        manual_grid.addWidget(self.lbl_gain, 1, 2)

        group_manual.setLayout(manual_grid)
        control_layout.addWidget(group_manual)

        # ---- 自动调节参数分组（亮度为主，增益为辅） ----
        group_auto = QGroupBox("自动调节（亮度为主，增益为辅）")
        auto_grid = QGridLayout()

        # 启用复选框
        self.chk_auto = QCheckBox("启用自动调节（亮度优先）")
        self.chk_auto.stateChanged.connect(lambda s: setattr(self, 'auto_adjust', s == Qt.Checked))
        auto_grid.addWidget(self.chk_auto, 0, 0, 1, 3)

        # 目标峰值
        auto_grid.addWidget(QLabel("目标峰值"), 1, 0)
        self.slider_target_peak = QSlider(Qt.Horizontal)
        self.slider_target_peak.setRange(100, 255)
        self.slider_target_peak.setValue(220)
        self.slider_target_peak.valueChanged.connect(self.on_target_peak_changed)
        auto_grid.addWidget(self.slider_target_peak, 1, 1)
        self.lbl_target_peak = QLabel("220")
        auto_grid.addWidget(self.lbl_target_peak, 1, 2)

        # 饱和比例上限
        auto_grid.addWidget(QLabel("饱和比例上限"), 2, 0)
        self.spin_target_sat = QDoubleSpinBox()
        self.spin_target_sat.setRange(0.0, 0.05)
        self.spin_target_sat.setSingleStep(0.0005)
        self.spin_target_sat.setDecimals(4)
        self.spin_target_sat.setValue(0.001)
        self.spin_target_sat.valueChanged.connect(lambda v: setattr(self, 'target_sat', v))
        auto_grid.addWidget(self.spin_target_sat, 2, 1, 1, 2)

        # 平滑系数 α
        auto_grid.addWidget(QLabel("平滑系数 α"), 3, 0)
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(5, 90)
        self.slider_alpha.setValue(20)
        self.slider_alpha.valueChanged.connect(self.on_alpha_changed)
        auto_grid.addWidget(self.slider_alpha, 3, 1)
        self.lbl_alpha = QLabel("0.20")
        auto_grid.addWidget(self.lbl_alpha, 3, 2)

        # ROI 起始
        auto_grid.addWidget(QLabel("ROI 起始 %"), 4, 0)
        self.slider_roi_start = QSlider(Qt.Horizontal)
        self.slider_roi_start.setRange(0, 90)
        self.slider_roi_start.setValue(0)
        self.slider_roi_start.valueChanged.connect(self.on_roi_start_changed)
        auto_grid.addWidget(self.slider_roi_start, 4, 1)
        self.lbl_roi_start = QLabel("0%")
        auto_grid.addWidget(self.lbl_roi_start, 4, 2)

        # ROI 结束
        auto_grid.addWidget(QLabel("ROI 结束 %"), 5, 0)
        self.slider_roi_end = QSlider(Qt.Horizontal)
        self.slider_roi_end.setRange(10, 100)
        self.slider_roi_end.setValue(100)
        self.slider_roi_end.valueChanged.connect(self.on_roi_end_changed)
        auto_grid.addWidget(self.slider_roi_end, 5, 1)
        self.lbl_roi_end = QLabel("100%")
        auto_grid.addWidget(self.lbl_roi_end, 5, 2)

        group_auto.setLayout(auto_grid)
        control_layout.addWidget(group_auto)

        # ---- 状态显示 ----
        self.lbl_peak = QLabel("峰值: --")
        self.lbl_sat = QLabel("饱和: --%")
        self.lbl_current_bright = QLabel("当前亮度: --")
        self.lbl_current_gain = QLabel("当前增益: --")
        self.lbl_camera_status = QLabel("状态: 未连接")
        control_layout.addWidget(self.lbl_peak)
        control_layout.addWidget(self.lbl_sat)
        control_layout.addWidget(self.lbl_current_bright)
        control_layout.addWidget(self.lbl_current_gain)
        control_layout.addWidget(self.lbl_camera_status)

        # ---- 驱动设置 ----
        self.btn_setting = QPushButton("相机驱动设置")
        self.btn_setting.clicked.connect(self.open_camera_setting)
        control_layout.addWidget(self.btn_setting)

        control_layout.addStretch()

    # ---------- 参数回调 ----------
    def on_target_peak_changed(self, val):
        self.target_peak = val
        self.lbl_target_peak.setText(str(val))

    def on_alpha_changed(self, val):
        self.alpha = val / 100.0
        self.lbl_alpha.setText(f"{self.alpha:.2f}")

    def on_roi_start_changed(self, val):
        self.roi_start = val / 100.0
        self.lbl_roi_start.setText(f"{val}%")
        if self.roi_start >= self.roi_end - 0.05:
            new_end = min(int((self.roi_start + 0.05) * 100), 100)
            self.slider_roi_end.setValue(new_end)

    def on_roi_end_changed(self, val):
        self.roi_end = val / 100.0
        self.lbl_roi_end.setText(f"{val}%")
        if self.roi_end <= self.roi_start + 0.05:
            new_start = max(int((self.roi_end - 0.05) * 100), 0)
            self.slider_roi_start.setValue(new_start)

    # ---------- 相机连接 ----------
    def toggle_camera(self):
        if not self.timer.isActive():
            self.open_camera()
        else:
            self.close_camera()

    def open_camera(self):
        try:
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                raise Exception("无法打开相机")

            # 尝试关闭自动曝光（某些相机需设为0或1）
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)

            # 读取当前参数
            brightness = int(self.capture.get(cv2.CAP_PROP_BRIGHTNESS))
            gain = int(self.capture.get(cv2.CAP_PROP_GAIN))
            exp = int(self.capture.get(cv2.CAP_PROP_EXPOSURE))

            self.slider_brightness.setValue(brightness)
            self.slider_gain.setValue(gain)
            # self.slider_exp.setValue(exp)

            self.timer.start(30)
            self.btn_camera.setText("关闭相机")
            self.btn_save.setEnabled(True)
            self.btn_hdr.setEnabled(True)      # 【新增】启用HDR按钮
            self.lbl_camera_status.setText("状态: 已连接")
            self.camera_ok = True
            self.reconnect_timer.stop()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开相机失败: {str(e)}")
            self.camera_ok = False
            self.lbl_camera_status.setText("状态: 连接失败")

    def close_camera(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
            self.capture = None
        self.btn_camera.setText("打开相机")
        self.btn_save.setEnabled(False)
        self.btn_hdr.setEnabled(False)        # 【新增】禁用HDR按钮
        self.lbl_camera_status.setText("状态: 已断开")
        self.camera_ok = False
        self.image_label.clear()
        self.image_label.setText("相机已关闭")

    def try_reconnect(self):
        if not self.camera_ok and not self.timer.isActive():
            self.open_camera()
            if self.camera_ok:
                self.reconnect_timer.stop()

    # ---------- 帧更新 ----------
    def update_frame(self):
        if self.capture is None:
            return

        ret, frame = self.capture.read()
        if not ret:
            self.camera_ok = False
            self.lbl_camera_status.setText("状态: 已断开 (等待重连)")
            self.timer.stop()
            self.reconnect_timer.start(2000)
            return

        self.current_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # ROI
        y_start = int(h * self.roi_start)
        y_end = int(h * self.roi_end)
        y_start = max(0, y_start)
        y_end = min(h, y_end)
        if y_end - y_start < 10:
            y_end = y_start + 10

        roi_gray = gray[y_start:y_end, :]

        # 自动调节（亮度为主，增益为辅）
        self.auto_adjust_brightness_gain(roi_gray)

        # 绘制 ROI 矩形
        frame_draw = frame.copy()
        cv2.rectangle(frame_draw, (0, y_start), (w-1, y_end-1), (0, 255, 0), 2)

        # 显示
        rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        pix = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

    # ---------- 自动调节核心（亮度优先） ----------
    def auto_adjust_brightness_gain(self, roi_gray):
        laser_pixels = roi_gray[roi_gray > 30]
        if len(laser_pixels) < 100:
            self.lbl_peak.setText("峰值: 无激光")
            self.lbl_sat.setText("饱和: --")
            return

        # ----- 修改点1：改用90%分位数 -----
        peak = np.percentile(laser_pixels, 90)   # 原为99.9
        sat_ratio = np.mean(laser_pixels >= 250)
        self.lbl_peak.setText(f"峰值: {peak:.1f}")
        self.lbl_sat.setText(f"饱和: {sat_ratio*100:.3f}%")

        if not self.auto_adjust:
            return

        current_bright = self.capture.get(cv2.CAP_PROP_BRIGHTNESS)
        current_gain = self.capture.get(cv2.CAP_PROP_GAIN)

        # ----- 修改点2：增加死区 -----
        deadzone = 20   # 可做成滑块，暂时固定
        if abs(peak - self.target_peak) <= deadzone and sat_ratio <= self.target_sat:
            # 峰值已在目标附近且未过饱和，不调节
            return

        # ----- 修改点3：饱和抑制平滑化 -----
        if sat_ratio > self.target_sat:
            # 计算目标亮度和增益（降低10%）
            new_bright = current_bright * 0.9
            new_gain = current_gain * 0.9
            # 限幅
            new_bright = np.clip(new_bright, 0, 2000)
            new_gain = np.clip(new_gain, 0, 255)
            # 应用平滑（这里也使用α，防止跳变）
            new_bright = (1 - self.alpha) * current_bright + self.alpha * new_bright
            new_gain = (1 - self.alpha) * current_gain + self.alpha * new_gain
            self.capture.set(cv2.CAP_PROP_BRIGHTNESS, new_bright)
            self.capture.set(cv2.CAP_PROP_GAIN, new_gain)
            self.update_slider_brightness(new_bright)
            self.update_slider_gain(new_gain)
            self.lbl_current_bright.setText(f"当前亮度: {new_bright:.1f}")
            self.lbl_current_gain.setText(f"当前增益: {new_gain:.1f}")
            return

        # ----- 修改点4：放宽比例因子限幅 -----
        ratio = self.target_peak / max(peak, 1.0)
        ratio = np.clip(ratio, 0.8, 1.25)   # 原为0.85~1.15

        desired_bright = current_bright * ratio
        if 0 <= desired_bright <= 2000:
            new_bright = (1 - self.alpha) * current_bright + self.alpha * desired_bright
            new_bright = np.clip(new_bright, 0, 2000)
            new_gain = current_gain
        else:
            # 亮度边界处理
            if desired_bright < 0:
                new_bright = 0
                # 增益调节量按比例折算
                gain_ratio = 1 - (1 - ratio) * 0.5
                new_gain = current_gain * gain_ratio
            else:
                new_bright = 2000
                gain_ratio = 1 + (ratio - 1) * 0.5
                new_gain = current_gain * gain_ratio
            new_gain = np.clip(new_gain, 0, 255)

        # 对增益也做平滑
        if new_gain != current_gain:
            new_gain = (1 - self.alpha) * current_gain + self.alpha * new_gain
            new_gain = np.clip(new_gain, 0, 255)

        self.capture.set(cv2.CAP_PROP_BRIGHTNESS, new_bright)
        self.capture.set(cv2.CAP_PROP_GAIN, new_gain)
        self.update_slider_brightness(new_bright)
        self.update_slider_gain(new_gain)
        self.lbl_current_bright.setText(f"当前亮度: {new_bright:.1f}")
        self.lbl_current_gain.setText(f"当前增益: {new_gain:.1f}")

    # 辅助：更新滑块（阻塞信号）
    def update_slider_brightness(self, value):
        int_val = int(round(value))
        self.slider_brightness.blockSignals(True)
        self.slider_brightness.setValue(int_val)
        self.lbl_brightness.setText(str(int_val))
        self.slider_brightness.blockSignals(False)

    def update_slider_gain(self, value):
        int_val = int(round(value))
        self.slider_gain.blockSignals(True)
        self.slider_gain.setValue(int_val)
        self.lbl_gain.setText(str(int_val))
        self.slider_gain.blockSignals(False)

    # ---------- 手动控制回调 ----------
    def change_brightness(self, value):
        self.lbl_brightness.setText(str(value))
        if self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_BRIGHTNESS, value)

    def change_gain(self, value):
        self.lbl_gain.setText(str(value))
        if self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_GAIN, value)

    # ---------- 【新增】HDR 拍摄核心方法 ----------
    
    def capture_hdr(self):
        """改进版HDR：基于当前亮度做绝对值偏移，确保覆盖高光和暗部"""
        if self.capture is None or not self.capture.isOpened():
            QMessageBox.warning(self, "警告", "相机未打开")
            return

        # 1. 保存状态并暂时关闭自动调节
        old_auto_state = self.auto_adjust
        if old_auto_state:
            self.auto_adjust = False
            self.chk_auto.setChecked(False)

        # 读取当前用户设定的“基准值”
        base_bright = self.capture.get(cv2.CAP_PROP_BRIGHTNESS)
        base_gain = self.capture.get(cv2.CAP_PROP_GAIN)

        # 2. 定义“绝对值步进”策略（不再用乘法！）
        # 亮度范围 0~2000，增益范围 0~255
        # 这里以当前值为中心，向两端拓展
        bright_offsets = [-400, -150, 0, 200, 600]   # 相对于基准值的加减量
        gain_factors = [0.6, 0.9, 1.0, 1.3, 1.8]    # 增益配合调整（增益越大，暗部越明显）
        
        # 为了应对极端情况（一边极亮一边极暗），再额外补两帧极端值
        # 如果基准值本身很高，强制补一帧极低亮度；如果基准值很低，补一帧极高亮度
        extra_configs = []
        if base_bright > 1000:
            extra_configs.append((100, 0.5))   # 强制捕获极暗帧保留高光
        if base_bright < 500:
            extra_configs.append((1800, 2.5))  # 强制捕获极亮帧提取暗部

        frames = []
        try:
            # 处理常规偏移
            for offset, gf in zip(bright_offsets, gain_factors):
                target_bright = np.clip(base_bright + offset, 0, 2000)
                target_gain = np.clip(base_gain * gf, 0, 255)
                
                self.capture.set(cv2.CAP_PROP_BRIGHTNESS, target_bright)
                self.capture.set(cv2.CAP_PROP_GAIN, target_gain)
                time.sleep(0.1)  # 等待稳定
                
                ret, frame = self.capture.read()
                if ret:
                    frames.append(frame)

            # 处理额外的极端帧
            for b_val, g_val in extra_configs:
                target_bright = np.clip(b_val, 0, 2000)
                target_gain = np.clip(base_gain * g_val, 0, 255)
                self.capture.set(cv2.CAP_PROP_BRIGHTNESS, target_bright)
                self.capture.set(cv2.CAP_PROP_GAIN, target_gain)
                time.sleep(0.1)
                ret, frame = self.capture.read()
                if ret:
                    frames.append(frame)

            if len(frames) < 3:
                QMessageBox.critical(self, "错误", f"仅捕获 {len(frames)} 帧，中止")
                return

            # 3. 融合（使用 Mertens）
            merge_mertens = cv2.createMergeMertens()
            hdr_result = merge_mertens.process(frames)

            # 4. 后处理：自动对比度拉伸 + 轻微锐化，让激光线更清晰
            ldr_8bit = np.clip(hdr_result * 255, 0, 255).astype('uint8')
            # 使用 CLAHE（限制对比度自适应直方图均衡）增强暗部激光线
            lab = cv2.cvtColor(ldr_8bit, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            ldr_8bit = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 保存
            filename = time.strftime("HDR_%Y%m%d_%H%M%S.png")
            full_path = os.path.join(self.save_dir, filename)
            cv2.imwrite(full_path, ldr_8bit)
            QMessageBox.information(self, "成功", f"HDR 已保存:\n{full_path}")

        except Exception as e:
            QMessageBox.critical(self, "异常", f"出错: {str(e)}")
        finally:
            # 5. 恢复基准值
            self.capture.set(cv2.CAP_PROP_BRIGHTNESS, base_bright)
            self.capture.set(cv2.CAP_PROP_GAIN, base_gain)
            if old_auto_state:
                self.auto_adjust = True
                self.chk_auto.setChecked(True)
            self.update_slider_brightness(base_bright)
            self.update_slider_gain(base_gain)
    
    
    
    
    # ---------- 保存图片 ----------
    def save_image(self):
        if self.current_frame is None:
            return
        filename = time.strftime("IMG_%Y%m%d_%H%M%S.png")
        full_path = os.path.join(self.save_dir, filename)
        ok = cv2.imwrite(full_path, self.current_frame)
        if ok:
            QMessageBox.information(self, "成功", f"已保存:\n{full_path}")
        else:
            QMessageBox.critical(self, "错误", "保存失败")

    # ---------- 驱动设置 ----------
    def open_camera_setting(self):
        if self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_SETTINGS, 0)

    # ---------- 关闭 ----------
    def closeEvent(self, event):
        self.timer.stop()
        self.reconnect_timer.stop()
        if self.capture:
            self.capture.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CameraApp()
    win.show()
    sys.exit(app.exec_())