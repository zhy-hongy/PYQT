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

        self.save_dir = "./saved_AE_images"
        os.makedirs(self.save_dir, exist_ok=True)

        # ---------- 相机相关 ----------
        self.capture = None
        self.current_frame = None
        self.camera_ok = False

        # ---------- 自动调节参数 ----------
        self.auto_adjust = False
        self.target_peak = 220
        self.target_sat = 0.001
        self.alpha = 0.2

        self.brightness_step = 5
        self.gain_step = 2

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

        # ---- HDR 拍摄按钮 ----
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

        # ---- 自动调节参数分组 ----
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

        # ===== 删除 ROI 起始/结束滑块（4、5行已移除） =====

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

    # ===== 移除 on_roi_start_changed / on_roi_end_changed =====

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

            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.0)

            brightness = int(self.capture.get(cv2.CAP_PROP_BRIGHTNESS))
            gain = int(self.capture.get(cv2.CAP_PROP_GAIN))

            self.slider_brightness.setValue(brightness)
            self.slider_gain.setValue(gain)

            self.timer.start(30)
            self.btn_camera.setText("关闭相机")
            self.btn_save.setEnabled(True)
            self.btn_hdr.setEnabled(True)
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
        self.btn_hdr.setEnabled(False)
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

        # ===== 直接使用整张灰度图，不再裁剪 ROI =====
        # 自动调节（亮度为主，增益为辅）
        self.auto_adjust_brightness_gain(gray)

        # ===== 移除绘制绿色矩形的代码 =====

        # 显示
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 显示原始帧（不画框）
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        pix = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

    # ---------- 优化调节速度后的自动调节核心(调节亮度到220等) ----------
    # def auto_adjust_brightness_gain(self, roi_gray):  
    #     # 自适应激光提取
    #     blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    #     _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #     laser_mask = otsu.astype(bool)
    #     laser_pixels = roi_gray[laser_mask]

    #     if len(laser_pixels) < 50:
    #         self.lbl_peak.setText("峰值: 无激光")
    #         self.lbl_sat.setText("饱和: --")
    #         if self.auto_adjust:
    #             # current_bright = self.capture.get(cv2.CAP_PROP_BRIGHTNESS)
    #             new_bright = np.clip(300, 0, 1800)  # 每次增加30
    #             self.capture.set(cv2.CAP_PROP_BRIGHTNESS, float(new_bright))
    #             self.update_slider_brightness(new_bright)
    #             self.lbl_current_bright.setText(f"当前亮度: {new_bright:.1f}")
    #         return

    #     p90 = np.percentile(laser_pixels, 90)
    #     p50 = np.percentile(laser_pixels, 50)
    #     mean = np.mean(laser_pixels)
    #     peak = 0.6 * p90 + 0.3 * mean + 0.1 * p50
    #     peak = np.clip(peak, 0, 255)

    #     sat_ratio = np.mean(laser_pixels >= 250)

    #     self.lbl_peak.setText(f"峰值: {peak:.1f}")
    #     self.lbl_sat.setText(f"饱和: {sat_ratio*100:.3f}%")

    #     if not self.auto_adjust:
    #         return

    #     current_bright = self.capture.get(cv2.CAP_PROP_BRIGHTNESS)
    #     error = self.target_peak - peak

    #     # ========================================================
    #     # 【修改点 1】动态变增益 Kp：误差大时用大步长，误差小时用小步长
    #     # ========================================================
    #     if abs(error) > 40:
    #         kp = 0.4   # 离目标很远，加大比例系数，快速逼近
    #         ki = 0.0   # 大误差时关闭积分，防止超调震荡（积分饱和）
    #     else:
    #         kp = 0.1   # 接近目标了，用小系数精细微调
    #         ki = 0.03

    #     # I 积分项计算（仅在靠近目标时累加）
    #     if not hasattr(self, "err_i"): self.err_i = 0
    #     if abs(error) <= 40:
    #         self.err_i = 0.95 * self.err_i + error * 0.02
    #     else:
    #         self.err_i = 0 # 误差大时清空历史积分

    #     # D 微分项计算
    #     if not hasattr(self, "last_err"): self.last_err = 0
    #     kd = 0.02
    #     d = error - self.last_err
    #     self.last_err = error

    #     # ========================================================
    #     # 【修改点 2】大幅放开单帧限幅（从 6 扩大到 80）
    #     # ========================================================
    #     delta = kp * error + ki * self.err_i + kd * d
    #     delta = np.clip(delta, -80, 80) # 允许单帧最大调整 80，提速 13 倍！
        
    #     desired = current_bright + delta

    #     # ========================================================
    #     # 【修改点 3】根据过曝严重程度，实行“阶梯式强力惩罚”
    #     # ========================================================
    #     if sat_ratio > self.target_sat:
    #         if sat_ratio > 0.05:
    #             desired -= 60  # 严重过曝（超过5%面积死白），极其猛烈地下调
    #         else:
    #             desired -= 15  # 轻微过曝，较大幅度下调

    #     # ========================================================
    #     # 【修改点 4】动态调整平滑系数 alpha（快到目标时才需要平滑）
    #     # ========================================================
    #     # 如果误差很大，强行让 alpha=0.8（甚至1.0），即“立刻响应，不搞平滑缓冲”
    #     actual_alpha = 0.8 if abs(error) > 50 else self.alpha

    #     new_bright = (1 - actual_alpha) * current_bright + actual_alpha * desired
    #     new_bright = np.clip(new_bright, 50, 1800)

    #     self.capture.set(cv2.CAP_PROP_BRIGHTNESS, float(new_bright))
    #     self.update_slider_brightness(new_bright)
    #     self.lbl_current_bright.setText(f"当前亮度: {new_bright:.1f}")
    
    

    
    # ---------- 丝滑稳定版：基于自适应动态门槛的线宽 AE 算法 ----------
    def auto_adjust_brightness_gain(self, roi_gray):  
        # 1. 动态计算每一列的极大值
        col_max = np.max(roi_gray, axis=0)
        
        # 过滤盲区：由于亮度下限改为 10，低光下激光可能较弱，门槛降低到 15 以防丢失暗线
        valid_cols = np.where(col_max > 15)[0] 

        if len(valid_cols) < 20:
            self.lbl_peak.setText("线宽: 无有效信号")
            self.lbl_sat.setText("饱和: --")
            if self.auto_adjust:
                # 盲目搜索状态：找不到线时温和地上拉亮度
                current_bright = self.capture.get(cv2.CAP_PROP_BRIGHTNESS)
                new_bright = np.clip(current_bright + 20, 10, 1800)
                self.capture.set(cv2.CAP_PROP_BRIGHTNESS, float(new_bright))
                self.update_slider_brightness(new_bright)
            return

        # 2. 自适应每列的动态门槛（半高全宽 FWHM 思想，精准提取核心线宽）
        widths = []
        for col_idx in valid_cols:
            max_val = col_max[col_idx]
            # 动态门槛：最低不小于15，或者是当前列最大值的 45%
            dynamic_thresh = max(15, int(max_val * 0.45))
            col_width = np.sum(roi_gray[:, col_idx] >= dynamic_thresh)
            widths.append(col_width)
            
        current_width = np.mean(widths) if len(widths) > 0 else 10.0

        # 安全计算饱和度，防止索引越界
        laser_mask = roi_gray > 15
        sat_ratio = np.mean(roi_gray[laser_mask] >= 252) if np.any(laser_mask) else 0

        self.lbl_peak.setText(f"当前线宽: {current_width:.2f} px")
        self.lbl_sat.setText(f"饱和比例: {sat_ratio*100:.2f}%")

        if not self.auto_adjust:
            return

        # 3. 【修改点 1】目标线宽映射调整为 5 ~ 15 像素
        # 原滑块范围 100 ~ 255，正好映射到 5.0 ~ 15.0 px
        # 映射公式：5.0 + (val - 100) * (10.0 / 155.0)
        target_width = 5.0 + (self.target_peak - 100) * (10.0 / 155.0)
        self.lbl_target_peak.setText(f"{target_width:.1f}px")

        current_bright = self.capture.get(cv2.CAP_PROP_BRIGHTNESS)
        error = target_width - current_width

        # ========================================================
        # 【修改点 2】引入控制死区 + 大幅压低 PID 参数（彻底根治抖动）
        # ========================================================
        if abs(error) < 0.3:
            # 进入死区：线宽误差小于 0.3 像素时，极小幅微调或不调，防止来回摆动
            kp = 1.0
            ki = 0.1
            kd = 0.0
        elif abs(error) > 3.0:
            # 误差很大（比如相差 3 个像素以上），用中等步长快速追赶
            kp = 8.0   
            ki = 0.0
            kd = 1.0
        else:
            # 正常调谐区间，采用极为温和的参数，慢速逼近
            kp = 3.5   
            ki = 0.4
            kd = 1.5

        # I 积分项计算（带抗饱和及遗忘因子）
        if not hasattr(self, "err_i"): self.err_i = 0
        if abs(error) <= 3.0:
            self.err_i = 0.92 * self.err_i + error * 0.05
            self.err_i = np.clip(self.err_i, -30, 30) # 限制积分上限，防止超调
        else:
            self.err_i = 0 # 误差大时清除积分

        # D 微分项计算
        if not hasattr(self, "last_err"): self.last_err = 0
        d = error - self.last_err
        self.last_err = error

        # 计算调整量
        delta = kp * error + ki * self.err_i + kd * d
        
        # 【收紧单帧限幅】从 150 压低到 45，配合柔和调光，单帧不会突变
        delta = np.clip(delta, -45, 45) 
        
        desired = current_bright + delta

        # 过曝硬惩罚温和化（降到低光区时，过曝惩罚要收敛，否则会引发震荡）
        if sat_ratio > 0.03 and error < 0 and current_bright > 100:
            desired -= 15

        # ========================================================
        # 【修改点 3】平滑系数与亮度下限修改为 10
        # ========================================================
        # 误差大时也保持一定的平滑（0.5），接近目标时极其平滑（self.alpha，通常为 0.2）
        actual_alpha = 0.5 if abs(error) > 3.0 else self.alpha

        new_bright = (1 - actual_alpha) * current_bright + actual_alpha * desired
        
        # 将亮度调节下限修改为 10
        new_bright = np.clip(new_bright, 10, 1800) 

        # 4. 下发硬件并刷新 UI
        self.capture.set(cv2.CAP_PROP_BRIGHTNESS, float(new_bright))
        self.update_slider_brightness(new_bright)
        self.lbl_current_bright.setText(f"当前相机亮度: {new_bright:.1f}")
    
    
    
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

    # ---------- HDR 拍摄 ----------
    def capture_hdr(self):
        if self.capture is None or not self.capture.isOpened():
            QMessageBox.warning(self, "警告", "相机未打开")
            return

        old_auto_state = self.auto_adjust
        if old_auto_state:
            self.auto_adjust = False
            self.chk_auto.setChecked(False)

        orig_bright = self.capture.get(cv2.CAP_PROP_BRIGHTNESS)
        orig_gain = self.capture.get(cv2.CAP_PROP_GAIN)

        factors = [0.6, 1.0, 1.4]
        frames = []

        try:
            for factor in factors:
                target_bright = np.clip(orig_bright * factor, 0, 2000)
                self.capture.set(cv2.CAP_PROP_BRIGHTNESS, target_bright)
                time.sleep(0.08)
                ret, frame = self.capture.read()
                if ret:
                    frames.append(frame)
                else:
                    QMessageBox.warning(self, "警告", f"捕获帧失败 (亮度系数 {factor})")
                    break

            if len(frames) < 3:
                QMessageBox.critical(self, "错误", "未能获取足够的帧，HDR 合成中止")
                return

            merge_mertens = cv2.createMergeMertens()
            hdr_result = merge_mertens.process(frames)
            ldr_8bit = np.clip(hdr_result * 255, 0, 255).astype('uint8')
            filename = time.strftime("HDR_%Y%m%d_%H%M%S.png")
            full_path = os.path.join(self.save_dir, filename)
            ok = cv2.imwrite(full_path, ldr_8bit)
            if ok:
                QMessageBox.information(self, "成功", f"HDR 图像已保存:\n{full_path}")
            else:
                QMessageBox.critical(self, "错误", "保存 HDR 图像失败")

        except Exception as e:
            QMessageBox.critical(self, "异常", f"HDR 处理出错: {str(e)}")
        finally:
            self.capture.set(cv2.CAP_PROP_BRIGHTNESS, orig_bright)
            self.capture.set(cv2.CAP_PROP_GAIN, orig_gain)
            if old_auto_state:
                self.auto_adjust = True
                self.chk_auto.setChecked(True)
            self.update_slider_brightness(orig_bright)
            self.update_slider_gain(orig_gain)
            self.lbl_current_bright.setText(f"当前亮度: {orig_bright:.1f}")
            self.lbl_current_gain.setText(f"当前增益: {orig_gain:.1f}")

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