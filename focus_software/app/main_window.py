# -*- coding: utf-8 -*-
"""
对焦检测软件主窗口 - 支持模糊检测和圆形直径检测
增强功能：
1. 帧间稳定性处理（连续多帧确认 + 缺失时沿用上一帧）
2. 实时直径曲线显示
3. 命令行参数支持
"""

from __future__ import annotations

import sys
import argparse
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSpinBox, QStatusBar,
    QVBoxLayout, QWidget, QCheckBox, QDialog,
    QScrollArea  # 添加滚动区域支持
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from circle_focus_processor import (
    CircleFocusResult,
    process_circle_focus_frame,
    CIRCLE_POINT_LABELS
)

# 导入我们新写的四角多圈圆检测函数
from star_focus_processor import process_multi_circle_frame, MultiCircleFocusResult, MULTI_CIRCLE_LABELS

# 导入原有的模糊检测模块
try:
    from focus_processor import (
        BOARD_MODE_AUTO, BOARD_MODE_CHART,
        FocusFrameResult, process_frame
    )
except ImportError:
    print("警告: 未找到 focus_processor 模块，模糊检测模式不可用")
    BOARD_MODE_AUTO = "auto_circle"
    BOARD_MODE_CHART = "camera_test_chart"
    FocusFrameResult = None
    process_frame = None


class CircleIndicator(QFrame):
    """圆形直径检测的状态灯"""
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.led = QLabel()
        self.led.setFixedSize(36, 36)
        self.led.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.led, alignment=Qt.AlignmentFlag.AlignCenter)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        self.diameter_label = QLabel("直径: -- px")
        self.diameter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.diameter_label)
        self.status_label = QLabel("未检测")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.status_label)
        self.set_off()

    def set_off(self) -> None:
        self._set_led("#444444")
        self.diameter_label.setText("直径: -- px")
        self.status_label.setText("未检测")

    def update_point(self, diameter: float | None, status_color: tuple) -> None:
        if diameter is None:
            self.set_off()
            return
        if status_color == (0, 255, 0):
            color = "#22cc22"
            status_text = "正常"
        elif status_color == (0, 0, 255):
            color = "#ee2222"
            status_text = "需调整"
        else:
            color = "#ffaa00"
            status_text = "检测中"
        self._set_led(color)
        self.diameter_label.setText(f"直径: {diameter:.1f} px")
        self.status_label.setText(status_text)

    def _set_led(self, color: str) -> None:
        self.led.setStyleSheet(
            f"background-color: {color}; border-radius: 18px; border: 2px solid #333;"
        )


class LiveCurveDialog(QDialog):
    """实时直径曲线弹窗"""
    def __init__(self, parent=None, max_len=100):
        super().__init__(parent)
        self.setWindowTitle("实时直径曲线")
        self.resize(800, 500)
        self.max_len = max_len
        self.data = deque(maxlen=max_len)
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Avg Diameter (px)")
        self.ax.set_title("Diameter Curve")
        self.ax.grid(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.show()

    def add_point(self, avg_diameter: float):
        idx = len(self.data)
        self.data.append((idx, avg_diameter))
        self.update_plot()

    def update_plot(self):
        if not self.data:
            return
        indices, diams = zip(*self.data)
        self.ax.clear()
        self.ax.plot(indices, diams, 'b-', linewidth=2, marker='o', markersize=4)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Avg Diameter (px)")
        self.ax.set_title("Diameter Curve")
        self.ax.grid(True)
        if hasattr(self.parent(), 'circle_ideal_diameter'):
            ideal = self.parent().circle_ideal_diameter.value()
            self.ax.axhline(y=ideal, color='r', linestyle='--', label=f'Ideal {ideal}px')
            self.ax.legend()
        self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self, args=None):
        super().__init__()
        self.setWindowTitle("对焦检测软件 - 增强版（四圆直径检测）")
        self.resize(1400, 900)

        self.args = args

        # 帧间稳定性缓存
        self.diameter_history = deque(maxlen=5)
        self.status_history = deque(maxlen=5)
        self.last_valid_result = None
        self.consecutive_focus_ok = 0
        self.consecutive_required = 3

        self.curve_dialog = None

        self._cap = None
        self._running = False
        self._last_result_img = None

        self._build_ui()
        self._init_from_args()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)

    def _init_from_args(self):
        if self.args:
            if self.args.camera is not None:
                self.camera_spin.setValue(self.args.camera)
            if self.args.ideal_diameter is not None:
                self.circle_ideal_diameter.setValue(self.args.ideal_diameter)
            if self.args.tolerance is not None:
                self.circle_tolerance.setValue(self.args.tolerance)
            if self.args.std_threshold is not None:
                self.circle_std_threshold.setValue(self.args.std_threshold)
            if self.args.min_radius is not None:
                self.hough_min_radius.setValue(self.args.min_radius)
            if self.args.max_radius is not None:
                self.hough_max_radius.setValue(self.args.max_radius)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # 左侧图像区域
        left = QVBoxLayout()
        self.image_label = QLabel("请打开摄像头或加载图片")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(850, 600)
        self.image_label.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; border: 1px solid #444;")
        left.addWidget(self.image_label, stretch=1)
        root.addLayout(left, stretch=3)

        # ========== 右侧可滚动面板 ==========
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        container = QWidget()
        panel = QVBoxLayout(container)
        panel.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 1. 图像源组
        src_group = QGroupBox("图像源")
        src_layout = QVBoxLayout(src_group)
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("摄像头:"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        cam_row.addWidget(self.camera_spin)
        src_layout.addLayout(cam_row)

        btn_row = QHBoxLayout()
        self.btn_open = QPushButton("打开摄像头")
        self.btn_open.clicked.connect(self._open_camera)
        btn_row.addWidget(self.btn_open)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self._stop_camera)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)
        src_layout.addLayout(btn_row)

        file_row = QHBoxLayout()
        self.btn_load = QPushButton("加载图片")
        self.btn_load.clicked.connect(self._load_image)
        file_row.addWidget(self.btn_load)
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self._save_result)
        file_row.addWidget(self.btn_save)
        src_layout.addLayout(file_row)
        panel.addWidget(src_group)

        # 2. 检测模式
        mode_group = QGroupBox("检测模式")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("圆形直径检测 (四圆)", "circle")
        if process_frame is not None:
            self.mode_combo.addItem("模糊检测 (Camera Test Chart)", "blur_chart")
            self.mode_combo.addItem("模糊检测 (通用圆点板)", "blur_auto")
            self.mode_combo.addItem("四角多圈圆检测", "star_focus")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        panel.addWidget(mode_group)

        # 3. 圆形检测参数组
        self.circle_param_group = QGroupBox("圆形检测参数")
        circle_layout = QVBoxLayout(self.circle_param_group)

        ideal_row = QHBoxLayout()
        ideal_row.addWidget(QLabel("理想直径 (px):"))
        self.circle_ideal_diameter = QDoubleSpinBox()
        self.circle_ideal_diameter.setRange(50, 500)
        self.circle_ideal_diameter.setSingleStep(5)
        self.circle_ideal_diameter.setValue(200)
        ideal_row.addWidget(self.circle_ideal_diameter)
        circle_layout.addLayout(ideal_row)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("直径容差 (px):"))
        self.circle_tolerance = QDoubleSpinBox()
        self.circle_tolerance.setRange(1, 50)
        self.circle_tolerance.setSingleStep(1)
        self.circle_tolerance.setValue(5)
        tol_row.addWidget(self.circle_tolerance)
        circle_layout.addLayout(tol_row)

        std_row = QHBoxLayout()
        std_row.addWidget(QLabel("标准差阈值 (px):"))
        self.circle_std_threshold = QDoubleSpinBox()
        self.circle_std_threshold.setRange(0.5, 20)
        self.circle_std_threshold.setSingleStep(0.5)
        self.circle_std_threshold.setValue(3.0)
        std_row.addWidget(self.circle_std_threshold)
        circle_layout.addLayout(std_row)

        hough_group = QGroupBox("Hough圆检测参数")
        hough_layout = QVBoxLayout(hough_group)
        dp_row = QHBoxLayout()
        dp_row.addWidget(QLabel("dp:"))
        self.hough_dp = QDoubleSpinBox()
        self.hough_dp.setRange(1.0, 3.0)
        self.hough_dp.setSingleStep(0.1)
        self.hough_dp.setValue(1.0)
        dp_row.addWidget(self.hough_dp)
        hough_layout.addLayout(dp_row)

        min_dist_row = QHBoxLayout()
        min_dist_row.addWidget(QLabel("minDist:"))
        self.hough_min_dist = QSpinBox()
        self.hough_min_dist.setRange(10, 200)
        self.hough_min_dist.setValue(50)
        min_dist_row.addWidget(self.hough_min_dist)
        hough_layout.addLayout(min_dist_row)

        param1_row = QHBoxLayout()
        param1_row.addWidget(QLabel("param1:"))
        self.hough_param1 = QSpinBox()
        self.hough_param1.setRange(20, 200)
        self.hough_param1.setValue(100)
        param1_row.addWidget(self.hough_param1)
        hough_layout.addLayout(param1_row)

        param2_row = QHBoxLayout()
        param2_row.addWidget(QLabel("param2:"))
        self.hough_param2 = QSpinBox()
        self.hough_param2.setRange(10, 100)
        self.hough_param2.setValue(30)
        param2_row.addWidget(self.hough_param2)
        hough_layout.addLayout(param2_row)

        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("半径范围:"))
        self.hough_min_radius = QSpinBox()
        self.hough_min_radius.setRange(5, 100)
        self.hough_min_radius.setValue(30)
        self.hough_max_radius = QSpinBox()
        self.hough_max_radius.setRange(50, 300)
        self.hough_max_radius.setValue(200)
        radius_row.addWidget(self.hough_min_radius)
        radius_row.addWidget(QLabel("-"))
        radius_row.addWidget(self.hough_max_radius)
        hough_layout.addLayout(radius_row)
        circle_layout.addWidget(hough_group)

        # 稳定化参数
        stab_group = QGroupBox("帧间稳定性")
        stab_layout = QVBoxLayout(stab_group)
        self.consecutive_spin = QSpinBox()
        self.consecutive_spin.setRange(1, 10)
        self.consecutive_spin.setValue(3)
        self.consecutive_spin.setToolTip("需要连续多少帧满足条件才判定为对焦完成")
        stab_layout.addWidget(QLabel("连续确认帧数:"))
        stab_layout.addWidget(self.consecutive_spin)
        self.fallback_check = QCheckBox("检测失败时沿用上一帧结果")
        self.fallback_check.setChecked(True)
        stab_layout.addWidget(self.fallback_check)
        circle_layout.addWidget(stab_group)

        panel.addWidget(self.circle_param_group)

        # 4. 状态指示器
        self.status_group = QGroupBox("四点对焦状态")
        status_grid = QGridLayout(self.status_group)
        self.circle_indicators = [
            CircleIndicator("P1 左上"), CircleIndicator("P2 右上"),
            CircleIndicator("P3 左下"), CircleIndicator("P4 右下")
        ]
        status_grid.addWidget(self.circle_indicators[0], 0, 0)
        status_grid.addWidget(self.circle_indicators[1], 0, 1)
        status_grid.addWidget(self.circle_indicators[2], 1, 0)
        status_grid.addWidget(self.circle_indicators[3], 1, 1)
        panel.addWidget(self.status_group)

        # 5. 直径信息
        info_group = QGroupBox("直径信息")
        info_layout = QVBoxLayout(info_group)
        self.circle_avg_label = QLabel("平均直径: -- px")
        self.circle_std_label = QLabel("标准差: -- px")
        self.circle_status_label = QLabel("状态: --")
        self.circle_status_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.circle_avg_label)
        info_layout.addWidget(self.circle_std_label)
        info_layout.addWidget(self.circle_status_label)
        panel.addWidget(info_group)

        # 6. 曲线按钮
        self.btn_curve = QPushButton("显示实时曲线")
        self.btn_curve.clicked.connect(self._show_curve)
        panel.addWidget(self.btn_curve)

        # 7. 总体状态
        self.lbl_summary = QLabel("总体: 等待检测")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet("font-weight: bold; font-size: 12px;")
        panel.addWidget(self.lbl_summary)

        # 8. 使用说明
        hint = QGroupBox("使用说明")
        hint_layout = QVBoxLayout(hint)
        hint_layout.addWidget(QLabel("1. 打印四圆标定板，合焦时每个圆直径等于理想直径"))
        hint_layout.addWidget(QLabel("2. 调整Hough参数使圆被正确检测"))
        hint_layout.addWidget(QLabel("3. 绿灯常亮表示对焦完成"))
        panel.addWidget(hint)

        panel.addStretch()

        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        self.setStatusBar(QStatusBar())

    # 以下方法都属于 MainWindow 类，注意缩进
    def _show_curve(self):
        if self.curve_dialog is None or not self.curve_dialog.isVisible():
            self.curve_dialog = LiveCurveDialog(self, max_len=200)
        else:
            self.curve_dialog.raise_()
            self.curve_dialog.activateWindow()

    def _on_mode_changed(self):
        """
        当检测模式切换时，自动重置当前缓存并清除旧状态
        """
        # 清空稳定性历史记录
        self.diameter_history.clear()
        self.status_history.clear()
        self.last_valid_result = None
        self.consecutive_focus_ok = 0
        
        # 将四点状态灯全部复位熄灭
        if hasattr(self, 'circle_indicators'):
            for led in self.circle_indicators:
                led.set_off()
                
        # 重置面板文本
        if hasattr(self, 'circle_avg_label'):
            self.circle_avg_label.setText("平均直径: -- px")
            self.circle_std_label.setText("标准差: -- px")
            self.circle_status_label.setText("状态: --")
            self.lbl_summary.setText("总体: 等待检测")
            self.lbl_summary.setStyleSheet("color: #ff9900; font-weight: bold;")
            
        # 给出提示
        current_text = self.mode_combo.currentText()
        self.statusBar().showMessage(f"已切换至：{current_text}", 3000)

    def _process_and_show(self, frame: np.ndarray):
        """
        核心图像处理分流与显示路由
        """
        # 获取当前 UI 下拉菜单选择的检测模式
        current_mode = self.mode_combo.currentData()

        if current_mode == "circle":
            # 模式 1：传统的四圆直径检测
            self._process_circle_detection(frame)
            
        elif current_mode == "star_focus":
            # 模式 2：新增的中心星图对焦检测
            self._process_star_focus_detection(frame)
            
        else:
           # 模式 3：模糊量图卡检测 (自动圆点板或标准测试图卡)
            if process_frame is not None:
                # 调用 focus_processor 处理图像
                processed, result = process_frame(frame, board_mode=current_mode)
                self._display_image(processed)
                self._last_result_img = processed
                if hasattr(result, 'message'):
                    self.statusBar().showMessage(result.message, 2000)
            else:
                # 如果没找到处理模块，直接输出原图，防止卡死
                self._display_image(frame)

    def _process_circle_detection(self, frame: np.ndarray):
        processed, result = process_circle_focus_frame(
            frame,
            ideal_diameter=self.circle_ideal_diameter.value(),
            tolerance=self.circle_tolerance.value(),
            std_threshold=self.circle_std_threshold.value(),
            hough_dp=self.hough_dp.value(),
            hough_min_dist=self.hough_min_dist.value(),
            hough_param1=self.hough_param1.value(),
            hough_param2=self.hough_param2.value(),
            hough_min_radius=self.hough_min_radius.value(),
            hough_max_radius=self.hough_max_radius.value(),
        )
        stable_result, stable_avg = self._apply_temporal_smoothing(result)
        self._update_circle_indicators(stable_result, stable_avg)
        self._display_image(processed)
        self._last_result_img = processed
        self.statusBar().showMessage(result.message, 2000)

        if result.avg_diameter is not None and self.curve_dialog and self.curve_dialog.isVisible():
            self.curve_dialog.add_point(result.avg_diameter)
    def _process_star_focus_detection(self, frame: np.ndarray):
        """
        已重构：专门用于测图中“4个角的3圈同心圆”的外圈直径
        """
        # 1. 引入新写的文件中的拓扑树圆孔过滤器，并传入你 UI 界面上的微调参数
        # from star_focus_processor import process_multi_circle_frame
        
        processed, result = process_multi_circle_frame(
            frame,
            ideal_diameter=self.circle_ideal_diameter.value(),  # 使用界面设定的理想外圈直径
            tolerance=self.circle_tolerance.value(),            # 直径容差
            std_threshold=self.circle_std_threshold.value()     # 四角均匀度标准差阈值
        )
        
        # 2. 将处理完毕并带有红绿画圆标记的图像进行实时显示
        self._display_image(processed)
        self._last_result_img = processed
        
        # 3. 状态栏输出检测提示（如：“四角对焦成功” 或 “未找齐四角圆”）
        self.statusBar().showMessage(result.message, 2000)
        
        # 4. 完美联动右侧的实时多帧波动曲线图
        # 当四个角的大圆均成功识别且计算出平均外外径时，将数据送入曲线
        if result.avg_diameter is not None and self.curve_dialog and self.curve_dialog.isVisible():
            # 这样你在旋转镜头调焦时，就能实时看到 4 个角圆环的总体像素收缩/扩张的直径曲线了
            self.curve_dialog.add_point(result.avg_diameter)
    
    def _apply_temporal_smoothing(self, result: CircleFocusResult):
        consec_required = self.consecutive_spin.value()
        fallback = self.fallback_check.isChecked()

        current_valid = (result.avg_diameter is not None)
        current_status = result.status
        current_avg = result.avg_diameter

        if current_valid:
            self.diameter_history.append(current_avg)
            self.status_history.append(current_status)
            self.last_valid_result = result
        else:
            if fallback and self.last_valid_result is not None:
                result = self.last_valid_result
                result.message += " [沿用上一帧]"
                current_avg = result.avg_diameter
                current_status = result.status
            else:
                self.consecutive_focus_ok = 0
                return result, current_avg

        if len(self.status_history) >= consec_required:
            recent_statuses = list(self.status_history)[-consec_required:]
            all_focus_ok = all(s == "focus_ok" for s in recent_statuses)

            if all_focus_ok:
                self.consecutive_focus_ok += 1
            else:
                self.consecutive_focus_ok = 0

            if self.consecutive_focus_ok >= consec_required and current_status != "focus_ok":
                result.status = "focus_ok"
                result.status_color = (0, 255, 0)
                result.message = f"稳定对焦完成 (连续{consec_required}帧确认)"

        return result, current_avg

    def _update_circle_indicators(self, result: CircleFocusResult, stable_avg: float = None):
        for i, led in enumerate(self.circle_indicators):
            if i < len(result.circles):
                circle = result.circles[i]
                led.update_point(circle.diameter, result.status_color)
            else:
                led.set_off()
        if result.avg_diameter is not None:
            avg = result.avg_diameter
            std = result.std_diameter
            self.circle_avg_label.setText(f"平均直径: {avg:.1f} px")
            self.circle_std_label.setText(f"标准差: {std:.2f} px")
        else:
            self.circle_avg_label.setText("平均直径: -- px")
            self.circle_std_label.setText("标准差: -- px")
        self.circle_status_label.setText(f"状态: {result.message.split('(')[0]}")
        if result.status == "focus_ok":
            self.lbl_summary.setText(f"✅ 对焦完成！平均直径 {result.avg_diameter:.1f}px")
            self.lbl_summary.setStyleSheet("color: #22aa22; font-weight: bold;")
        elif result.status == "too_near":
            self.lbl_summary.setText(f"⚠️ 离焦太近！平均直径 {result.avg_diameter:.1f}px (目标: {self.circle_ideal_diameter.value()}px)")
            self.lbl_summary.setStyleSheet("color: #cc2222; font-weight: bold;")
        elif result.status == "too_far":
            self.lbl_summary.setText(f"⚠️ 离焦太远！平均直径 {result.avg_diameter:.1f}px (目标: {self.circle_ideal_diameter.value()}px)")
            self.lbl_summary.setStyleSheet("color: #cc2222; font-weight: bold;")
        else:
            self.lbl_summary.setText(f"🔍 {result.message}")
            self.lbl_summary.setStyleSheet("color: #ff9900; font-weight: bold;")

    def _display_image(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def _open_camera(self):
        self._stop_camera()
        cap = cv2.VideoCapture(self.camera_spin.value(), cv2.CAP_DSHOW)
        if not cap.isOpened():
            QMessageBox.warning(self, "错误", "无法打开摄像头")
            return
        self._cap = cap
        self._running = True
        self.btn_open.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._timer.start(33)

    def _stop_camera(self):
        self._timer.stop()
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.btn_open.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_timer(self):
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if ok:
            self._process_and_show(frame)

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "错误", "无法读取图片")
            return
        self._stop_camera()
        self._process_and_show(frame)

    def _save_result(self):
        if self._last_result_img is None:
            QMessageBox.information(self, "提示", "暂无结果可保存")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", f"focus_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self._last_result_img)

    def closeEvent(self, event):
        self._stop_camera()
        if self.curve_dialog:
            self.curve_dialog.close()
        super().closeEvent(event)


def parse_args():
    parser = argparse.ArgumentParser(description="对焦检测软件 - 四圆直径检测模式")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--ideal-diameter", type=float, default=200.0, help="理想直径（像素）")
    parser.add_argument("--tolerance", type=float, default=5.0, help="直径容差（像素）")
    parser.add_argument("--std-threshold", type=float, default=3.0, help="标准差阈值（像素）")
    parser.add_argument("--min-radius", type=int, default=30, help="Hough最小半径")
    parser.add_argument("--max-radius", type=int, default=200, help="Hough最大半径")
    return parser.parse_args()


def run():
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()