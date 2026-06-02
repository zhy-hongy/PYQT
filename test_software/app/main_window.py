"""激光线对准软件主窗口。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from app.image_processor import process_frame


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("激光线位置对准软件")
        self.resize(1200, 800)

        self._cap: cv2.VideoCapture | None = None
        self._static_frame: np.ndarray | None = None
        self._camera_index = 0
        self._running = False

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # 左侧：图像显示
        left = QVBoxLayout()
        self.image_label = QLabel("请打开摄像头或加载图片")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("background-color: #1e1e1e; color: #aaaaaa; border: 1px solid #444;")
        left.addWidget(self.image_label, stretch=1)
        root.addLayout(left, stretch=3)

        # 右侧：控制面板
        panel = QVBoxLayout()

        src_group = QGroupBox("图像源")
        src_layout = QVBoxLayout(src_group)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("摄像头:"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(0)
        cam_row.addWidget(self.camera_spin)
        src_layout.addLayout(cam_row)

        btn_row = QHBoxLayout()
        self.btn_open_cam = QPushButton("打开摄像头")
        self.btn_open_cam.clicked.connect(self._open_camera)
        btn_row.addWidget(self.btn_open_cam)

        self.btn_stop_cam = QPushButton("停止")
        self.btn_stop_cam.clicked.connect(self._stop_camera)
        self.btn_stop_cam.setEnabled(False)
        btn_row.addWidget(self.btn_stop_cam)
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

        laser_group = QGroupBox("激光检测参数")
        laser_layout = QVBoxLayout(laser_group)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("激光波长:"))
        self.color_combo = QComboBox()
        self.color_combo.addItem("405nm (紫外/紫)", "violet")
        self.color_combo.addItem("红色 650nm", "red")
        self.color_combo.addItem("绿色 532nm", "green")
        color_row.addWidget(self.color_combo)
        laser_layout.addLayout(color_row)

        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("颜色灵敏度:"))
        self.sens_slider = QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(10, 80)
        self.sens_slider.setValue(30)
        self.sens_label = QLabel("30")
        self.sens_slider.valueChanged.connect(lambda v: self.sens_label.setText(str(v)))
        sens_row.addWidget(self.sens_slider)
        sens_row.addWidget(self.sens_label)
        laser_layout.addLayout(sens_row)

        cal_row = QHBoxLayout()
        cal_row.addWidget(QLabel("像素/mm:"))
        self.ppm_spin = QDoubleSpinBox()
        self.ppm_spin.setRange(0.0, 1000.0)
        self.ppm_spin.setDecimals(2)
        self.ppm_spin.setSingleStep(0.1)
        self.ppm_spin.setToolTip("标定值，0 表示仅显示像素距离")
        cal_row.addWidget(self.ppm_spin)
        laser_layout.addLayout(cal_row)
        panel.addWidget(laser_group)

        result_group = QGroupBox("测量结果")
        result_layout = QVBoxLayout(result_group)
        self.lbl_angle = QLabel("激光与Y轴夹角: --")
        self.lbl_distance = QLabel("中点到交点距离: --")
        self.lbl_status = QLabel("状态: 就绪")
        for lbl in (self.lbl_angle, self.lbl_distance, self.lbl_status):
            lbl.setWordWrap(True)
            result_layout.addWidget(lbl)
        panel.addWidget(result_group)

        legend = QGroupBox("图例")
        legend_layout = QVBoxLayout(legend)
        legend_layout.addWidget(QLabel("橙色 — X轴（水平）"))
        legend_layout.addWidget(QLabel("青色 — Y轴（竖直）"))
        legend_layout.addWidget(QLabel("黄色十字 — 交点"))
        legend_layout.addWidget(QLabel("品红 — 405nm 激光线及中点"))
        legend_layout.addWidget(QLabel("紫色 — 中点到交点连线"))
        panel.addWidget(legend)

        panel.addStretch()
        root.addLayout(panel, stretch=1)

        self.setStatusBar(QStatusBar())

    def _laser_color(self) -> str:
        return self.color_combo.currentData() or "violet"

    def _sensitivity(self) -> int:
        return self.sens_slider.value()

    def _pixels_per_mm(self) -> float:
        return self.ppm_spin.value()

    def _show_frame(self, bgr: np.ndarray) -> None:
        processed, result = process_frame(
            bgr,
            laser_color=self._laser_color(),
            sensitivity=self._sensitivity(),
            pixels_per_mm=self._pixels_per_mm(),
        )

        self._update_result_labels(result)

        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        scaled = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self._last_result = processed

    def _update_result_labels(self, result) -> None:
        if result.angle_to_vertical_deg is not None:
            self.lbl_angle.setText(f"激光与Y轴夹角: {result.angle_to_vertical_deg:.2f}°")
        else:
            self.lbl_angle.setText("激光与Y轴夹角: --")

        if result.distance_px is not None:
            text = f"中点到交点距离: {result.distance_px:.1f} px"
            ppm = self._pixels_per_mm()
            if ppm > 0:
                text += f"  ({result.distance_px / ppm:.2f} mm)"
            self.lbl_distance.setText(text)
        else:
            self.lbl_distance.setText("中点到交点距离: --")

        self.lbl_status.setText(f"状态: {result.message or '检测完成'}")
        self.statusBar().showMessage(result.message or "检测完成", 2000)

    def _open_camera(self) -> None:
        self._stop_camera()
        index = self.camera_spin.value()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开摄像头 {index}")
            return

        self._cap = cap
        self._static_frame = None
        self._running = True
        self.btn_open_cam.setEnabled(False)
        self.btn_stop_cam.setEnabled(True)
        self._timer.start(33)

    def _stop_camera(self) -> None:
        self._timer.stop()
        self._running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.btn_open_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)

    def _on_timer(self) -> None:
        if not self._running or self._cap is None:
            return
        ok, frame = self._cap.read()
        if ok:
            self._show_frame(frame)

    def _load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "错误", "无法读取图片")
            return

        self._stop_camera()
        self._static_frame = frame
        self._show_frame(frame)

    def _save_result(self) -> None:
        if not hasattr(self, "_last_result"):
            QMessageBox.information(self, "提示", "暂无结果可保存")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存结果图片",
            "alignment_result.png",
            "PNG (*.png);;JPEG (*.jpg)",
        )
        if path:
            cv2.imwrite(path, self._last_result)
            self.statusBar().showMessage(f"已保存: {path}", 3000)

    def closeEvent(self, event) -> None:
        self._stop_camera()
        super().closeEvent(event)


def run() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
