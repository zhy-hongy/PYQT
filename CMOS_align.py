# -*- coding: utf-8 -*-
import os
import sys
import time
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List

from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
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
    QGroupBox,
    QProgressBar,
    QCheckBox,
    QFrame,
    QTextEdit,
)


# =====================================================
# 清晰度评价函数
# =====================================================
def calculate_tenengrad(image_gray):
    """Tenengrad清晰度评价函数 - 值越大越清晰"""
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    focus_measure = np.sum(sobel_x**2 + sobel_y**2) / image_gray.size
    return focus_measure


def calculate_laplacian_variance(image_gray):
    """拉普拉斯方差法 - 值越大越清晰"""
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


# =====================================================
# CMOS对齐校准相关类
# =====================================================
@dataclass
class LineResult:
    """直线检测结果"""
    points: np.ndarray  # 拟合用的点集
    line: Tuple[float, float, float]  # (vx, vy, x0, y0) 方向向量和经过点
    angle_deg: float  # 角度（度）
    is_valid: bool = True


@dataclass
class CrossResult:
    """十字线检测结果"""
    horizontal: Optional[LineResult]  # 水平线
    vertical: Optional[LineResult]    # 垂直线
    center: Optional[Tuple[float, float]]  # 交点中心
    angle_error: float = 0.0  # 旋转角度误差
    center_error: float = 0.0  # 中心距离误差


class CMOSAlignmentWidget(QWidget):
    """CMOS对齐校准面板（可折叠）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.alignment_enabled = False  # 对齐检测开关状态
        self.cross_result = None  # 十字线检测结果
        
        # 图像中心点（固定）
        self.image_center = None  # 将在检测时从图像尺寸获取
        
        # 缓存最近检测的十字线中心
        self.last_cross_center = None
        
        self.init_ui()
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # ==========================
        # 可折叠的CMOS对齐校准组
        # ==========================
        self.align_group = QGroupBox("🎯 CMOS对齐校准")
        self.align_group.setCheckable(True)
        self.align_group.setChecked(False)  # 默认折叠
        self.align_group.toggled.connect(self.on_align_toggled)
        
        group_layout = QVBoxLayout(self.align_group)
        
        # 对齐检测开关
        switch_layout = QHBoxLayout()
        switch_layout.addWidget(QLabel("对齐检测:"))
        self.align_switch = QCheckBox("开启")
        self.align_switch.setChecked(False)
        self.align_switch.toggled.connect(self.on_align_switch_toggled)
        switch_layout.addWidget(self.align_switch)
        switch_layout.addStretch()
        
        # 状态指示
        self.align_status_label = QLabel("⚫ 对齐检测已关闭")
        self.align_status_label.setStyleSheet("QLabel { color: gray; font-weight: bold; }")
        switch_layout.addWidget(self.align_status_label)
        
        group_layout.addLayout(switch_layout)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        group_layout.addWidget(line)
        
        # 参数设置组
        param_group = QGroupBox("检测参数")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)
        
        # 边缘阈值
        param_layout.addWidget(QLabel("Canny阈值1:"), 0, 0)
        self.thresh1_spin = QSlider(Qt.Horizontal)
        self.thresh1_spin.setRange(10, 200)
        self.thresh1_spin.setValue(50)
        param_layout.addWidget(self.thresh1_spin, 0, 1)
        self.thresh1_label = QLabel("50")
        param_layout.addWidget(self.thresh1_label, 0, 2)
        self.thresh1_spin.valueChanged.connect(lambda v: self.thresh1_label.setText(str(v)))
        
        param_layout.addWidget(QLabel("Canny阈值2:"), 1, 0)
        self.thresh2_spin = QSlider(Qt.Horizontal)
        self.thresh2_spin.setRange(50, 300)
        self.thresh2_spin.setValue(150)
        param_layout.addWidget(self.thresh2_spin, 1, 1)
        self.thresh2_label = QLabel("150")
        param_layout.addWidget(self.thresh2_label, 1, 2)
        self.thresh2_spin.valueChanged.connect(lambda v: self.thresh2_label.setText(str(v)))
        
        param_layout.addWidget(QLabel("ROI扩展像素:"), 2, 0)
        self.roi_padding_spin = QSlider(Qt.Horizontal)
        self.roi_padding_spin.setRange(10, 100)
        self.roi_padding_spin.setValue(30)
        param_layout.addWidget(self.roi_padding_spin, 2, 1)
        self.roi_padding_label = QLabel("30")
        param_layout.addWidget(self.roi_padding_label, 2, 2)
        self.roi_padding_spin.valueChanged.connect(lambda v: self.roi_padding_label.setText(str(v)))
        
        group_layout.addWidget(param_group)
        
        # 检测按钮
        self.detect_btn = QPushButton("手动检测十字线")
        self.detect_btn.clicked.connect(self.manual_detect)
        self.detect_btn.setEnabled(False)
        group_layout.addWidget(self.detect_btn)
        
        # 结果显示
        result_group = QGroupBox("偏差结果")
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)
        
        self.result_text = QTextEdit()
        self.result_text.setMaximumHeight(120)
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 11px; }")
        result_layout.addWidget(self.result_text)
        
        group_layout.addWidget(result_group)
        
        # 校准提示
        hint_label = QLabel("💡 提示: 使用405nm背光十字标靶，系统将检测十字线中心与图像中心的偏差")
        hint_label.setStyleSheet("QLabel { color: gray; font-size: 10px; padding: 5px; }")
        hint_label.setWordWrap(True)
        group_layout.addWidget(hint_label)
        
        main_layout.addWidget(self.align_group)
        
        # 初始化结果显示
        self.update_result_display()
    
    def on_align_toggled(self, checked):
        """CMOS对齐校准组折叠/展开"""
        # 折叠/展开组内所有子控件
        layout = self.align_group.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.widget():
                    item.widget().setVisible(checked)
    
    def on_align_switch_toggled(self, checked):
        """对齐检测开关切换"""
        self.alignment_enabled = checked
        if checked:
            self.align_status_label.setText("🟢 对齐检测已开启")
            self.align_status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            self.detect_btn.setEnabled(True)
        else:
            self.align_status_label.setText("⚫ 对齐检测已关闭")
            self.align_status_label.setStyleSheet("QLabel { color: gray; font-weight: bold; }")
            self.detect_btn.setEnabled(False)
            self.cross_result = None
            self.update_result_display()
    
    def manual_detect(self):
        """手动触发检测"""
        if self.parent() and hasattr(self.parent(), 'current_frame'):
            frame = self.parent().current_frame
            if frame is not None:
                self.detect_cross(frame)
    
    def detect_cross(self, image: np.ndarray) -> Optional[CrossResult]:
        """检测十字线中心
        
        Args:
            image: 输入图像（BGR格式）
            
        Returns:
            CrossResult: 检测结果，包含十字线中心、角度偏差、距离偏差
        """
        if not self.alignment_enabled:
            return None
        
        if image is None:
            return None
        
        h, w = image.shape[:2]
        self.image_center = (w // 2, h // 2)
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Canny边缘检测
        thresh1 = self.thresh1_spin.value()
        thresh2 = self.thresh2_spin.value()
        edges = cv2.Canny(gray, thresh1, thresh2)
        
        # 可选：使用霍夫线检测找到十字线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is None:
            self.cross_result = None
            self.update_result_display()
            return None
        
        # 分类水平线和垂直线
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # 判断接近水平（角度在 -10 到 10 度之间）
            if abs(angle) < 10 or abs(angle) > 170:
                horizontal_lines.append(line[0])
            # 判断接近垂直（角度在 80 到 100 度之间）
            elif 80 < abs(angle) < 100:
                vertical_lines.append(line[0])
        
        # 使用亚像素边缘精确定位
        horizontal_result = self._fit_line_subpixel(gray, horizontal_lines, is_horizontal=True)
        vertical_result = self._fit_line_subpixel(gray, vertical_lines, is_horizontal=False)
        
        # 计算交点
        cross_center = None
        if horizontal_result and vertical_result:
            cross_center = self._compute_intersection(
                horizontal_result.line, vertical_result.line
            )
        
        # 计算偏差
        angle_error = 0.0
        center_error = 0.0
        if cross_center:
            # 角度偏差：应该接近0度（水平线和垂直线应垂直）
            if horizontal_result:
                angle_error = abs(horizontal_result.angle_deg)
                if angle_error > 90:
                    angle_error = 180 - angle_error
            
            # 中心距离偏差（像素）
            cx, cy = cross_center
            center_error = np.sqrt((cx - self.image_center[0])**2 + 
                                  (cy - self.image_center[1])**2)
            
            self.last_cross_center = cross_center
        
        # 保存结果
        self.cross_result = CrossResult(
            horizontal=horizontal_result,
            vertical=vertical_result,
            center=cross_center,
            angle_error=angle_error,
            center_error=center_error
        )
        
        self.update_result_display()
        return self.cross_result
    
    def _fit_line_subpixel(self, gray: np.ndarray, lines: List, 
                           is_horizontal: bool) -> Optional[LineResult]:
        """使用亚像素边缘精确定位直线
        
        对霍夫线检测到的直线附近区域，使用Canny边缘和最小二乘法拟合精确直线
        """
        if not lines:
            return None
        
        h, w = gray.shape
        padding = self.roi_padding_spin.value()
        
        all_points = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # 确定ROI区域
            if is_horizontal:
                # 水平线：取线的y坐标附近的区域
                y_center = (y1 + y2) // 2
                roi_y1 = max(0, y_center - padding)
                roi_y2 = min(h, y_center + padding)
                roi_x1 = 0
                roi_x2 = w
            else:
                # 垂直线：取线的x坐标附近的区域
                x_center = (x1 + x2) // 2
                roi_x1 = max(0, x_center - padding)
                roi_x2 = min(w, x_center + padding)
                roi_y1 = 0
                roi_y2 = h
            
            # 提取ROI区域
            roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0:
                continue
            
            # 使用Canny提取边缘
            thresh1 = self.thresh1_spin.value()
            thresh2 = self.thresh2_spin.value()
            edges_roi = cv2.Canny(roi, thresh1, thresh2)
            
            # 找到边缘点
            points = np.column_stack(np.where(edges_roi > 0))
            if len(points) < 10:
                continue
            
            # 转换回全局坐标
            for py, px in points:
                global_x = roi_x1 + px
                global_y = roi_y1 + py
                all_points.append([global_x, global_y])
        
        if len(all_points) < 10:
            return None
        
        points = np.array(all_points)
        
        # 使用最小二乘法拟合直线
        if is_horizontal:
            # 水平线：拟合 y = a*x + b
            x = points[:, 0]
            y = points[:, 1]
            A = np.vstack([x, np.ones(len(x))]).T
            try:
                a, b = np.linalg.lstsq(A, y, rcond=None)[0]
                # 转换为方向向量形式 (vx, vy, x0, y0)
                # 对于 y = a*x + b，方向向量为 (1, a)
                vx, vy = 1.0, a
                # 归一化
                norm = np.sqrt(vx**2 + vy**2)
                vx, vy = vx / norm, vy / norm
                # 选择一个点
                x0, y0 = x[0], a * x[0] + b
                angle_deg = np.degrees(np.arctan2(vy, vx))
                return LineResult(points=points, line=(vx, vy, x0, y0), 
                                 angle_deg=angle_deg, is_valid=True)
            except:
                return None
        else:
            # 垂直线：拟合 x = c*y + d
            x = points[:, 0]
            y = points[:, 1]
            A = np.vstack([y, np.ones(len(y))]).T
            try:
                c, d = np.linalg.lstsq(A, x, rcond=None)[0]
                # 方向向量 (c, 1)
                vx, vy = c, 1.0
                norm = np.sqrt(vx**2 + vy**2)
                vx, vy = vx / norm, vy / norm
                x0, y0 = c * y[0] + d, y[0]
                angle_deg = np.degrees(np.arctan2(vy, vx))
                return LineResult(points=points, line=(vx, vy, x0, y0),
                                 angle_deg=angle_deg, is_valid=True)
            except:
                return None
        
        return None
    
    def _compute_intersection(self, line1: Tuple[float, float, float, float],
                              line2: Tuple[float, float, float, float]) -> Optional[Tuple[float, float]]:
        """计算两条直线的交点
        
        line: (vx, vy, x0, y0) 方向向量和经过点
        """
        vx1, vy1, x01, y01 = line1
        vx2, vy2, x02, y02 = line2
        
        # 解方程组：
        # x = x01 + t1*vx1 = x02 + t2*vx2
        # y = y01 + t1*vy1 = y02 + t2*vy2
        
        A = np.array([[vx1, -vx2], [vy1, -vy2]])
        b = np.array([x02 - x01, y02 - y01])
        
        try:
            t = np.linalg.solve(A, b)
            x = x01 + t[0] * vx1
            y = y01 + t[0] * vy1
            return (x, y)
        except:
            return None
    
    def update_result_display(self):
        """更新结果显示"""
        if not self.cross_result or not self.cross_result.center:
            self.result_text.setText("等待检测十字线...\n\n"
                                     "请确保:\n"
                                     "1. 405nm背光已开启\n"
                                     "2. 十字标靶在视野内\n"
                                     "3. 点击「手动检测十字线」按钮")
            return
        
        result = self.cross_result
        text = f"【十字线检测结果】\n"
        text += f"{'='*40}\n"
        
        if result.center:
            text += f"📍 十字线中心: ({result.center[0]:.1f}, {result.center[1]:.1f})\n"
            text += f"🎯 图像中心: ({self.image_center[0]}, {self.image_center[1]})\n"
            text += f"📏 中心偏差: {result.center_error:.2f} 像素\n"
        
        if result.horizontal:
            text += f"➡️ 水平线角度: {result.horizontal.angle_deg:.2f}°\n"
        
        if result.vertical:
            text += f"⬆️ 垂直线角度: {result.vertical.angle_deg:.2f}°\n"
        
        text += f"🔄 旋转偏差: {result.angle_error:.2f}°\n"
        
        if result.center_error < 5 and result.angle_error < 0.5:
            text += f"\n✅ 校准状态: 良好"
        elif result.center_error < 20 and result.angle_error < 2:
            text += f"\n⚠️ 校准状态: 可接受"
        else:
            text += f"\n❌ 校准状态: 需要调整"
        
        self.result_text.setText(text)
    
    def is_alignment_enabled(self):
        """获取对齐检测是否开启"""
        return self.alignment_enabled
    
    def get_current_result(self) -> Optional[CrossResult]:
        """获取当前检测结果"""
        return self.cross_result
    
    def draw_alignment_overlay(self, image: np.ndarray) -> np.ndarray:
        """在图像上绘制对齐校准的叠加层
        
        Args:
            image: 原始图像
            
        Returns:
            绘制了叠加层的图像
        """
        if not self.alignment_enabled:
            return image
        
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # 1. 绘制图像中心十字（绿色）
        center_x, center_y = w // 2, h // 2
        cv2.line(overlay, (center_x - 50, center_y), (center_x + 50, center_y), 
                (0, 255, 0), 2)
        cv2.line(overlay, (center_x, center_y - 50), (center_x, center_y + 50), 
                (0, 255, 0), 2)
        cv2.circle(overlay, (center_x, center_y), 8, (0, 255, 0), 2)
        cv2.putText(overlay, "Image Center", (center_x + 10, center_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 2. 绘制检测到的十字线中心（红色）
        if self.cross_result and self.cross_result.center:
            cx, cy = self.cross_result.center
            cv2.line(overlay, (int(cx - 50), int(cy)), (int(cx + 50), int(cy)), 
                    (0, 0, 255), 2)
            cv2.line(overlay, (int(cx), int(cy - 50)), (int(cx), int(cy + 50)), 
                    (0, 0, 255), 2)
            cv2.circle(overlay, (int(cx), int(cy)), 8, (0, 0, 255), 2)
            cv2.putText(overlay, "Detected Cross", (int(cx) + 10, int(cy) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 绘制连接线（显示偏差）
            cv2.line(overlay, (center_x, center_y), (int(cx), int(cy)), 
                    (255, 255, 0), 1, cv2.LINE_AA)
            
            # 显示偏差数值
            cv2.putText(overlay, f"Offset: {self.cross_result.center_error:.1f}px",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(overlay, f"Angle: {self.cross_result.angle_error:.2f}deg",
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # 3. 绘制检测到的直线（蓝色）
        if self.cross_result:
            if self.cross_result.horizontal:
                line = self.cross_result.horizontal.line
                vx, vy, x0, y0 = line
                # 延长直线到图像边界
                left_y = y0 + vy/vx * (0 - x0) if abs(vx) > 1e-6 else y0
                right_y = y0 + vy/vx * (w - x0) if abs(vx) > 1e-6 else y0
                cv2.line(overlay, (0, int(left_y)), (w, int(right_y)), 
                        (255, 0, 0), 1)
            
            if self.cross_result.vertical:
                line = self.cross_result.vertical.line
                vx, vy, x0, y0 = line
                # 延长直线到图像边界
                top_x = x0 + vx/vy * (0 - y0) if abs(vy) > 1e-6 else x0
                bottom_x = x0 + vx/vy * (h - y0) if abs(vy) > 1e-6 else x0
                cv2.line(overlay, (int(top_x), 0), (int(bottom_x), h), 
                        (255, 0, 0), 1)
        
        return overlay


class FocusScoreWidget(QWidget):
    """清晰度评分面板（可折叠）"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_names = ["ROI1", "ROI2", "ROI3", "ROI4"]
        self.roi_scores = {name: 0.0 for name in self.roi_names}
        self.score_history = {name: deque(maxlen=100) for name in self.roi_names}
        self.max_score_history = 0
        self.focus_enabled = False  # 对焦开关状态
        
        self.init_ui()
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # ==========================
        # 可折叠的对焦功能组
        # ==========================
        self.focus_group = QGroupBox("🔍 对焦辅助功能")
        self.focus_group.setCheckable(True)
        self.focus_group.setChecked(False)  # 默认折叠
        self.focus_group.toggled.connect(self.on_focus_toggled)
        
        group_layout = QVBoxLayout(self.focus_group)
        
        # 对焦开关（独立开关，控制是否检测）
        switch_layout = QHBoxLayout()
        switch_layout.addWidget(QLabel("对焦检测:"))
        self.focus_switch = QCheckBox("开启")
        self.focus_switch.setChecked(False)
        self.focus_switch.toggled.connect(self.on_focus_switch_toggled)
        switch_layout.addWidget(self.focus_switch)
        switch_layout.addStretch()
        
        # 状态指示
        self.status_label = QLabel("⚫ 对焦检测已关闭")
        self.status_label.setStyleSheet("QLabel { color: gray; font-weight: bold; }")
        switch_layout.addWidget(self.status_label)
        
        group_layout.addLayout(switch_layout)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        group_layout.addWidget(line)
        
        # 实时评分显示组
        score_group = QGroupBox("实时清晰度评分")
        score_layout = QGridLayout()
        
        self.score_labels = {}
        self.score_bars = {}
        for i, name in enumerate(self.roi_names):
            label = QLabel(f"{name}:")
            value_label = QLabel("0.00")
            bar = QProgressBar()
            bar.setOrientation(Qt.Horizontal)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat(f"{name}: %p%")
            self.score_labels[name] = value_label
            self.score_bars[name] = bar
            score_layout.addWidget(label, i, 0)
            score_layout.addWidget(value_label, i, 1)
            score_layout.addWidget(bar, i, 2)
        
        # 综合评分
        self.combined_label = QLabel("综合评分: 0.00")
        self.combined_bar = QProgressBar()
        self.combined_bar.setOrientation(Qt.Horizontal)
        self.combined_bar.setRange(0, 100)
        self.combined_bar.setFormat("综合: %p%")
        score_layout.addWidget(self.combined_label, len(self.roi_names), 0, 1, 2)
        score_layout.addWidget(self.combined_bar, len(self.roi_names), 2)
        
        score_group.setLayout(score_layout)
        group_layout.addWidget(score_group)
        
        # 算法选择
        algo_layout = QHBoxLayout()
        algo_layout.addWidget(QLabel("评价算法:"))
        self.algo_combo = QCheckBox("使用Laplacian")
        self.algo_combo.setChecked(False)
        self.algo_combo.toggled.connect(self.on_algo_changed)
        algo_layout.addWidget(self.algo_combo)
        
        algo_info = QLabel("(不勾选=Tenengrad)")
        algo_info.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        algo_layout.addWidget(algo_info)
        algo_layout.addStretch()
        group_layout.addLayout(algo_layout)
        
        # 峰值指示
        self.peak_label = QLabel("⏸ 对焦检测未开启")
        self.peak_label.setAlignment(Qt.AlignCenter)
        self.peak_label.setStyleSheet("QLabel { color: gray; font-weight: bold; padding: 5px; }")
        group_layout.addWidget(self.peak_label)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.reset_btn = QPushButton("重置峰值记录")
        self.reset_btn.clicked.connect(self.reset_peak)
        self.reset_btn.setEnabled(False)
        btn_layout.addWidget(self.reset_btn)
        
        self.clear_history_btn = QPushButton("清空历史")
        self.clear_history_btn.clicked.connect(self.clear_history)
        self.clear_history_btn.setEnabled(False)
        btn_layout.addWidget(self.clear_history_btn)
        btn_layout.addStretch()
        group_layout.addLayout(btn_layout)
        
        # ROI提示
        roi_hint = QLabel("💡 提示: 在画面上拖拽鼠标可绘制自定义ROI区域（最多4个）")
        roi_hint.setStyleSheet("QLabel { color: gray; font-size: 10px; padding: 5px; }")
        roi_hint.setWordWrap(True)
        group_layout.addWidget(roi_hint)
        
        main_layout.addWidget(self.focus_group)
    
    def get_current_algorithm(self):
        """获取当前选择的算法 (0=Tenengrad, 1=Laplacian)"""
        return 1 if self.algo_combo.isChecked() else 0
    
    def on_focus_toggled(self, checked):
        """对焦功能组折叠/展开时的处理"""
        # 折叠/展开组内所有子控件
        layout = self.focus_group.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.widget():
                    item.widget().setVisible(checked)
    
    def on_focus_switch_toggled(self, checked):
        """对焦检测开关切换"""
        self.focus_enabled = checked
        if checked:
            self.status_label.setText("🟢 对焦检测已开启")
            self.status_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            self.peak_label.setText("🎯 对焦检测已开启，开始调节...")
            self.peak_label.setStyleSheet("QLabel { color: blue; font-weight: bold; padding: 5px; }")
            self.reset_btn.setEnabled(True)
            self.clear_history_btn.setEnabled(True)
            # 重置峰值，重新开始
            self.reset_peak()
        else:
            self.status_label.setText("⚫ 对焦检测已关闭")
            self.status_label.setStyleSheet("QLabel { color: gray; font-weight: bold; }")
            self.peak_label.setText("⏸ 对焦检测已关闭")
            self.peak_label.setStyleSheet("QLabel { color: gray; font-weight: bold; padding: 5px; }")
            self.reset_btn.setEnabled(False)
            self.clear_history_btn.setEnabled(False)
            # 清空显示
            for name in self.roi_names:
                self.score_labels[name].setText("0.00")
                self.score_bars[name].setValue(0)
            self.combined_label.setText("综合评分: 0.00")
            self.combined_bar.setValue(0)
    
    def on_algo_changed(self, checked):
        """算法切换时的处理"""
        if self.focus_enabled:
            self.reset_peak()
            self.peak_label.setText("算法已切换，重新对焦")
            self.peak_label.setStyleSheet("QLabel { color: orange; font-weight: bold; padding: 5px; }")
    
    def update_scores(self, scores):
        """更新评分显示（仅当对焦检测开启时）"""
        if not self.focus_enabled:
            return
        
        if not scores or len(scores) != len(self.roi_names):
            return
        
        for i, name in enumerate(self.roi_names):
            score = scores[i] if i < len(scores) else 0
            self.roi_scores[name] = score
            self.score_labels[name].setText(f"{score:.2f}")
            
            # 动态归一化
            if score > self.max_score_history:
                self.max_score_history = score
            if self.max_score_history > 0:
                bar_val = int(score / self.max_score_history * 100)
            else:
                bar_val = 0
            self.score_bars[name].setValue(bar_val)
            
            # 更新历史记录
            self.score_history[name].append(score)
        
        # 综合评分
        combined = np.mean(scores)
        self.combined_label.setText(f"综合评分: {combined:.2f}")
        if self.max_score_history > 0:
            combined_bar_val = int(combined / self.max_score_history * 100)
        else:
            combined_bar_val = 0
        self.combined_bar.setValue(combined_bar_val)
        
        # 峰值检测
        if hasattr(self, '_peak_combined'):
            if combined > self._peak_combined:
                self._peak_combined = combined
                self.peak_label.setText(f"🎉 达到新峰值! 综合评分: {combined:.2f} 🎉")
                self.peak_label.setStyleSheet("QLabel { color: red; font-weight: bold; font-size: 12px; padding: 5px; }")
                return
        
        # 如果没有设置峰值，初始化
        if not hasattr(self, '_peak_combined'):
            self._peak_combined = combined
            self.peak_label.setText(f"📊 当前峰值: {combined:.2f}")
            self.peak_label.setStyleSheet("QLabel { color: blue; font-weight: bold; padding: 5px; }")
    
    def get_score_history(self):
        """获取历史分数"""
        return self.score_history
    
    def reset_peak(self):
        """重置峰值记录"""
        self._peak_combined = 0
        self.max_score_history = 0
        self.peak_label.setText("📈 峰值已重置，开始对焦")
        self.peak_label.setStyleSheet("QLabel { color: green; font-weight: bold; padding: 5px; }")
        
        # 重置进度条
        for name in self.roi_names:
            self.score_bars[name].setValue(0)
        self.combined_bar.setValue(0)
    
    def clear_history(self):
        """清空历史记录"""
        for name in self.roi_names:
            self.score_history[name].clear()
        self.max_score_history = 0
        self.peak_label.setText("📈 历史已清空，重新对焦")
        self.peak_label.setStyleSheet("QLabel { color: green; font-weight: bold; padding: 5px; }")
        
        # 重置进度条
        for name in self.roi_names:
            self.score_bars[name].setValue(0)
        self.combined_bar.setValue(0)
    
    def is_focus_enabled(self):
        """获取对焦检测是否开启"""
        return self.focus_enabled


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("沙姆相机对焦辅助系统 - UVC相机控制")
        self.resize(1600, 900)

        self.save_dir = "./saved_images"
        os.makedirs(self.save_dir, exist_ok=True)

        self.capture = None
        self.current_frame = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # ROI相关变量
        self.rois = []  # ROI矩形列表
        self.drawing_roi = False
        self.roi_start = QPoint()
        self.roi_rect = None
        
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)

        # ==========================
        # 图像显示区（支持ROI绘制）
        # ==========================
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        
        self.image_label = QLabel("点击打开相机")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            background:black;
            color:white;
            min-width:800px;
            min-height:600px;
        """)
        # 启用鼠标追踪以绘制ROI
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.label_mouse_press
        self.image_label.mouseMoveEvent = self.label_mouse_move
        self.image_label.mouseReleaseEvent = self.label_mouse_release
        
        display_layout.addWidget(self.image_label)
        
        # ROI控制按钮
        roi_btn_layout = QHBoxLayout()
        self.btn_reset_roi = QPushButton("重置默认ROI")
        self.btn_reset_roi.clicked.connect(self.set_default_rois)
        self.btn_clear_roi = QPushButton("清除所有ROI")
        self.btn_clear_roi.clicked.connect(self.clear_rois)
        roi_btn_layout.addWidget(self.btn_reset_roi)
        roi_btn_layout.addWidget(self.btn_clear_roi)
        roi_btn_layout.addStretch()
        display_layout.addLayout(roi_btn_layout)
        
        main_layout.addWidget(display_widget, stretch=3)

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

        # ==========================
        # 相机参数调节
        # ==========================
        param_group = QGroupBox("相机参数调节")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)

        # 亮度
        param_layout.addWidget(QLabel("亮度"), 0, 0)
        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(0, 2000)
        self.slider_brightness.valueChanged.connect(self.change_brightness)
        param_layout.addWidget(self.slider_brightness, 0, 1)
        self.lbl_brightness = QLabel("0")
        param_layout.addWidget(self.lbl_brightness, 0, 2)

        # 增益
        param_layout.addWidget(QLabel("增益"), 1, 0)
        self.slider_gain = QSlider(Qt.Horizontal)
        self.slider_gain.setRange(0, 255)
        self.slider_gain.valueChanged.connect(self.change_gain)
        param_layout.addWidget(self.slider_gain, 1, 1)
        self.lbl_gain = QLabel("0")
        param_layout.addWidget(self.lbl_gain, 1, 2)

        control_layout.addWidget(param_group)
        
        # ==========================
        # 清晰度评分面板（可折叠）
        # ==========================
        self.focus_widget = FocusScoreWidget()
        control_layout.addWidget(self.focus_widget)
        
        # ==========================
        # CMOS对齐校准面板（可折叠）
        # ==========================
        self.alignment_widget = CMOSAlignmentWidget(self)
        control_layout.addWidget(self.alignment_widget)
        
        control_layout.addSpacing(10)

        # 相机驱动设置
        self.btn_setting = QPushButton("相机驱动设置")
        self.btn_setting.clicked.connect(self.open_camera_setting)
        control_layout.addWidget(self.btn_setting)

        control_layout.addStretch()

    # =====================================================
    # ROI操作
    # =====================================================
    def set_default_rois(self):
        """设置默认ROI（四角）"""
        if self.current_frame is None:
            QMessageBox.warning(self, "提示", "请先打开相机")
            return
        
        h, w = self.current_frame.shape[:2]
        roi_w = min(150, w // 4)
        roi_h = min(150, h // 4)
        margin = 20
        
        self.rois = [
            QRect(margin, margin, roi_w, roi_h),                           # 左上
            QRect(w - roi_w - margin, margin, roi_w, roi_h),              # 右上
            QRect(margin, h - roi_h - margin, roi_w, roi_h),              # 左下
            QRect(w - roi_w - margin, h - roi_h - margin, roi_w, roi_h)   # 右下
        ]
        
        # 重置峰值
        if self.focus_widget.is_focus_enabled():
            self.focus_widget.reset_peak()
        
    def clear_rois(self):
        """清除所有ROI"""
        self.rois = []
        if self.focus_widget.is_focus_enabled():
            self.focus_widget.reset_peak()
    
    def label_mouse_press(self, event):
        """鼠标按下 - 开始绘制ROI"""
        if event.button() == Qt.LeftButton and self.current_frame is not None:
            self.drawing_roi = True
            self.roi_start = event.pos()
            self.roi_rect = None
    
    def label_mouse_move(self, event):
        """鼠标移动 - 更新ROI矩形"""
        if self.drawing_roi:
            current_pos = event.pos()
            self.roi_rect = QRect(self.roi_start, current_pos).normalized()
            self.image_label.update()
    
    def label_mouse_release(self, event):
        """鼠标释放 - 完成ROI绘制"""
        if self.drawing_roi and self.roi_rect and self.current_frame is not None:
            # 获取当前显示的pixmap
            pixmap = self.image_label.pixmap()
            if pixmap and not pixmap.isNull():
                label_size = self.image_label.size()
                pix_size = pixmap.size()
                
                # 计算图像在label中的偏移
                offset_x = (label_size.width() - pix_size.width()) // 2
                offset_y = (label_size.height() - pix_size.height()) // 2
                
                # 将label坐标映射到pixmap坐标
                roi_in_pix = QRect(
                    self.roi_rect.x() - offset_x,
                    self.roi_rect.y() - offset_y,
                    self.roi_rect.width(),
                    self.roi_rect.height()
                ).intersected(QRect(0, 0, pix_size.width(), pix_size.height()))
                
                # 映射到原始图像坐标
                scale_x = self.current_frame.shape[1] / pix_size.width()
                scale_y = self.current_frame.shape[0] / pix_size.height()
                roi_in_original = QRect(
                    int(roi_in_pix.x() * scale_x),
                    int(roi_in_pix.y() * scale_y),
                    int(roi_in_pix.width() * scale_x),
                    int(roi_in_pix.height() * scale_y)
                )
                
                if roi_in_original.width() > 10 and roi_in_original.height() > 10:
                    # 添加新ROI（最多4个）
                    if len(self.rois) >= 4:
                        self.rois.pop(0)
                    self.rois.append(roi_in_original)
                    if self.focus_widget.is_focus_enabled():
                        self.focus_widget.reset_peak()
        
        self.drawing_roi = False
        self.roi_rect = None
        self.image_label.update()

    # =====================================================
    # 打开关闭相机
    # =====================================================
    def toggle_camera(self):
        if not self.timer.isActive():
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not self.capture.isOpened():
                QMessageBox.critical(self, "错误", "无法打开相机")
                return
            
            # 读取当前硬件参数
            brightness = int(self.capture.get(cv2.CAP_PROP_BRIGHTNESS))
            gain = int(self.capture.get(cv2.CAP_PROP_GAIN))
            
            # 尝试关闭自动曝光（可选）
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动模式
            self.capture.set(cv2.CAP_PROP_AUTO_WB, 0)          # 手动白平衡
            
            self.slider_brightness.setValue(brightness)
            self.slider_gain.setValue(gain)
            self.lbl_brightness.setText(str(brightness))
            self.lbl_gain.setText(str(gain))
            
            self.timer.start(30)
            self.btn_camera.setText("关闭相机")
            self.btn_save.setEnabled(True)
            
            # 延迟一下，等待相机初始化完成
            QTimer.singleShot(500, self.set_default_rois)
        else:
            self.timer.stop()
            if self.capture:
                self.capture.release()
            self.image_label.clear()
            self.image_label.setText("相机已关闭")
            self.btn_camera.setText("打开相机")
            self.btn_save.setEnabled(False)
            self.rois = []

    # =====================================================
    # 实时显示和清晰度计算
    # =====================================================
    def update_frame(self):
        if self.capture is None:
            return
        
        ret, frame = self.capture.read()
        if not ret:
            return
        
        self.current_frame = frame.copy()
        
        # 计算清晰度分数（仅当对焦检测开启时）
        scores = []
        if self.focus_widget.is_focus_enabled() and self.rois:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            for rect in self.rois:
                x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
                # 确保ROI在图像范围内
                if x < 0: x = 0
                if y < 0: y = 0
                if x + w > frame.shape[1]: w = frame.shape[1] - x
                if y + h > frame.shape[0]: h = frame.shape[0] - y
                if w <= 0 or h <= 0:
                    scores.append(0.0)
                    continue
                
                roi_gray = gray[y:y+h, x:x+w]
                if roi_gray.size == 0:
                    scores.append(0.0)
                    continue
                
                # 根据选择的算法计算分数
                if self.focus_widget.get_current_algorithm() == 0:
                    score = calculate_tenengrad(roi_gray)
                else:
                    score = calculate_laplacian_variance(roi_gray)
                scores.append(score)
        
        # 补齐到4个ROI
        while len(scores) < 4:
            scores.append(0.0)
        
        # 更新评分面板
        self.focus_widget.update_scores(scores)
        
        # 执行CMOS对齐检测（如果开启）
        if self.alignment_widget.is_alignment_enabled():
            self.alignment_widget.detect_cross(frame)
        
        # 在图像上绘制ROI和分数
        display_img = frame.copy()
        
        # 绘制ROI（仅当有ROI时）
        for i, rect in enumerate(self.rois):
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if i < 4 and scores[i] > 0:
                cv2.putText(display_img, f"ROI{i+1}: {scores[i]:.1f}", 
                           (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 显示综合评分（如果对焦检测开启）
        if self.focus_widget.is_focus_enabled() and scores:
            combined = np.mean(scores)
            cv2.putText(display_img, f"Focus Score: {combined:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_img, "Focus Detection OFF", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # 绘制对齐校准叠加层（如果开启）
        if self.alignment_widget.is_alignment_enabled():
            display_img = self.alignment_widget.draw_alignment_overlay(display_img)
        else:
            # 如果对齐检测关闭，仍可显示图像中心（可选）
            h, w = display_img.shape[:2]
            center_x, center_y = w // 2, h // 2
            cv2.drawMarker(display_img, (center_x, center_y), (128, 128, 128), 
                          cv2.MARKER_CROSS, 20, 1)
        
        # 转换并显示
        rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        pix = pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

    # =====================================================
    # 相机参数调节
    # =====================================================
    def change_brightness(self, value):
        self.lbl_brightness.setText(str(value))
        if self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_BRIGHTNESS, value)

    def change_gain(self, value):
        self.lbl_gain.setText(str(value))
        if self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_GAIN, value)

    # =====================================================
    # 保存图片
    # =====================================================
    def save_image(self):
        if self.current_frame is None:
            return
        
        filename = time.strftime("IMG_%Y%m%d_%H%M%S.jpg")
        full_path = os.path.join(self.save_dir, filename)
        
        # 保存原始图像
        ok = cv2.imwrite(full_path, self.current_frame)
        
        if ok:
            # 同时保存评分信息和对齐信息
            score_file = full_path.replace('.jpg', '_scores.txt')
            with open(score_file, 'w') as f:
                f.write(f"Image: {filename}\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Focus Detection: {'ON' if self.focus_widget.is_focus_enabled() else 'OFF'}\n")
                f.write(f"Algorithm: {'Laplacian' if self.focus_widget.get_current_algorithm() == 1 else 'Tenengrad'}\n")
                f.write("ROI Scores:\n")
                for i, rect in enumerate(self.rois):
                    f.write(f"  ROI{i+1}: ({rect.x()}, {rect.y()}, {rect.width()}, {rect.height()})\n")
                f.write(f"Combined Score: {self.focus_widget.combined_label.text()}\n")
                f.write("\n")
                f.write("=== CMOS Alignment Result ===\n")
                result = self.alignment_widget.get_current_result()
                if result and result.center:
                    f.write(f"Cross Center: ({result.center[0]:.1f}, {result.center[1]:.1f})\n")
                    h, w = self.current_frame.shape[:2]
                    f.write(f"Image Center: ({w//2}, {h//2})\n")
                    f.write(f"Center Error: {result.center_error:.2f} pixels\n")
                    f.write(f"Rotation Error: {result.angle_error:.2f} deg\n")
                else:
                    f.write("No cross detected\n")
            
            QMessageBox.information(self, "成功", f"已保存:\n{full_path}\n评分信息: {score_file}")
        else:
            QMessageBox.critical(self, "错误", "保存失败")

    # =====================================================
    # 打开驱动属性页
    # =====================================================
    def open_camera_setting(self):
        if self.capture and self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_SETTINGS, 0)

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