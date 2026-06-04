import sys
import cv2
import numpy as np
import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QGroupBox, 
                             QGridLayout, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QRect, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import pyqtgraph as pg

# ------------------------- 清晰度评价与 RMS 归一化 -------------------------
def calculate_tenengrad_rms(image_gray):
    """
    计算 Tenengrad 均方根得分
    1. 使用高斯模糊滤除高频电子噪点
    2. 计算 Sobel 梯度
    3. 采用均方根（RMS）形式降低平方带来的数据剧烈跳动，提高人眼感知的线性度
    """
    # 滤除微小噪点干扰
    blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
    
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # 均方根计算
    mean_square = np.sum(sobel_x**2 + sobel_y**2) / image_gray.size
    score_rms = np.sqrt(mean_square)
    return score_rms

# ------------------------- 安全的相机采集线程 -------------------------
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = None
        
    def run(self):
        # 使用 DSHOW 后端打开
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("无法打开相机，请检查设备连接")
            return
        
        # 设置常见工业级/商用分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 代表关闭自动曝光，切入手动模式
        
        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                # 必须发送深拷贝，防止跨线程因内存复用引发的冲突闪退
                self.change_pixmap_signal.emit(frame.copy())
            else:
                self.msleep(10)
        self.cap.release()
        
    def stop(self):
        self._run_flag = False
        self.wait()

# ------------------------- 自定义交互式 QLabel -------------------------
class ROILabel(QLabel):
    """支持鼠标拖动实时绘制虚线框的 Label"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.start_p = QPoint()
        self.end_p = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_p = event.pos()
            self.end_p = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_p = event.pos()
            self.update()  # 触发 paintEvent 刷新视窗中的绿框

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if hasattr(self.window(), 'handle_roi_release'):
                self.window().handle_roi_release(self.start_p, self.end_p)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing:
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            rect = QRect(self.start_p, self.end_p)
            painter.drawRect(rect)

# ------------------------- 主视窗界面 -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("沙姆相机对焦辅助系统 (动态归一化版)")
        self.setGeometry(100, 100, 1400, 850)
        
        # ROI 相关配置
        self.rois = []
        self.roi_names = ["ROI1", "ROI2", "ROI3", "ROI4"]
        
        # 核心：为每个 ROI 区域维护一套独立的历史极值，用于执行 Min-Max 归一化映射
        self.roi_min_max = {name: {"min": float('inf'), "max": float('-inf')} for name in self.roi_names}
        self.score_history = {name: [] for name in self.roi_names}
        self.history_length = 120
        self._peak_combined = 0.0  # 记录历史最高综合得分
        
        self.current_frame = None
        self.waiting_for_frame = True  # 控制首帧引导默认 ROI
        
        self.init_ui()
        
        # 启动相机流
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_frame)
        self.camera_thread.start()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # --------- 左侧：视频及框选交互区 ---------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.image_label = ROILabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #121212;")
        self.image_label.setMinimumSize(720, 540)
        left_layout.addWidget(self.image_label)
        
        btn_layout = QHBoxLayout()
        self.btn_reset_roi = QPushButton("恢复默认四角 ROI")
        self.btn_reset_roi.clicked.connect(self.set_default_rois_flag)
        self.btn_clear_roi = QPushButton("清除当前所有 ROI")
        self.btn_clear_roi.clicked.connect(self.clear_rois)
        btn_layout.addWidget(self.btn_reset_roi)
        btn_layout.addWidget(self.btn_clear_roi)
        left_layout.addLayout(btn_layout)
        main_layout.addWidget(left_widget, 2)
        
        # --------- 右侧：可读性评分面板 ---------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 实时百分比面板
        score_group = QGroupBox("实时相对清晰度 (归一化)")
        score_layout = QGridLayout()
        self.score_labels = {}
        self.score_bars = {}
        
        for i, name in enumerate(self.roi_names):
            score_layout.addWidget(QLabel(f"{name} 状态:"), i, 0)
            self.score_labels[name] = QLabel("0.0%")
            self.score_labels[name].setStyleSheet("font-weight: bold; color: #00FF00;")
            score_layout.addWidget(self.score_labels[name], i, 1)
            
            self.score_bars[name] = QProgressBar()
            self.score_bars[name].setRange(0, 100)
            self.score_bars[name].setTextVisible(False)
            score_layout.addWidget(self.score_bars[name], i, 2)
            
        score_layout.addWidget(QLabel("--------------------------------------------------"), len(self.roi_names), 0, 1, 3)
        
        # 综合评分
        self.combined_label = QLabel("综合焦面覆盖率: 0.0%")
        self.combined_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.combined_bar = QProgressBar()
        self.combined_bar.setRange(0, 100)
        score_layout.addWidget(self.combined_label, len(self.roi_names) + 1, 0, 1, 2)
        score_layout.addWidget(self.combined_bar, len(self.roi_names) + 1, 2)
        
        score_group.setLayout(score_layout)
        right_layout.addWidget(score_group)
        
        # 实时波形曲线
        curve_group = QGroupBox("各区清晰度趋势收敛曲线")
        curve_layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1e1e1e')
        self.plot_widget.setLabel('left', '清晰度归一化值 (%)')
        self.plot_widget.setLabel('bottom', '时间轴(帧)')
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.addLegend()
        
        self.curves = {}
        colors = ['#FF5555', '#55FF55', '#5555FF', '#55FFFF']
        for i, name in enumerate(self.roi_names):
            self.curves[name] = self.plot_widget.plot(pen=pg.mkPen(color=colors[i], width=2), name=name)
        curve_layout.addWidget(self.plot_widget)
        curve_group.setLayout(curve_layout)
        right_layout.addWidget(curve_group)
        
        # 控制面板
        control_group = QGroupBox("系统控制")
        control_layout = QHBoxLayout()
        self.btn_save_snapshot = QPushButton("保存图像快照")
        self.btn_save_snapshot.clicked.connect(self.save_snapshot)
        self.btn_reset_curve = QPushButton("重置记忆与极值")
        self.btn_reset_curve.clicked.connect(self.reset_curves)
        control_layout.addWidget(self.btn_save_snapshot)
        control_layout.addWidget(self.btn_reset_curve)
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)
        
        main_layout.addWidget(right_widget, 1)

    def set_default_rois_flag(self):
        self.waiting_for_frame = True

    def update_default_rois(self, frame_h, frame_w):
        """依实际分辨率自动分配边缘四角 ROI"""
        roi_w = min(160, frame_w // 5)
        roi_h = min(160, frame_h // 5)
        margin = 40
        self.rois = [
            QRect(margin, margin, roi_w, roi_h), # 左上
            QRect(frame_w - roi_w - margin, margin, roi_w, roi_h), # 右上
            QRect(margin, frame_h - roi_h - margin, roi_w, roi_h), # 左下
            QRect(frame_w - roi_w - margin, frame_h - roi_h - margin, roi_w, roi_h) # 右下
        ]
        self.waiting_for_frame = False
        self.reset_curves()

    def clear_rois(self):
        self.rois = []
        for name in self.roi_names:
            self.score_labels[name].setText("0.0%")
            self.score_bars[name].setValue(0)
        self.combined_label.setText("综合焦面覆盖率: 0.0%")
        self.combined_bar.setValue(0)

    def handle_roi_release(self, start_p, end_p):
        """精准逆向映射：将 PyQt 渲染控件坐标系还原回真实的相机传感器坐标系"""
        if self.current_frame is None: return
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull(): return

        # 还原 KeepAspectRatio 机制下的四周黑边填充偏移量
        lbl_w, lbl_h = self.image_label.width(), self.image_label.height()
        pix_w, pix_h = pixmap.width(), pixmap.height()
        offset_x = (lbl_w - pix_w) // 2
        offset_y = (lbl_h - pix_h) // 2

        rect = QRect(start_p, end_p).normalized()
        # 裁剪掉超出有效图像区域外的鼠标轨迹
        roi_in_pix = rect.translated(-offset_x, -offset_y).intersected(QRect(0, 0, pix_w, pix_h))

        if roi_in_pix.width() > 15 and roi_in_pix.height() > 15:
            scale_x = self.current_frame.shape[1] / pix_w
            scale_y = self.current_frame.shape[0] / pix_h
            
            roi_org = QRect(int(roi_in_pix.x() * scale_x), int(roi_in_pix.y() * scale_y),
                            int(roi_in_pix.width() * scale_x), int(roi_in_pix.height() * scale_y))
            
            if len(self.rois) >= 4:
                self.rois.pop(0)  # 循环覆盖，最多允许 4 个 ROI
            self.rois.append(roi_org)
            self.reset_curves() # 重置极值以适应新选区的特征纹理

    def update_frame(self, frame):
        """主核心回调处理机制"""
        self.current_frame = frame
        h, w = frame.shape[:2]
        if self.waiting_for_frame:
            self.update_default_rois(h, w)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm_scores_list = []
        display_img = frame.copy()

        for i, rect in enumerate(self.rois):
            rx, ry, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()
            # 越界鲁棒性保护
            rx, ry = max(0, rx), max(0, ry)
            rw = min(w - rx, rw)
            rh = min(h - ry, rh)

            if rw <= 10 or rh <= 10: continue

            # 1. 截取灰度 ROI 区域并计算原始梯度的均方根
            roi_gray = gray[ry:ry+rh, rx:rx+rw]
            raw_score = calculate_tenengrad_rms(roi_gray)

            if i < len(self.roi_names):
                name = self.roi_names[i]
                
                # 2. 动态捕捉历史最低分与最高分
                if raw_score < self.roi_min_max[name]["min"]:
                    self.roi_min_max[name]["min"] = raw_score
                if raw_score > self.roi_min_max[name]["max"]:
                    self.roi_min_max[name]["max"] = raw_score
                
                # 3. 核心数学转换：执行 Min-Max 归一化，将绝对值转换成 0 - 100 相对百分比
                span = self.roi_min_max[name]["max"] - self.roi_min_max[name]["min"]
                if span > 1e-4:
                    norm_score = ((raw_score - self.roi_min_max[name]["min"]) / span) * 100
                else:
                    norm_score = 0.0
                
                norm_scores_list.append(norm_score)
                
                # 4. 刷新 UI 指示元件
                self.score_labels[name].setText(f"{norm_score:.1f}%")
                self.score_bars[name].setValue(int(norm_score))
                
                # 5. 更新右侧 PyQtGraph 曲线趋势
                self.score_history[name].append(norm_score)
                if len(self.score_history[name]) > self.history_length:
                    self.score_history[name].pop(0)
                self.curves[name].setData(self.score_history[name])

            # 在相机视窗画布上同步绘制出绿色边界框和实时数值
            cv2.rectangle(display_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            if i < len(self.roi_names):
                cv2.putText(display_img, f"{self.roi_names[i]}: {norm_score:.1f}%", (rx, ry - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 6. 计算和反映综合焦面覆盖状况
        if norm_scores_list:
            combined_percentage = np.mean(norm_scores_list)
            self.combined_label.setText(f"综合焦面覆盖率: {combined_percentage:.1f}%")
            self.combined_bar.setValue(int(combined_percentage))
            
            if combined_percentage > self._peak_combined:
                self._peak_combined = combined_percentage
                self.setWindowTitle(f"沙姆相机对焦辅助系统 ★ 历史最高综合峰值: {combined_percentage:.1f}% ★")

        # 7. 转译并渲染输出
        rgb_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def save_snapshot(self):
        """保存快照和元数据档案"""
        if self.current_frame is None: return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"snapshot_{ts}.png", self.current_frame)
        print(f"快照文件保存成功: snapshot_{ts}.png")

    def reset_curves(self):
        """重置所有记忆区，当你改变了相机倾角或大幅更换被摄物时，必须点此按钮"""
        self._peak_combined = 0.0
        self.setWindowTitle("沙姆相机对焦辅助系统 (动态归一化版)")
        self.roi_min_max = {name: {"min": float('inf'), "max": float('-inf')} for name in self.roi_names}
        for name in self.roi_names:
            self.score_history[name] = []
            self.curves[name].setData([])
            self.score_bars[name].setValue(0)
        self.combined_bar.setValue(0)

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

# ------------------------- 模块执行入口 -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())