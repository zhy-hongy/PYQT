"""CMOS 位置对准软件 — 白色十字线检测 + 角度偏差 + 中心偏移（改进版）"""

import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QGridLayout, QSplitter,
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QStatusBar,
    QTextEdit, QFrame, QCheckBox, QSlider,
)


# ============================================================
#  图像处理模块 — 白色十字线检测（改进版）
# ============================================================

def detect_white_crosshair_v2(
    image: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    angle_tol: float = 15.0,
    min_line_length: float = 50.0,
    max_line_length: float = 1000.0,
    white_threshold: int = 200,
    use_color_filter: bool = True,
    # 新增：十字线特征筛选参数
    max_distance_from_center: float = 200.0,  # 十字线交点离图像中心的最大距离（像素）
    min_perpendicular_score: float = 0.8,    # 最小垂直度得分（0-1）
    max_line_count: int = 20,                # 最多考虑的直线数量
):
    """检测白色十字线，使用更严格的十字线特征筛选
    
    Args:
        max_distance_from_center: 十字线交点离图像中心的最大距离，超过则认为是干扰
        min_perpendicular_score: 最小垂直度得分，用于筛选真正的十字线
        max_line_count: 最多考虑的直线数量（按长度排序取前N条）
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    result = {
        'h_line': None, 'v_line': None,
        'intersection': None, 'image_center': center,
        'h_angle': None, 'v_angle': None,
        'cross_angle': None, 'distance': None,
        'dx': None, 'dy': None, 'edges': None,
        'h_count': 0, 'v_count': 0, 'message': '',
        'white_mask': None,
        'candidate_lines': [],  # 保存候选直线用于调试
    }
    
    # 1. 提取白色区域
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if use_color_filter:
        _, white_mask = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        result['white_mask'] = white_mask
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_all = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)
        edges = cv2.bitwise_and(edges_all, white_mask)
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)
    
    result['edges'] = edges
    
    # 2. 检测所有直线
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=int(min_line_length),
        maxLineGap=20,
    )
    
    if lines is None:
        result['message'] = '未检测到直线'
        return result
    
    # 3. 收集所有线段并计算特征
    all_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_length = np.hypot(x2 - x1, y2 - y1)
        
        # 长度过滤
        if line_length < min_line_length or line_length > max_line_length:
            continue
        
        # 计算线段角度
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        while angle > 90:
            angle -= 180
        while angle <= -90:
            angle += 180
        
        # 计算线段中心点
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # 计算线段到图像中心的距离
        dist_to_center = np.hypot(cx - center[0], cy - center[1])
        
        all_segments.append({
            'coords': (x1, y1, x2, y2),
            'length': line_length,
            'angle': angle,
            'center': (cx, cy),
            'dist_to_center': dist_to_center,
        })
    
    # 4. 按长度排序，只保留最长的N条直线（减少干扰）
    all_segments.sort(key=lambda x: x['length'], reverse=True)
    all_segments = all_segments[:max_line_count]
    
    # 5. 分类为水平和竖直候选
    h_candidates = [s for s in all_segments if abs(s['angle']) <= angle_tol]
    v_candidates = [s for s in all_segments if abs(abs(s['angle']) - 90) <= angle_tol]
    
    result['h_count'] = len(h_candidates)
    result['v_count'] = len(v_candidates)
    result['candidate_lines'] = all_segments
    
    if len(h_candidates) < 1 or len(v_candidates) < 1:
        result['message'] = f"检测到 {len(h_candidates)} 水平, {len(v_candidates)} 竖直（不足）"
        return result
    
    # 6. 尝试所有水平-竖直组合，找到最优的十字线
    best_score = -1
    best_h_line = None
    best_v_line = None
    best_intersection = None
    
    for h_seg in h_candidates[:5]:  # 只考虑前5条最长的水平线
        for v_seg in v_candidates[:5]:  # 只考虑前5条最长的竖直线
            # 计算两条线的交点
            x1, y1, x2, y2 = h_seg['coords']
            x3, y3, x4, y4 = v_seg['coords']
            
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-8:
                continue
            
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            
            # 检查交点是否在线段范围内（允许一定容差）
            def point_near_segment(px, py, x1, y1, x2, y2, tolerance=20):
                # 计算点到线段的距离
                line_len = np.hypot(x2 - x1, y2 - y1)
                if line_len < 1e-6:
                    return False
                # 投影参数
                t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_len ** 2)
                if t < 0:
                    proj_x, proj_y = x1, y1
                elif t > 1:
                    proj_x, proj_y = x2, y2
                else:
                    proj_x = x1 + t * (x2 - x1)
                    proj_y = y1 + t * (y2 - y1)
                dist = np.hypot(px - proj_x, py - proj_y)
                return dist <= tolerance
            
            # 检查交点是否接近两条线段
            on_h = point_near_segment(px, py, x1, y1, x2, y2, tolerance=30)
            on_v = point_near_segment(px, py, x3, y3, x4, y4, tolerance=30)
            
            # 计算交点离图像中心的距离
            dist_to_center = np.hypot(px - center[0], py - center[1])
            
            # 计算垂直度得分
            angle_diff = abs(abs(h_seg['angle'] - v_seg['angle']) - 90)
            perpendicular_score = max(0, 1 - angle_diff / 45.0)
            
            # 综合评分：长度 + 垂直度 + 交点位置
            h_len_score = min(1.0, h_seg['length'] / 200.0)
            v_len_score = min(1.0, v_seg['length'] / 200.0)
            center_score = max(0, 1 - dist_to_center / max_distance_from_center)
            
            # 如果交点离中心太远，降低评分
            if dist_to_center > max_distance_from_center:
                center_score *= 0.1
            
            # 如果不交于线段上，降低评分
            intersection_score = 1.0 if (on_h and on_v) else 0.3
            
            total_score = (h_len_score + v_len_score) * 0.3 + perpendicular_score * 0.3 + center_score * 0.2 + intersection_score * 0.2
            
            if total_score > best_score:
                best_score = total_score
                best_h_line = h_seg['coords']
                best_v_line = v_seg['coords']
                best_intersection = (float(px), float(py))
    
    if best_h_line is None or best_v_line is None:
        result['message'] = f'未找到合适的十字线组合（最佳得分={best_score:.2f}）'
        return result
    
    # 7. 使用检测到的最优直线进行拟合（扩展到整幅图像）
    def extend_line(x1, y1, x2, y2):
        """将线段扩展到整幅图像范围"""
        # 计算直线参数
        if abs(x2 - x1) < 1e-6:
            # 垂直线
            x = x1
            return (x, 0, x, h)
        # 计算斜率和截距
        k = (y2 - y1) / (x2 - x1)
        b = y1 - k * x1
        # 计算与图像边界的交点
        x_left = 0
        y_left = b
        x_right = w
        y_right = k * w + b
        return (x_left, y_left, x_right, y_right)
    
    # 扩展直线到整幅图像
    hx1, hy1, hx2, hy2 = best_h_line
    vx1, vy1, vx2, vy2 = best_v_line
    
    extended_h = extend_line(hx1, hy1, hx2, hy2)
    extended_v = extend_line(vx1, vy1, vx2, vy2)
    
    # 重新计算扩展后的交点
    hx1, hy1, hx2, hy2 = extended_h
    vx1, vy1, vx2, vy2 = extended_v
    
    denom = (hx1 - hx2) * (vy1 - vy2) - (hy1 - hy2) * (vx1 - vx2)
    if abs(denom) > 1e-8:
        px = ((hx1 * hy2 - hy1 * hx2) * (vx1 - vx2) - (hx1 - hx2) * (vx1 * vy2 - vy1 * vx2)) / denom
        py = ((hx1 * hy2 - hy1 * hx2) * (vy1 - vy2) - (hy1 - hy2) * (vx1 * vy2 - vy1 * vx2)) / denom
        best_intersection = (float(px), float(py))
    
    # 计算角度
    h_angle = np.degrees(np.arctan2(hy2 - hy1, hx2 - hx1))
    while h_angle > 90:
        h_angle -= 180
    while h_angle <= -90:
        h_angle += 180
    
    v_angle = np.degrees(np.arctan2(vy2 - vy1, vx2 - vx1))
    while v_angle > 90:
        v_angle -= 180
    while v_angle <= -90:
        v_angle += 180
    
    cross_angle = abs(h_angle - v_angle)
    if cross_angle > 90:
        cross_angle = 180 - cross_angle
    
    dx = best_intersection[0] - center[0]
    dy = best_intersection[1] - center[1]
    distance = float(np.hypot(dx, dy))
    
    result.update({
        'h_line': extended_h,
        'v_line': extended_v,
        'intersection': best_intersection,
        'h_angle': float(h_angle),
        'v_angle': float(v_angle),
        'cross_angle': float(cross_angle),
        'distance': distance,
        'dx': float(dx),
        'dy': float(dy),
        'message': f'检测完成（得分={best_score:.2f}）',
    })
    
    # 记录检测到的线段信息
    result['detected_h_count'] = len(h_candidates)
    result['detected_v_count'] = len(v_candidates)
    
    return result


def draw_result_v2(image: np.ndarray, det: dict, ppm: float = 0, show_mask: bool = False, show_candidates: bool = False) -> np.ndarray:
    """在图像上绘制检测结果标注（改进版）"""
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    if show_mask and det.get('white_mask') is not None:
        out = cv2.cvtColor(det['white_mask'], cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()
    
    h, w = out.shape[:2]
    center = det['image_center']
    
    # 字体加载
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "simhei.ttf",
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 18)
                break
            except:
                continue
    if font is None:
        font = ImageFont.load_default()
    
    def draw_chinese(img_bgr, text, pos, color_bgr, font_size=None):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        if font_size is None:
            used_font = font
        else:
            try:
                used_font = ImageFont.truetype(font.path, font_size) if hasattr(font, 'path') else font
            except:
                used_font = font
        
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text((pos[0]-1, pos[1]-1), text, font=used_font, fill=(0,0,0))
        draw.text((pos[0]+1, pos[1]+1), text, font=used_font, fill=(0,0,0))
        draw.text(pos, text, font=used_font, fill=color_rgb)
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 绘制候选线段（调试用）
    if show_candidates and det.get('candidate_lines'):
        for seg in det['candidate_lines']:
            x1, y1, x2, y2 = seg['coords']
            # 用灰色虚线绘制候选线段
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 1, cv2.LINE_AA)
    
    # 绘制最终的十字线
    if det['h_line']:
        x1, y1, x2, y2 = det['h_line']
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 0), 3, cv2.LINE_AA)
        out = draw_chinese(out, 'X轴', (int(x2)-30, int(y2)-10), (255, 180, 0), font_size=16)
    
    if det['v_line']:
        x1, y1, x2, y2 = det['v_line']
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 3, cv2.LINE_AA)
        out = draw_chinese(out, 'Y轴', (int(x1)+10, int(y1)+25), (0, 200, 255), font_size=16)
    
    # 十字交点
    if det['intersection']:
        cx, cy = int(det['intersection'][0]), int(det['intersection'][1])
        cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        out = draw_chinese(out, '十字交点', (cx+12, cy-12), (0, 255, 255), font_size=16)
    
    # CMOS中心
    ccx, ccy = int(center[0]), int(center[1])
    cv2.drawMarker(out, (ccx, ccy), (0, 220, 0), cv2.MARKER_CROSS, 40, 2)
    cv2.circle(out, (ccx, ccy), 4, (0, 220, 0), -1, cv2.LINE_AA)
    out = draw_chinese(out, 'CMOS中心', (ccx+12, ccy+12), (0, 220, 0), font_size=16)
    
    # 连线
    if det['intersection'] and det['distance'] and det['distance'] > 0:
        cx, cy = int(det['intersection'][0]), int(det['intersection'][1])
        cv2.line(out, (cx, cy), (ccx, ccy), (255, 0, 255), 2, cv2.LINE_AA)
    
    # 数值标注
    y_offset = 30
    if det['h_angle'] is not None:
        out = draw_chinese(out, f'旋转误差: {det["h_angle"]:.3f}°', (12, y_offset), (255, 255, 100), font_size=16)
        y_offset += 28
    if det['distance'] is not None:
        txt = f'中心偏移: {det["distance"]:.2f} px'
        if ppm > 0:
            txt += f'  ({det["distance"] / ppm:.3f} mm)'
        out = draw_chinese(out, txt, (12, y_offset), (255, 255, 100), font_size=16)
        y_offset += 28
    if det['dx'] is not None:
        out = draw_chinese(out, f'dX={det["dx"]:+.2f}  dY={det["dy"]:+.2f} px', (12, y_offset), (255, 255, 100), font_size=16)
        y_offset += 28
    if det['cross_angle'] is not None:
        out = draw_chinese(out, f'X-Y 夹角: {det["cross_angle"]:.2f}°', (12, y_offset), (255, 255, 100), font_size=16)
    
    return out


# ============================================================
#  PyQt5 主界面
# ============================================================

class CmosAligner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CMOS 位置对准软件 - 白色十字线检测（改进版）')
        self.resize(1280, 860)
        
        self._image = None
        self._result = None
        self._result_image = None
        
        self._setup_ui()
        
        # 摄像头
        self._camera = None
        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self._grab_frame)
        self._cam_active = False
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        
        # ===== 左侧：图像显示 =====
        self.viewer = QLabel('加载图片或打开摄像头')
        self.viewer.setAlignment(Qt.AlignCenter)
        self.viewer.setMinimumSize(800, 600)
        self.viewer.setStyleSheet(
            'background-color: #1e1e1e; color: #888;'
            ' border: 1px solid #333; font-size: 16px;'
        )
        root.addWidget(self.viewer, stretch=3)
        
        # ===== 右侧：控制面板 =====
        panel = QVBoxLayout()
        
        # -- 图像源 --
        grp_src = QGroupBox('图像源')
        lay_src = QVBoxLayout(grp_src)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel('摄像头:'))
        self.spin_cam = QSpinBox()
        self.spin_cam.setRange(0, 10)
        row1.addWidget(self.spin_cam)
        lay_src.addLayout(row1)
        row2 = QHBoxLayout()
        self.btn_cam = QPushButton('打开摄像头')
        self.btn_cam.clicked.connect(self._toggle_camera)
        row2.addWidget(self.btn_cam)
        self.btn_capture = QPushButton('拍照')
        self.btn_capture.clicked.connect(self._capture)
        self.btn_capture.setEnabled(False)
        row2.addWidget(self.btn_capture)
        lay_src.addLayout(row2)
        row3 = QHBoxLayout()
        self.btn_load = QPushButton('加载图片')
        self.btn_load.clicked.connect(self._load_image)
        row3.addWidget(self.btn_load)
        self.btn_save = QPushButton('保存结果')
        self.btn_save.clicked.connect(self._save_result)
        row3.addWidget(self.btn_save)
        lay_src.addLayout(row3)
        panel.addWidget(grp_src)
        
        # -- 白色十字线检测参数 --
        grp_white = QGroupBox('白色十字线检测')
        lay_white = QGridLayout(grp_white)
        
        lay_white.addWidget(QLabel('白色阈值:'), 0, 0)
        self.slider_white = QSlider(Qt.Horizontal)
        self.slider_white.setRange(100, 255)
        self.slider_white.setValue(200)
        self.slider_white.valueChanged.connect(self._on_white_threshold_changed)
        lay_white.addWidget(self.slider_white, 0, 1)
        self.lbl_white_val = QLabel('200')
        lay_white.addWidget(self.lbl_white_val, 0, 2)
        
        self.chk_color_filter = QCheckBox('启用白色区域过滤')
        self.chk_color_filter.setChecked(True)
        lay_white.addWidget(self.chk_color_filter, 1, 0, 1, 3)
        
        self.chk_show_mask = QCheckBox('显示白色掩码（调试）')
        self.chk_show_mask.setChecked(False)
        lay_white.addWidget(self.chk_show_mask, 2, 0, 1, 3)
        
        self.chk_show_candidates = QCheckBox('显示候选线段（调试）')
        self.chk_show_candidates.setChecked(False)
        lay_white.addWidget(self.chk_show_candidates, 3, 0, 1, 3)
        
        panel.addWidget(grp_white)
        
        # -- 检测参数 --
        grp_params = QGroupBox('检测参数')
        lay_params = QGridLayout(grp_params)
        
        lay_params.addWidget(QLabel('Canny 低阈值:'), 0, 0)
        self.spin_canny_lo = QSpinBox()
        self.spin_canny_lo.setRange(10, 500)
        self.spin_canny_lo.setValue(50)
        lay_params.addWidget(self.spin_canny_lo, 0, 1)
        
        lay_params.addWidget(QLabel('Canny 高阈值:'), 1, 0)
        self.spin_canny_hi = QSpinBox()
        self.spin_canny_hi.setRange(50, 500)
        self.spin_canny_hi.setValue(150)
        lay_params.addWidget(self.spin_canny_hi, 1, 1)
        
        lay_params.addWidget(QLabel('Hough 阈值:'), 2, 0)
        self.spin_hough = QSpinBox()
        self.spin_hough.setRange(10, 500)
        self.spin_hough.setValue(80)
        lay_params.addWidget(self.spin_hough, 2, 1)
        
        lay_params.addWidget(QLabel('角度容差:'), 3, 0)
        self.spin_angle = QSpinBox()
        self.spin_angle.setRange(5, 45)
        self.spin_angle.setValue(15)
        self.spin_angle.setSuffix('°')
        lay_params.addWidget(self.spin_angle, 3, 1)
        
        lay_params.addWidget(QLabel('最小线长(px):'), 4, 0)
        self.spin_min_len = QSpinBox()
        self.spin_min_len.setRange(10, 500)
        self.spin_min_len.setValue(50)
        lay_params.addWidget(self.spin_min_len, 4, 1)
        
        lay_params.addWidget(QLabel('最大线长(px):'), 5, 0)
        self.spin_max_len = QSpinBox()
        self.spin_max_len.setRange(50, 2000)
        self.spin_max_len.setValue(1000)
        self.spin_max_len.setToolTip('0表示不限制')
        lay_params.addWidget(self.spin_max_len, 5, 1)
        
        # 十字线特征参数
        lay_params.addWidget(QLabel('最大交点偏移(px):'), 6, 0)
        self.spin_max_offset = QSpinBox()
        self.spin_max_offset.setRange(50, 500)
        self.spin_max_offset.setValue(200)
        self.spin_max_offset.setToolTip('十字线交点离图像中心的最大距离')
        lay_params.addWidget(self.spin_max_offset, 6, 1)
        
        lay_params.addWidget(QLabel('最大候选直线数:'), 7, 0)
        self.spin_max_lines = QSpinBox()
        self.spin_max_lines.setRange(5, 50)
        self.spin_max_lines.setValue(20)
        lay_params.addWidget(self.spin_max_lines, 7, 1)
        
        lay_params.addWidget(QLabel('像素/mm:'), 8, 0)
        self.spin_ppm = QDoubleSpinBox()
        self.spin_ppm.setRange(0, 1000)
        self.spin_ppm.setDecimals(2)
        self.spin_ppm.setSingleStep(0.1)
        lay_params.addWidget(self.spin_ppm, 8, 1)
        
        self.btn_redetect = QPushButton('重新检测')
        self.btn_redetect.clicked.connect(self._detect)
        lay_params.addWidget(self.btn_redetect, 9, 0, 1, 2)
        
        panel.addWidget(grp_params)
        
        # -- 结果 --
        grp_res = QGroupBox('对准结果')
        lay_res = QVBoxLayout(grp_res)
        self.lbl_angle = QLabel('旋转误差 (Δθ): --')
        self.lbl_dist = QLabel('中心偏移: --')
        self.lbl_dx = QLabel('dX: --')
        self.lbl_dy = QLabel('dY: --')
        self.lbl_cross_angle = QLabel('X-Y 夹角: --')
        self.lbl_lines = QLabel('检测线段: --')
        self.lbl_score = QLabel('匹配得分: --')
        self.lbl_status = QLabel('状态: 就绪')
        for lb in (self.lbl_angle, self.lbl_dist, self.lbl_dx, self.lbl_dy,
                   self.lbl_cross_angle, self.lbl_lines, self.lbl_score, self.lbl_status):
            lb.setWordWrap(True)
            lay_res.addWidget(lb)
        panel.addWidget(grp_res)
        
        # -- 图例 --
        grp_legend = QGroupBox('图例')
        lay_legend = QVBoxLayout(grp_legend)
        for txt in ('橙色粗线 — 最终X轴', '青色粗线 — 最终Y轴',
                    '灰色细线 — 候选线段（调试模式）',
                    '黄色十字 — 十字交点', '绿色十字 — CMOS图像中心',
                    '品红虚线 — 中点到交点连线'):
            lay_legend.addWidget(QLabel(txt))
        panel.addWidget(grp_legend)
        
        # -- 日志 --
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        panel.addWidget(QLabel('日志:'))
        panel.addWidget(self._log)
        
        panel.addStretch()
        root.addLayout(panel, stretch=1)
        
        self.statusBar().showMessage('就绪')
    
    def _on_white_threshold_changed(self, value):
        self.lbl_white_val.setText(str(value))
        if self._image is not None:
            self._detect()
    
    def _log_msg(self, msg):
        self._log.append(msg)
    
    def _detect(self):
        """对当前图像运行白色十字线检测（改进版）"""
        if self._image is None:
            return
        
        max_len = self.spin_max_len.value()
        if max_len == 0:
            max_len = float('inf')
        
        det = detect_white_crosshair_v2(
            self._image,
            canny_low=self.spin_canny_lo.value(),
            canny_high=self.spin_canny_hi.value(),
            hough_threshold=self.spin_hough.value(),
            angle_tol=float(self.spin_angle.value()),
            min_line_length=float(self.spin_min_len.value()),
            max_line_length=float(max_len),
            white_threshold=self.slider_white.value(),
            use_color_filter=self.chk_color_filter.isChecked(),
            max_distance_from_center=float(self.spin_max_offset.value()),
            max_line_count=self.spin_max_lines.value(),
        )
        self._result = det
        self._update_ui()
        
        # 提取得分信息
        if '检测完成' in det['message']:
            # 提取得分
            import re
            match = re.search(r'得分=([\d.]+)', det['message'])
            if match:
                score = float(match.group(1))
                self.lbl_score.setText(f'匹配得分: {score:.2f}')
        else:
            self.lbl_score.setText('匹配得分: --')
        
        filter_msg = f'白色阈值={self.slider_white.value()}'
        if self.chk_color_filter.isChecked():
            filter_msg += ', 颜色过滤=启用'
        else:
            filter_msg += ', 颜色过滤=禁用'
        self._log_msg(f'{det["message"]} | {filter_msg}')
    
    def _update_ui(self):
        """更新结果标签和图像显示"""
        det = self._result
        if det is None:
            return
        
        ppm = self.spin_ppm.value()
        
        if det['h_angle'] is not None:
            self.lbl_angle.setText(f'旋转误差 (Δθ): {det["h_angle"]:+.3f}°')
        else:
            self.lbl_angle.setText('旋转误差 (Δθ): --')
        
        if det['distance'] is not None:
            txt = f'中心偏移: {det["distance"]:.2f} px'
            if ppm > 0:
                txt += f'  ({det["distance"] / ppm:.3f} mm)'
            self.lbl_dist.setText(txt)
        else:
            self.lbl_dist.setText('中心偏移: --')
        
        if det['dx'] is not None:
            self.lbl_dx.setText(f'dX: {det["dx"]:+.2f} px')
            self.lbl_dy.setText(f'dY: {det["dy"]:+.2f} px')
        else:
            self.lbl_dx.setText('dX: --')
            self.lbl_dy.setText('dY: --')
        
        if det['cross_angle'] is not None:
            self.lbl_cross_angle.setText(f'X-Y 夹角: {det["cross_angle"]:.2f}°')
        else:
            self.lbl_cross_angle.setText('X-Y 夹角: --')
        
        self.lbl_lines.setText(f'检测到: {det["h_count"]} 水平候选, {det["v_count"]} 竖直候选')
        self.lbl_status.setText(f'状态: {det["message"]}')
        self.statusBar().showMessage(det['message'], 3000)
        
        # 绘制标注图像
        if self._image is not None:
            annotated = draw_result_v2(
                self._image, det, ppm, 
                show_mask=self.chk_show_mask.isChecked(),
                show_candidates=self.chk_show_candidates.isChecked()
            )
            self._result_image = annotated
            self._show_image(annotated)
    
    def _show_image(self, bgr: np.ndarray):
        """在 QLabel 上显示 BGR 图像"""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.viewer.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.viewer.setPixmap(scaled)
    
    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', '',
            'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, '错误', '无法读取图片')
            return
        self._stop_camera()
        self._image = img
        self._detect()
        self._log_msg(f'加载: {path}')
    
    def _save_result(self):
        if self._result_image is None:
            QMessageBox.information(self, '提示', '暂无结果')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '保存结果', 'cmos_alignment.png',
            'PNG (*.png);;JPEG (*.jpg)')
        if path:
            cv2.imwrite(path, self._result_image)
            self._log_msg(f'已保存: {path}')
    
    def _toggle_camera(self):
        if not self._cam_active:
            idx = self.spin_cam.value()
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                QMessageBox.warning(self, '错误', f'无法打开摄像头 {idx}')
                return
            self._camera = cap
            self._cam_active = True
            self.btn_cam.setText('关闭摄像头')
            self.btn_capture.setEnabled(True)
            self._cam_timer.start(33)
            self._log_msg(f'已打开摄像头 {idx}')
        else:
            self._stop_camera()
    
    def _stop_camera(self):
        self._cam_timer.stop()
        if self._camera:
            self._camera.release()
            self._camera = None
        self._cam_active = False
        self.btn_cam.setText('打开摄像头')
        self.btn_capture.setEnabled(False)
    
    def _grab_frame(self):
        if self._camera is None:
            return
        ok, frame = self._camera.read()
        if ok:
            self._image = frame
            self._detect()
    
    def _capture(self):
        if self._image is not None:
            self._stop_camera()
            self._log_msg('已拍照（停止实时）')
    
    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = CmosAligner()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()