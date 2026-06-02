"""CMOS 位置对准软件 — 十字线检测 + 角度偏差 + 中心偏移"""

import sys

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QGridLayout, QSplitter,
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QStatusBar,
    QTextEdit, QFrame,
)


# ============================================================
#  图像处理模块 — 十字线检测 + CMOS 对准计算
# ============================================================

def detect_crosshair(
    gray: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    min_length_ratio: float = 0.12,
    angle_tol: float = 15.0,
):
    """检测十字线，返回水平线/竖直线/交点/角度等信息。

    Returns:
        dict 包含 h_line, v_line, intersection, image_center,
              h_angle, v_angle, cross_angle, distance, dx, dy, message
    """
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    result = {
        'h_line': None, 'v_line': None,
        'intersection': None, 'image_center': center,
        'h_angle': None, 'v_angle': None,
        'cross_angle': None, 'distance': None,
        'dx': None, 'dy': None, 'edges': None,
        'h_count': 0, 'v_count': 0, 'message': '',
    }

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)
    result['edges'] = edges

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=int(min(h, w) * min_length_ratio),
        maxLineGap=20,
    )

    if lines is None:
        result['message'] = '未检测到直线'
        return result

    # 收集线段并按角度分类
    h_segs, v_segs = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if np.hypot(x2 - x1, y2 - y1) < 20:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # 归一化到 [-90, 90]
        while angle > 90:
            angle -= 180
        while angle <= -90:
            angle += 180
        a = abs(angle)
        if a <= angle_tol:
            h_segs.append((x1, y1, x2, y2))
        elif a >= 90 - angle_tol:
            v_segs.append((x1, y1, x2, y2))

    result['h_count'] = len(h_segs)
    result['v_count'] = len(v_segs)

    if not h_segs or not v_segs:
        result['message'] = f"检测到 {len(h_segs)} 水平, {len(v_segs)} 竖直（不足）"
        return result

    def fit_line(segments):
        """用中位数角度 + 中位数偏移拟合直线（比 SVD/fitLine 更鲁棒）。"""
        # 收集所有线段的中点和角度
        mids = []
        angles = []
        for x1, y1, x2, y2 in segments:
            mids.append([(x1 + x2) / 2, (y1 + y2) / 2])
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            while a > 90:
                a -= 180
            while a <= -90:
                a += 180
            angles.append(a)

        theta = np.radians(np.median(angles))
        vx, vy = np.cos(theta), np.sin(theta)   # 方向向量
        nx, ny = -vy, vx                        # 法向量

        mids_arr = np.array(mids)
        # 每个中点在法线方向上的投影
        dists = mids_arr[:, 0] * nx + mids_arr[:, 1] * ny
        d = np.median(dists)  # 直线到原点的有符号距离

        # 直线上距离原点最近的点
        px, py = d * nx, d * ny

        # 扩展到图像尺寸
        L = 2000.0
        return (px - vx * L, py - vy * L, px + vx * L, py + vy * L)

    h_line = fit_line(h_segs)
    v_line = fit_line(v_segs)

    # 计算交点
    x1, y1, x2, y2 = h_line
    x3, y3, x4, y4 = v_line
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        result['message'] = '十字线近似平行，无法计算交点'
        return result

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    # 水平线角度（相对水平轴）
    h_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    while h_angle > 90:
        h_angle -= 180
    while h_angle <= -90:
        h_angle += 180

    # 竖直线角度（相对水平轴），再换算成相对竖直轴的偏差
    v_angle = np.degrees(np.arctan2(y4 - y3, x4 - x3))
    while v_angle > 90:
        v_angle -= 180
    while v_angle <= -90:
        v_angle += 180

    # X-Y 夹角
    cross_angle = abs(h_angle - v_angle)
    if cross_angle > 90:
        cross_angle = 180 - cross_angle

    # 偏移
    dx = px - center[0]
    dy = py - center[1]
    distance = float(np.hypot(dx, dy))

    result.update({
        'h_line': h_line,
        'v_line': v_line,
        'intersection': (float(px), float(py)),
        'h_angle': float(h_angle),
        'v_angle': float(v_angle),
        'cross_angle': float(cross_angle),
        'distance': distance,
        'dx': float(dx),
        'dy': float(dy),
        'message': '检测完成',
    })
    return result

def draw_result(image: np.ndarray, det: dict, ppm: float = 0) -> np.ndarray:
    """在图像上绘制检测结果标注（支持中文）。"""
    from PIL import Image, ImageDraw, ImageFont
    import os

    out = image.copy()
    h, w = out.shape[:2]
    center = det['image_center']
    ext_len = max(h, w)

    # ---------- 中文字体加载（跨平台尝试）----------
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",          # Windows 黑体
        "C:/Windows/Fonts/msyh.ttc",            # Windows 微软雅黑
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",  # Linux
        "/System/Library/Fonts/PingFang.ttc",   # macOS
        "simhei.ttf",                           # 当前目录下的字体文件
    ]
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, 18)  # 基础字号 18
                break
            except:
                continue
    if font is None:
        font = ImageFont.load_default()  # 最后保底

    def draw_chinese(img_bgr, text, pos, color_bgr, font_size=None):
        """在 OpenCV BGR 图像上绘制中文（带黑色描边）"""
        # 转换到 PIL RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 动态调整字体大小
        if font_size is None:
            used_font = font
        else:
            try:
                used_font = ImageFont.truetype(font.path, font_size) if hasattr(font, 'path') else font
            except:
                used_font = font

        # BGR -> RGB 颜色元组
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        # 黑色描边 (偏移 1px)
        draw.text((pos[0]-1, pos[1]-1), text, font=used_font, fill=(0,0,0))
        draw.text((pos[0]+1, pos[1]+1), text, font=used_font, fill=(0,0,0))
        draw.text(pos, text, font=used_font, fill=color_rgb)

        # 转回 OpenCV BGR
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ---------- 绘制水平和竖直线 ----------
    if det['h_line']:
        x1, y1, x2, y2 = det['h_line']
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 180, 0), 2, cv2.LINE_AA)
        out = draw_chinese(out, 'X', (int(x2)-30, int(y2)-10), (255, 180, 0), font_size=16)

    if det['v_line']:
        x1, y1, x2, y2 = det['v_line']
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2, cv2.LINE_AA)
        out = draw_chinese(out, 'Y', (int(x1)+10, int(y1)+25), (0, 200, 255), font_size=16)

    # ---------- 十字交点（黄色）----------
    if det['intersection']:
        cx, cy = int(det['intersection'][0]), int(det['intersection'][1])
        cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        out = draw_chinese(out, '十字交点', (cx+12, cy-12), (0, 255, 255), font_size=16)

    # ---------- CMOS 图像中心（绿色）—— 十字放大到 40 ----------
    ccx, ccy = int(center[0]), int(center[1])
    cv2.drawMarker(out, (ccx, ccy), (0, 220, 0), cv2.MARKER_CROSS, 40, 2)   # 改为 40
    cv2.circle(out, (ccx, ccy), 4, (0, 220, 0), -1, cv2.LINE_AA)
    out = draw_chinese(out, 'CMOS中心', (ccx+12, ccy+12), (0, 220, 0), font_size=16)

    # ---------- 中点到交点连线（品红虚线）----------
    if det['intersection'] and det['distance'] and det['distance'] > 0:
        cx, cy = int(det['intersection'][0]), int(det['intersection'][1])
        cv2.line(out, (cx, cy), (ccx, ccy), (255, 0, 255), 1, cv2.LINE_AA)

    # ---------- 数值标注（仍用英文数字，避免中英文混排麻烦，但全部改为支持中文）----------
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
        self.setWindowTitle('CMOS 位置对准软件')
        self.resize(1280, 860)

        self._image = None        # 当前处理的 BGR 图像
        self._result = None       # 最新检测结果 dict
        self._result_image = None # 标注后的图像

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
        lay_params.addWidget(QLabel('像素/mm:'), 4, 0)
        self.spin_ppm = QDoubleSpinBox()
        self.spin_ppm.setRange(0, 1000)
        self.spin_ppm.setDecimals(2)
        self.spin_ppm.setSingleStep(0.1)
        self.spin_ppm.setToolTip('标定值，0=仅像素距离')
        lay_params.addWidget(self.spin_ppm, 4, 1)
        self.btn_redetect = QPushButton('重新检测')
        self.btn_redetect.clicked.connect(self._detect)
        lay_params.addWidget(self.btn_redetect, 5, 0, 1, 2)
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
        self.lbl_status = QLabel('状态: 就绪')
        for lb in (self.lbl_angle, self.lbl_dist, self.lbl_dx, self.lbl_dy,
                   self.lbl_cross_angle, self.lbl_lines, self.lbl_status):
            lb.setWordWrap(True)
            lay_res.addWidget(lb)
        panel.addWidget(grp_res)

        # -- 图例 --
        grp_legend = QGroupBox('图例')
        lay_legend = QVBoxLayout(grp_legend)
        for txt in ('橙色 — 十字线 X 轴（水平）',
                    '青色 — 十字线 Y 轴（竖直）',
                    '黄色十字 — 十字交点',
                    '绿色十字 — CMOS 图像中心',
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

    # ============================================================
    #  核心流程
    # ============================================================

    def _log_msg(self, msg):
        self._log.append(msg)

    def _detect(self):
        """对当前图像运行检测并更新显示。"""
        if self._image is None:
            return
        gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        det = detect_crosshair(
            gray,
            canny_low=self.spin_canny_lo.value(),
            canny_high=self.spin_canny_hi.value(),
            hough_threshold=self.spin_hough.value(),
            angle_tol=float(self.spin_angle.value()),
        )
        self._result = det
        self._update_ui()
        self._log_msg(det['message'])

    def _update_ui(self):
        """更新结果标签和图像显示。"""
        det = self._result
        if det is None:
            return

        ppm = self.spin_ppm.value()

        # 角度
        if det['h_angle'] is not None:
            self.lbl_angle.setText(f'旋转误差 (Δθ): {det["h_angle"]:+.3f}°')
        else:
            self.lbl_angle.setText('旋转误差 (Δθ): --')

        # 距离
        if det['distance'] is not None:
            txt = f'中心偏移: {det["distance"]:.2f} px'
            if ppm > 0:
                txt += f'  ({det["distance"] / ppm:.3f} mm)'
            self.lbl_dist.setText(txt)
        else:
            self.lbl_dist.setText('中心偏移: --')

        # dX, dY
        if det['dx'] is not None:
            self.lbl_dx.setText(f'dX: {det["dx"]:+.2f} px')
            self.lbl_dy.setText(f'dY: {det["dy"]:+.2f} px')
        else:
            self.lbl_dx.setText('dX: --')
            self.lbl_dy.setText('dY: --')

        # X-Y 夹角
        if det['cross_angle'] is not None:
            self.lbl_cross_angle.setText(f'X-Y 夹角: {det["cross_angle"]:.2f}°')
        else:
            self.lbl_cross_angle.setText('X-Y 夹角: --')

        # 线段统计
        self.lbl_lines.setText(f'检测线段: {det["h_count"]} 水平, {det["v_count"]} 竖直')

        # 状态
        self.lbl_status.setText(f'状态: {det["message"]}')
        self.statusBar().showMessage(det['message'], 3000)

        # 绘制标注图像
        if self._image is not None:
            annotated = draw_result(self._image, det, ppm)
            self._result_image = annotated
            self._show_image(annotated)

    def _show_image(self, bgr: np.ndarray):
        """在 QLabel 上显示 BGR 图像。"""
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

    # ============================================================
    #  文件 / 摄像头
    # ============================================================

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
        """冻结当前摄像头画面（停止实时检测）。"""
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
