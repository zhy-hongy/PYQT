"""十字线与激光线检测、角度与距离计算。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class Line2D:
    """线段，由两端点表示。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def angle_deg(self) -> float:
        """相对水平方向的角度，范围 [-90, 90]。"""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return float(np.degrees(np.arctan2(dy, dx)))

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    def extend(self, length: float) -> "Line2D":
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        norm = np.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        half = length / 2
        return Line2D(cx - ux * half, cy - uy * half, cx + ux * half, cy + uy * half)


@dataclass
class AlignmentResult:
    """对准测量结果。"""

    h_line: Optional[Line2D] = None
    v_line: Optional[Line2D] = None
    cross_point: Optional[Tuple[float, float]] = None
    laser_line: Optional[Line2D] = None
    laser_center: Optional[Tuple[float, float]] = None
    angle_to_vertical_deg: Optional[float] = None
    distance_px: Optional[float] = None
    message: str = ""


def _normalize_angle(angle: float) -> float:
    """归一化到 [-90, 90]。"""
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return angle


def _line_from_segment(x1: float, y1: float, x2: float, y2: float) -> Line2D:
    return Line2D(float(x1), float(y1), float(x2), float(y2))


def _cluster_lines(
    segments: list[Line2D],
    angle_threshold: float = 15.0,
) -> Tuple[Optional[Line2D], Optional[Line2D]]:
    """将线段聚类为水平线与竖直线。"""
    if not segments:
        return None, None

    horizontals: list[Line2D] = []
    verticals: list[Line2D] = []

    for seg in segments:
        angle = abs(_normalize_angle(seg.angle_deg))
        if angle <= angle_threshold:
            horizontals.append(seg)
        elif angle >= 90 - angle_threshold:
            verticals.append(seg)

    def _merge(group: list[Line2D]) -> Optional[Line2D]:
        if not group:
            return None
        best = max(group, key=lambda s: s.length)
        return best.extend(best.length * 1.5)

    return _merge(horizontals), _merge(verticals)


def _detect_board_lines(gray: np.ndarray) -> Tuple[Optional[Line2D], Optional[Line2D]]:
    """检测板上的十字线（水平 + 竖直）。"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min(gray.shape) // 6,
        maxLineGap=20,
    )

    segments: list[Line2D] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segments.append(_line_from_segment(x1, y1, x2, y2))

    return _cluster_lines(segments)


def _intersect_lines(a: Line2D, b: Line2D) -> Optional[Tuple[float, float]]:
    """两线段所在直线的交点。"""
    x1, y1, x2, y2 = a.x1, a.y1, a.x2, a.y2
    x3, y3, x4, y4 = b.x1, b.y1, b.x2, b.y2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return float(px), float(py)


def _laser_mask(
    bgr: np.ndarray,
    laser_color: str = "violet",
    sensitivity: int = 30,
) -> np.ndarray:
    """根据颜色提取激光区域掩膜。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = max(10, sensitivity)

    if laser_color in ("violet", "405nm", "405"):
        # 405nm 紫/蓝激光：HSV 紫蓝区间 + 蓝色通道占优（部分相机偏蓝）
        h_pad = s // 2
        lower_hsv = np.array([max(100, 125 - h_pad), max(30, 80 - s), max(40, 80 - s)])
        upper_hsv = np.array([min(165, 145 + h_pad), 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        b, g, r = cv2.split(bgr)
        margin = max(15, 40 - s // 2)
        min_bright = max(80, 120 - s)
        mask_blue = ((b.astype(np.int16) > g + margin) &
                     (b.astype(np.int16) > r + margin) &
                     (b > min_bright)).astype(np.uint8) * 255

        mask = mask_hsv | mask_blue
    elif laser_color == "green":
        lower = np.array([35, 80, 80])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:
        lower1 = np.array([0, 100, 100])
        upper1 = np.array([10 + s // 3, 255, 255])
        lower2 = np.array([160 - s // 3, 100, 100])
        upper2 = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _detect_laser_line(
    bgr: np.ndarray,
    laser_color: str = "violet",
    sensitivity: int = 30,
) -> Tuple[Optional[Line2D], Optional[Tuple[float, float]]]:
    """检测激光线及其中心点。"""
    mask = _laser_mask(bgr, laser_color, sensitivity)
    points = cv2.findNonZero(mask)

    if points is None or len(points) < 20:
        return None, None

    pts = points.reshape(-1, 2).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx = float(vx.flat[0])
    vy = float(vy.flat[0])
    x0 = float(x0.flat[0])
    y0 = float(y0.flat[0])

    projections = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
    t_min, t_max = projections.min(), projections.max()

    x1 = x0 + vx * t_min
    y1 = y0 + vy * t_min
    x2 = x0 + vx * t_max
    y2 = y0 + vy * t_max

    line = _line_from_segment(x1, y1, x2, y2)
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    return line, center


def _angle_between_lines(laser: Line2D, vertical: Line2D) -> float:
    """激光线与竖直线的夹角（0~90°）。"""
    la = _normalize_angle(laser.angle_deg)
    va = _normalize_angle(vertical.angle_deg)
    diff = abs(la - va)
    if diff > 90:
        diff = 180 - diff
    return float(diff)


def process_frame(
    frame: np.ndarray,
    laser_color: str = "violet",
    sensitivity: int = 30,
    pixels_per_mm: float = 0.0,
) -> Tuple[np.ndarray, AlignmentResult]:
    """
    处理单帧图像，返回标注后的图像与测量结果。

    pixels_per_mm > 0 时可在 UI 中换算物理距离。
    """
    result = AlignmentResult()
    output = frame.copy()

    if frame is None or frame.size == 0:
        result.message = "无效图像"
        return output, result

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_line, v_line = _detect_board_lines(gray)
    result.h_line = h_line
    result.v_line = v_line

    if h_line and v_line:
        result.cross_point = _intersect_lines(h_line, v_line)
    elif h_line or v_line:
        result.message = "仅检测到单轴十字线"
    else:
        result.message = "未检测到十字线"

    laser_line, laser_center = _detect_laser_line(frame, laser_color, sensitivity)
    result.laser_line = laser_line
    result.laser_center = laser_center

    if not laser_line:
        if not result.message:
            result.message = "未检测到激光线"
        else:
            result.message += "；未检测到激光线"

    if laser_line and v_line:
        result.angle_to_vertical_deg = _angle_between_lines(laser_line, v_line)

    if laser_center and result.cross_point:
        cx, cy = result.cross_point
        lx, ly = laser_center
        result.distance_px = float(np.hypot(lx - cx, ly - cy))

    # --- 绘制 ---
    h, w = output.shape[:2]
    extend_len = max(h, w)

    if h_line:
        ext = h_line.extend(extend_len)
        cv2.line(
            output,
            (int(ext.x1), int(ext.y1)),
            (int(ext.x2), int(ext.y2)),
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output, "X", (int(ext.x2) - 30, int(ext.y2) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2,
        )

    if v_line:
        ext = v_line.extend(extend_len)
        cv2.line(
            output,
            (int(ext.x1), int(ext.y1)),
            (int(ext.x2), int(ext.y2)),
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output, "Y", (int(ext.x1) + 10, int(ext.y1) + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
        )

    if result.cross_point:
        cx, cy = int(result.cross_point[0]), int(result.cross_point[1])
        cv2.drawMarker(output, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(
            output, "Cross", (cx + 12, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
        )

    if laser_line:
        cv2.line(
            output,
            (int(laser_line.x1), int(laser_line.y1)),
            (int(laser_line.x2), int(laser_line.y2)),
            (255, 0, 255),
            3,
            cv2.LINE_AA,
        )

    if laser_center:
        lx, ly = int(laser_center[0]), int(laser_center[1])
        cv2.circle(output, (lx, ly), 8, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            output, "Laser", (lx + 12, ly + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2,
        )

    if laser_center and result.cross_point:
        cv2.line(
            output,
            (int(result.cross_point[0]), int(result.cross_point[1])),
            (int(laser_center[0]), int(laser_center[1])),
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # 数值结果由 PyQt 右侧面板显示；图像上仅用 ASCII 标注，避免 cv2.putText 中文乱码
    if not result.message and result.angle_to_vertical_deg is not None:
        result.message = "检测完成"

    return output, result
