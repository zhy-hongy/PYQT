# -*- coding: utf-8 -*-
"""对焦测试板四点检测与模糊量估计。"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Tuple

import cv2
import numpy as np

POINT_LABELS = ("P1-左上", "P2-右上", "P3-左下", "P4-右下")

# Camera Test Chart 四角对焦点在图卡边框内的归一化坐标
# P1 左上色轮 | P2 右上 Siemens 星 | P3 左下 Siemens 星 | P4 右下色轮
CHART_POINT_ANCHORS: Tuple[Tuple[float, float], ...] = (
    (0.130, 0.172),
    (0.834, 0.172),
    (0.130, 0.785),
    (0.834, 0.785),
)
CHART_ASPECT_RATIO = 794.0 / 633.0  # 标准图卡宽高比

BOARD_MODE_CHART = "camera_test_chart"
BOARD_MODE_AUTO = "auto_circle"


@dataclass
class ChartFrame:
    """图卡坐标系（边框矩形）。"""

    x: float
    y: float
    w: float
    h: float
    detected: bool
    partial: bool = False


@dataclass
class FocusPointResult:
    """单个测点的对焦结果。"""

    index: int
    label: str
    center: Tuple[float, float]
    radius: float
    blur_px: float
    sharpness: float
    focused: bool
    detected: bool = True
    visible: bool = True


@dataclass
class FocusFrameResult:
    """整帧对焦检测结果。"""

    points: List[FocusPointResult] = field(default_factory=list)
    message: str = ""


def _cluster_line_pos(values: List[Tuple[float, float]], tol: float) -> List[Tuple[float, float]]:
    """按位置聚类线段，返回 (代表坐标, 最大长度)。"""
    if not values:
        return []
    values = sorted(values, key=lambda t: t[0])
    clusters: List[List[Tuple[float, float]]] = [[values[0]]]
    for pos, length in values[1:]:
        if abs(pos - clusters[-1][0][0]) <= tol:
            clusters[-1].append((pos, length))
        else:
            clusters.append([(pos, length)])
    return [(float(np.mean([p for p, _ in c])), max(l for _, l in c)) for c in clusters]


def _find_border_lines(
    gray: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """检测图卡外框四条边的位置，返回 top, bottom, left, right。"""
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    min_len = int(min(w, h) * 0.18)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 60,
        minLineLength=max(min_len, int(min(w, h) * 0.22)),
        maxLineGap=25,
    )

    horiz: List[Tuple[float, float]] = []
    vert: List[Tuple[float, float]] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_len:
                continue
            if abs(y2 - y1) < abs(x2 - x1) * 0.12:
                horiz.append(((y1 + y2) / 2, length))
            elif abs(x2 - x1) < abs(y2 - y1) * 0.12:
                vert.append(((x1 + x2) / 2, length))

    h_clusters = _cluster_line_pos(horiz, tol=max(8, h * 0.015))
    v_clusters = _cluster_line_pos(vert, tol=max(8, w * 0.015))

    min_horiz_len = w * 0.45
    min_vert_len = h * 0.45

    top_cands = [(y, ln) for y, ln in h_clusters if ln >= min_horiz_len]
    bot_cands = [(y, ln) for y, ln in h_clusters if ln >= min_horiz_len]
    left_cands = [(x, ln) for x, ln in v_clusters if ln >= min_vert_len]
    right_cands = [(x, ln) for x, ln in v_clusters if ln >= min_vert_len]

    top = min((y for y, _ in top_cands), default=None)
    bottom = max((y for y, _ in bot_cands), default=None)
    left = min((x for x, _ in left_cands), default=None)
    right = max((x for x, _ in right_cands), default=None)

    if top_cands and bot_cands and left_cands and right_cands:
        if right - left < w * 0.12 or bottom - top < h * 0.12:
            return None, None, None, None
        return top, bottom, left, right

    return top, bottom, left, right


def _detect_chart_frame(gray: np.ndarray) -> ChartFrame:
    """
    检测 Camera Test Chart 外框；支持部分入镜（仅可见部分边框时外推图卡坐标系）。
    """
    h, w = gray.shape
    top, bottom, left, right = _find_border_lines(gray)

    has = {
        "top": top is not None,
        "bottom": bottom is not None,
        "left": left is not None,
        "right": right is not None,
    }

    if all(has.values()):
        return ChartFrame(float(left), float(top), float(right - left), float(bottom - top), True)

    if has["top"] and has["left"]:
        x0, y0 = float(left), float(top)
        if has["right"]:
            fw = float(right - left)
            fh = fw / CHART_ASPECT_RATIO
        elif has["bottom"]:
            fh = float(bottom - top)
            fw = fh * CHART_ASPECT_RATIO
        else:
            fw = float(w - x0)
            fh = float(h - y0)
        return ChartFrame(x0, y0, fw, fh, True, partial=True)

    if has["top"] and has["right"]:
        x0 = float(right) - (float(right - left) if has["left"] else (h * CHART_ASPECT_RATIO))
        y0 = float(top)
        fw = float(right - x0) if has["left"] else float(w)
        fh = fw / CHART_ASPECT_RATIO
        return ChartFrame(x0, y0, fw, fh, True, partial=True)

    if has["bottom"] and has["left"]:
        x0, y0 = float(left), float(bottom) - (float(bottom - top) if has["top"] else w / CHART_ASPECT_RATIO)
        fh = float(bottom - y0)
        fw = fh * CHART_ASPECT_RATIO
        return ChartFrame(x0, y0, fw, fh, True, partial=True)

    if has["bottom"] and has["right"]:
        fw = float(right - left) if has["left"] else float(w)
        fh = float(bottom - top) if has["top"] else fw / CHART_ASPECT_RATIO
        x0 = float(right) - fw
        y0 = float(bottom) - fh
        return ChartFrame(x0, y0, fw, fh, True, partial=True)

    if has["top"] and has["bottom"]:
        y0, fh = float(top), float(bottom - top)
        fw = fh * CHART_ASPECT_RATIO
        x0 = float(left) if has["left"] else (w - fw) / 2
        return ChartFrame(x0, y0, fw, fh, True, partial=True)

    if has["left"] and has["right"]:
        x0, fw = float(left), float(right - left)
        fh = fw / CHART_ASPECT_RATIO
        y0 = float(top) if has["top"] else (h - fh) / 2
        return ChartFrame(x0, y0, fw, fh, True, partial=True)

    return ChartFrame(0.0, 0.0, float(w), float(h), False, partial=True)


def _markers_corner_search(
    gray: np.ndarray,
    roi_scale: float = 2.5,
) -> List[Tuple[float, float, float, bool]]:
    """
    局部入镜时：在画面四角区域搜索高清晰度特征，仅启用有内容的测点。
    """
    h, w = gray.shape
    scale = max(h, w) / 720.0
    radius = max(18.0, 55.0 * scale)
    search_r = int(min(w, h) * 0.22)

    corner_seeds = [
        (search_r, search_r),
        (w - search_r, search_r),
        (search_r, h - search_r),
        (w - search_r, h - search_r),
    ]

    raw: List[Tuple[float, float, float, float, bool]] = []
    border_margin = max(12, int(min(w, h) * 0.04))
    for sx, sy in corner_seeds:
        cx, cy = _refine_point_by_sharpness(gray, float(sx), float(sy), search_r=search_r)
        if (
            cx < border_margin or cy < border_margin
            or cx >= w - border_margin or cy >= h - border_margin
        ):
            raw.append((cx, cy, radius, 0.0, False))
            continue
        r = int(max(10, radius * 0.7))
        x1 = max(0, int(cx) - r)
        y1 = max(0, int(cy) - r)
        x2 = min(w, int(cx) + r)
        y2 = min(h, int(cy) + r)
        patch = gray[y1:y2, x1:x2]
        sharpness = float(cv2.Laplacian(patch, cv2.CV_64F).var()) if patch.size else 0.0
        has = _point_has_content(gray, cx, cy, radius)
        raw.append((cx, cy, radius, sharpness, has))

    valid_sharp = [s for _, _, _, s, h_ok in raw if h_ok]
    max_sharp = max(valid_sharp, default=0.0)
    rel_threshold = max(200.0, max_sharp * 0.48)

    markers: List[Tuple[float, float, float, bool]] = []
    roi_r = max(12.0, radius * roi_scale)
    for cx, cy, rad, sharpness, has in raw:
        visible = (
            has
            and sharpness >= rel_threshold
            and _roi_analyzable(gray, cx, cy, roi_r)
        )
        markers.append((cx, cy, rad, visible))
    return markers


def _chart_frame_usable(chart: ChartFrame, img_w: int, img_h: int) -> bool:
    """图卡边框是否足够可靠，可用于归一化坐标定位。"""
    if not chart.detected or chart.partial:
        return False
    area_ratio = (chart.w * chart.h) / float(img_w * img_h)
    if area_ratio < 0.32:
        return False
    aspect = chart.w / max(chart.h, 1)
    if aspect < 1.05 or aspect > 1.45:
        return False
    return True


def _point_has_content(gray: np.ndarray, cx: float, cy: float, radius: float) -> bool:
    """判断该位置是否有图卡纹理（避免局部入镜时在空白区域误检）。"""
    h, w = gray.shape
    r = int(max(10, radius * 0.7))
    x1 = max(0, int(cx) - r)
    y1 = max(0, int(cy) - r)
    x2 = min(w, int(cx) + r)
    y2 = min(h, int(cy) + r)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 6:
        return False
    contrast = float(patch.max() - patch.min())
    sharpness = float(cv2.Laplacian(patch, cv2.CV_64F).var())
    return contrast >= 40 and sharpness >= 120


def _roi_analyzable(gray: np.ndarray, cx: float, cy: float, roi_r: float, min_size: int = 24) -> bool:
    """ROI 允许贴边裁剪，只要剩余区域足够分析即可。"""
    h, w = gray.shape
    x1 = max(0, int(cx) - int(roi_r))
    y1 = max(0, int(cy) - int(roi_r))
    x2 = min(w, int(cx) + int(roi_r))
    y2 = min(h, int(cy) + int(roi_r))
    return (x2 - x1) >= min_size and (y2 - y1) >= min_size


def _chart_markers(
    gray: np.ndarray,
    chart: ChartFrame,
    roi_scale: float = 2.5,
) -> List[Tuple[float, float, float, bool]]:
    """根据图卡坐标系计算四点，返回 (cx, cy, radius, visible)。"""
    scale = max(chart.w, chart.h) / 760.0
    radius = max(18.0, 55.0 * scale)
    roi_r = max(12.0, radius * roi_scale)

    markers: List[Tuple[float, float, float, bool]] = []
    for nx, ny in CHART_POINT_ANCHORS:
        cx = chart.x + chart.w * nx
        cy = chart.y + chart.h * ny
        in_bounds = _roi_analyzable(gray, cx, cy, roi_r)
        has_content = _point_has_content(gray, cx, cy, radius) if in_bounds else False
        visible = in_bounds and has_content
        markers.append((cx, cy, radius, visible))
    return markers


def _refine_point_by_sharpness(
    gray: np.ndarray,
    cx: float,
    cy: float,
    search_r: int = 35,
) -> Tuple[float, float]:
    """在预期位置附近搜索清晰度最高的点，微调对焦点。"""
    h, w = gray.shape
    r = search_r
    x1 = max(0, int(cx) - r)
    y1 = max(0, int(cy) - r)
    x2 = min(w, int(cx) + r)
    y2 = min(h, int(cy) + r)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 8 or patch.shape[1] < 8:
        return cx, cy

    win = 15
    if patch.shape[0] <= win or patch.shape[1] <= win:
        return cx, cy

    lap = cv2.Laplacian(patch, cv2.CV_64F)
    scores = cv2.boxFilter(lap ** 2, -1, (win, win))
    _, max_val, _, max_loc = cv2.minMaxLoc(scores)
    if max_val < 1:
        return cx, cy

    rx = x1 + max_loc[0] + win // 2
    ry = y1 + max_loc[1] + win // 2
    return float(rx), float(ry)


def _sort_four_points(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """按 左上、右上、左下、右下 排序。"""
    pts = sorted(pts, key=lambda p: (p[1], p[0]))
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bottom[0], bottom[1]]


def _select_four_corners(
    candidates: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """从多个候选圆中选出分布于四角的四个。"""
    if len(candidates) <= 4:
        return sorted(candidates, key=lambda c: (c[1], c[0]))

    best_combo: Optional[tuple] = None
    best_area = -1.0
    for combo in combinations(candidates, 4):
        pts = np.array([(c[0], c[1]) for c in combo], dtype=np.float32)
        hull = cv2.convexHull(pts)
        area = float(cv2.contourArea(hull))
        if area > best_area:
            best_area = area
            best_combo = combo

    return list(best_combo) if best_combo else candidates[:4]


def _detect_markers(
    gray: np.ndarray,
    min_area: int = 80,
    max_area: int = 50000,
) -> List[Tuple[float, float, float]]:
    """
    检测对焦板上的圆形标记，返回 (cx, cy, radius) 列表。
    支持黑圆白底或白圆黑底。
    """
    h, w = gray.shape
    scale = max(h, w) / 720.0
    min_r = max(3, int(4 * scale))
    max_r = max(min_r + 2, int(min(h, w) * 0.15))

    found: List[Tuple[float, float, float]] = []

    for img in (gray, 255 - gray):
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min(h, w) * 0.12),
            param1=100,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            for c in circles[0]:
                x, y, r = float(c[0]), float(c[1]), float(c[2])
                area = np.pi * r * r
                if min_area <= area <= max_area:
                    found.append((x, y, r))

    if len(found) < 4:
        found.extend(_detect_blobs(gray, min_area, max_area))

    # 去重：合并距离很近的重复圆
    merged: List[Tuple[float, float, float]] = []
    for x, y, r in sorted(found, key=lambda c: c[2], reverse=True):
        if all(np.hypot(x - mx, y - my) > max(r, mr) * 0.6 for mx, my, mr in merged):
            merged.append((x, y, r))

    if len(merged) >= 4:
        return _select_four_corners(merged[:12])

    return merged


def _detect_blobs(
    gray: np.ndarray,
    min_area: int,
    max_area: int,
) -> List[Tuple[float, float, float]]:
    """SimpleBlobDetector 备选方案。"""
    results: List[Tuple[float, float, float]] = []
    for invert in (0, 1):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = float(min_area)
        params.maxArea = float(max_area)
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.filterByInertia = True
        params.minInertiaRatio = 0.3
        if invert:
            params.minThreshold = 10
            params.maxThreshold = 200
        detector = cv2.SimpleBlobDetector_create(params)
        src = (255 - gray) if invert else gray
        kps = detector.detect(src)
        for kp in kps:
            results.append((kp.pt[0], kp.pt[1], kp.size / 2))
    return results


def _sample_line(
    gray: np.ndarray,
    cx: float,
    cy: float,
    angle: float,
    half_len: int = 15,
) -> Optional[np.ndarray]:
    """沿指定角度采样灰度剖面。"""
    h, w = gray.shape
    values = []
    for t in range(-half_len, half_len + 1):
        x = int(round(cx + t * np.cos(angle)))
        y = int(round(cy + t * np.sin(angle)))
        if 0 <= x < w and 0 <= y < h:
            values.append(float(gray[y, x]))
    if len(values) < half_len:
        return None
    return np.array(values, dtype=np.float64)


def _transition_width(profile: np.ndarray) -> float:
    """从灰度剖面估计边缘过渡宽度（像素）。"""
    p = profile - profile.min()
    if p.max() < 8:
        return 0.0
    p = p / p.max()

    low_idx = np.where(p <= 0.2)[0]
    high_idx = np.where(p >= 0.8)[0]
    if len(low_idx) == 0 or len(high_idx) == 0:
        grad = np.abs(np.diff(p))
        if grad.max() < 0.05:
            return 0.0
        peak = int(np.argmax(grad))
        left = peak
        while left > 0 and p[left] > 0.2:
            left -= 1
        right = peak
        while right < len(p) - 1 and p[right] < 0.8:
            right += 1
        return float(max(1, right - left))

    left = int(low_idx[-1])
    right = int(high_idx[0])
    if right <= left:
        grad = np.abs(np.diff(p))
        peak = int(np.argmax(grad))
        return float(max(1, min(len(p) - 1, peak + 4) - max(0, peak - 4)))

    return float(max(1, right - left))


def estimate_blur_pixels(gray_roi: np.ndarray) -> Tuple[float, float]:
    """
    估计 ROI 模糊量（像素）及清晰度分数。

    返回 (blur_px, sharpness)，blur_px 越大表示越模糊。
    """
    if gray_roi.size == 0:
        return 99.0, 0.0

    if gray_roi.ndim == 3:
        gray = cv2.cvtColor(gray_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_roi

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    h, w = gray.shape
    cx, cy = w / 2, h / 2
    max_r = max(4, min(h, w) // 2 - 2)

    # 从中心沿径向采样，适合圆形标记的边缘
    radial_widths: List[float] = []
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        profile = _sample_line(gray, cx, cy, angle, half_len=max_r)
        if profile is not None and profile.max() - profile.min() > 12:
            radial_widths.append(_transition_width(profile))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    grad_widths: List[float] = []
    if mag.max() >= 1.0:
        thresh = mag.max() * 0.45
        ys, xs = np.where(mag >= thresh)
        order = np.argsort(mag[ys, xs])[::-1]
        step = max(1, len(order) // 20)
        for idx in order[::step][:20]:
            x, y = int(xs[idx]), int(ys[idx])
            angle = float(np.arctan2(gy[y, x], gx[y, x]) + np.pi / 2)
            profile = _sample_line(gray, x, y, angle, half_len=min(20, max_r))
            if profile is not None and profile.max() - profile.min() > 8:
                grad_widths.append(_transition_width(profile))

    edge_widths = radial_widths + grad_widths
    if edge_widths:
        edge_blur = max(0.0, float(np.median(edge_widths)) - 1.5)
    else:
        edge_blur = None

    # 以 Laplacian 为主映射模糊像素（与失焦程度相关性更好）
    if sharpness >= 800:
        sharp_blur = 0.0
    elif sharpness >= 400:
        sharp_blur = (800 - sharpness) / 200.0
    elif sharpness >= 100:
        sharp_blur = 2.0 + (400 - sharpness) / 100.0
    elif sharpness >= 30:
        sharp_blur = 5.0 + (100 - sharpness) / 20.0
    else:
        sharp_blur = 8.5 + (30 - min(sharpness, 30)) / 10.0

    if edge_blur is not None and 100 <= sharpness <= 600:
        blur_px = 0.35 * edge_blur + 0.65 * sharp_blur
    else:
        blur_px = sharp_blur

    return float(min(max(blur_px, 0.0), 99.0)), sharpness


def _fallback_quadrant_centers(w: int, h: int) -> List[Tuple[float, float, float]]:
    """未检测到标记时，按画面四象限取默认测点。"""
    margin_x, margin_y = w * 0.25, h * 0.25
    r = min(w, h) * 0.06
    return [
        (margin_x, margin_y, r),
        (w - margin_x, margin_y, r),
        (margin_x, h - margin_y, r),
        (w - margin_x, h - margin_y, r),
    ]


def process_frame(
    frame: np.ndarray,
    blur_threshold_px: float = 2.0,
    roi_scale: float = 2.5,
    min_marker_area: int = 80,
    board_mode: str = BOARD_MODE_CHART,
) -> Tuple[np.ndarray, FocusFrameResult]:
    """检测四个测点并评估对焦状态。"""
    result = FocusFrameResult()
    output = frame.copy()

    if frame is None or frame.size == 0:
        result.message = "无效图像"
        return output, result

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    chart_rect: Optional[ChartFrame] = None
    marker_specs: List[Tuple[float, float, float, bool]] = []

    if board_mode == BOARD_MODE_CHART:
        chart_rect = _detect_chart_frame(gray)
        if _chart_frame_usable(chart_rect, w, h):
            marker_specs = _chart_markers(gray, chart_rect, roi_scale=roi_scale)
            x, y, cw, ch = int(chart_rect.x), int(chart_rect.y), int(chart_rect.w), int(chart_rect.h)
            cv2.rectangle(output, (x, y), (x + cw, y + ch), (0, 255, 255), 2, cv2.LINE_AA)
        else:
            marker_specs = _markers_corner_search(gray, roi_scale=roi_scale)
            if chart_rect.partial:
                result.message = "部分入镜：四角搜索模式"
            else:
                result.message = "局部/未完整入镜：四角搜索模式"
            if chart_rect.detected:
                x, y, cw, ch = int(chart_rect.x), int(chart_rect.y), int(chart_rect.w), int(chart_rect.h)
                cv2.rectangle(output, (x, y), (x + cw, y + ch), (255, 160, 0), 1, cv2.LINE_AA)
    else:
        markers = _detect_markers(gray, min_area=min_marker_area)
        use_fallback = len(markers) < 4
        if use_fallback:
            markers = _fallback_quadrant_centers(w, h)
            result.message = f"仅检测到 {len(_detect_markers(gray))} 个圆，已使用四象限默认位置"

        centers = [(m[0], m[1]) for m in markers[:4]]
        if len(centers) == 4:
            sorted_centers = _sort_four_points(centers)
            sorted_markers = []
            for sc in sorted_centers:
                best = min(markers[:4], key=lambda m: np.hypot(m[0] - sc[0], m[1] - sc[1]))
                sorted_markers.append(best)
            markers = sorted_markers

        for cx, cy, radius in markers[:4]:
            marker_specs.append((cx, cy, radius, True))

    visible_count = 0
    evaluated_count = 0

    for i, spec in enumerate(marker_specs[:4]):
        cx, cy, radius, visible = spec
        if not visible:
            result.points.append(FocusPointResult(
                index=i, label=POINT_LABELS[i], center=(cx, cy), radius=radius,
                blur_px=0.0, sharpness=0.0, focused=False, detected=False, visible=False,
            ))
            cv2.putText(
                output, f"P{i + 1} off", (max(5, int(cx) - 30), max(20, int(cy))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA,
            )
            continue

        visible_count += 1

        if board_mode == BOARD_MODE_CHART and _chart_frame_usable(chart_rect, w, h):
            rx, ry = _refine_point_by_sharpness(gray, cx, cy)
            if np.hypot(rx - cx, ry - cy) <= 40:
                cx, cy = rx, ry

        roi_r = int(max(12, radius * roi_scale))
        x1 = max(0, int(cx) - roi_r)
        y1 = max(0, int(cy) - roi_r)
        x2 = min(w, int(cx) + roi_r)
        y2 = min(h, int(cy) + roi_r)
        roi = gray[y1:y2, x1:x2]

        blur_px, sharpness = estimate_blur_pixels(roi)
        focused = blur_px <= blur_threshold_px and sharpness > 20
        evaluated_count += 1

        pt_result = FocusPointResult(
            index=i,
            label=POINT_LABELS[i],
            center=(cx, cy),
            radius=radius,
            blur_px=blur_px,
            sharpness=sharpness,
            focused=focused,
            detected=bool(chart_rect and chart_rect.detected and _chart_frame_usable(chart_rect, w, h)) if board_mode == BOARD_MODE_CHART else True,
            visible=True,
        )
        result.points.append(pt_result)

        color = (0, 220, 0) if focused else (0, 0, 255)
        cv2.circle(output, (int(cx), int(cy)), int(radius), color, 2, cv2.LINE_AA)
        cv2.circle(output, (int(cx), int(cy)), 4, color, -1, cv2.LINE_AA)

        tag = f"P{i + 1}"
        cv2.putText(
            output, tag, (int(cx) - 12, int(cy) - int(radius) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            output, f"{blur_px:.1f}px", (int(cx) - 20, int(cy) + int(radius) + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)

    ok_count = sum(1 for p in result.points if p.visible and p.focused)
    vis_eval = sum(1 for p in result.points if p.visible)
    if vis_eval > 0:
        status = f"{ok_count}/{vis_eval} OK"
        if visible_count < 4:
            status += f" ({visible_count}/4 in view)"
    else:
        status = "0/0 无测点在视野内"

    if not result.message:
        result.message = status
    elif status not in result.message:
        result.message = f"{status} | {result.message}"

    cv2.putText(
        output, status, (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (0, 255, 0) if vis_eval > 0 and ok_count == vis_eval else (0, 200, 255),
        2,
    )

    return output, result
