

"""
圆形直径对焦检测处理模块
基于圆形直径判断对焦状态
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
from PIL import Image, ImageDraw, ImageFont

import cv2
import numpy as np


@dataclass
class CirclePointResult:
    """单个圆检测结果"""
    index: int
    label: str
    center: Tuple[float, float]
    radius: float
    diameter: float
    detected: bool = True


@dataclass
class CircleFocusResult:
    """整帧圆形直径对焦检测结果"""
    circles: List[CirclePointResult] = field(default_factory=list)
    avg_diameter: Optional[float] = None
    std_diameter: Optional[float] = None
    status: str = "insufficient"
    message: str = ""
    status_color: Tuple[int, int, int] = (128, 128, 128)


# 检测模式常量
BOARD_MODE_CHART = "camera_test_chart"
BOARD_MODE_AUTO = "auto_circle"
BOARD_MODE_CIRCLE_DIAMETER = "circle_diameter"

CIRCLE_POINT_LABELS = ("P1-左上", "P2-右上", "P3-左下", "P4-右下")


def detect_multiple_circles(
    gray: np.ndarray,
    dp: float = 1.0,
    min_dist: float = 50.0,
    param1: int = 100,
    param2: int = 30,
    min_radius: int = 30,
    max_radius: int = 200
) -> List[Tuple[float, float, float]]:
    """
    使用 Hough 变换检测多个圆

    Args:
        gray: 灰度图像
        dp: 累加器分辨率与图像分辨率的反比
        min_dist: 检测到的圆心之间的最小距离
        param1: Canny 边缘检测的高阈值
        param2: 累加器阈值，值越小检测到的圆越多
        min_radius: 最小半径
        max_radius: 最大半径

    Returns:
        检测到的圆列表 [(cx, cy, radius), ...]
    """
    # 中值滤波去噪
    gray_blur = cv2.medianBlur(gray, 5)
    
    # 检测圆
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is None:
        return []
    
    # 转换为列表格式并按面积从大到小排序
    circles_list = [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0]]
    circles_list.sort(key=lambda c: c[2], reverse=True)
    
    return circles_list


def select_four_circles_by_quadrants(
    circles: List[Tuple[float, float, float]],
    img_width: int,
    img_height: int
) -> List[Tuple[float, float, float]]:
    """
    从检测到的圆中选择分布在四个象限的四个圆

    Args:
        circles: 检测到的圆列表
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        选择的四个圆，按左上、右上、左下、右下排序
    """
    if len(circles) <= 4:
        return circles
    
    mid_x = img_width / 2
    mid_y = img_height / 2
    
    # 按象限分组
    quadrants = {
        "top_left": [],
        "top_right": [],
        "bottom_left": [],
        "bottom_right": []
    }
    
    for circle in circles:
        cx, cy, r = circle
        if cx < mid_x and cy < mid_y:
            quadrants["top_left"].append(circle)
        elif cx >= mid_x and cy < mid_y:
            quadrants["top_right"].append(circle)
        elif cx < mid_x and cy >= mid_y:
            quadrants["bottom_left"].append(circle)
        else:
            quadrants["bottom_right"].append(circle)
    
    # 从每个象限选择最大的圆
    selected = []
    for quadrant in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        if quadrants[quadrant]:
            selected.append(max(quadrants[quadrant], key=lambda c: c[2]))
    
    # 如果某些象限没有，用其他象限的补充
    if len(selected) < 4:
        remaining = [c for c in circles if c not in selected]
        selected.extend(remaining[:4 - len(selected)])
    
    return selected[:4]


def sort_four_points(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    按左上、右上、左下、右下排序四个点

    Args:
        pts: 四个点的列表 [(x, y), ...]

    Returns:
        排序后的点列表
    """
    if len(pts) != 4:
        return pts
    
    # 先按 y 坐标排序，再按 x 坐标排序
    sorted_by_y = sorted(pts, key=lambda p: (p[1], p[0]))
    top_two = sorted(sorted_by_y[:2], key=lambda p: p[0])
    bottom_two = sorted(sorted_by_y[2:], key=lambda p: p[0])
    
    return [top_two[0], top_two[1], bottom_two[0], bottom_two[1]]


def process_circle_focus_frame(
    frame: np.ndarray,
    ideal_diameter: float = 200.0,
    tolerance: float = 5.0,
    std_threshold: float = 3.0,
    hough_dp: float = 1.0,
    hough_min_dist: float = 50.0,
    hough_param1: int = 100,
    hough_param2: int = 30,
    hough_min_radius: int = 30,
    hough_max_radius: int = 200
) -> Tuple[np.ndarray, CircleFocusResult]:
    """
    处理一帧图像，检测圆形并判断对焦状态

    Args:
        frame: 输入图像
        ideal_diameter: 理想直径（像素）
        tolerance: 直径容差（像素）
        std_threshold: 直径标准差阈值
        hough_dp: Hough 变换 dp 参数
        hough_min_dist: Hough 变换 minDist 参数
        hough_param1: Hough 变换 param1 参数
        hough_param2: Hough 变换 param2 参数
        hough_min_radius: Hough 变换最小半径
        hough_max_radius: Hough 变换最大半径

    Returns:
        (处理后的图像, 检测结果)
    """
    result = CircleFocusResult()
    output = frame.copy()
    
    if frame is None or frame.size == 0:
        result.message = "无效图像"
        return output, result
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 检测圆
    circles = detect_multiple_circles(
        gray,
        dp=hough_dp,
        min_dist=hough_min_dist,
        param1=hough_param1,
        param2=hough_param2,
        min_radius=hough_min_radius,
        max_radius=hough_max_radius
    )
    
    # 选择四个圆
    selected_circles = select_four_circles_by_quadrants(circles, w, h)
    
    # 排序四个圆
    if len(selected_circles) == 4:
        centers = [(c[0], c[1]) for c in selected_circles]
        sorted_centers = sort_four_points(centers)
        
        # 重新排序圆
        sorted_circles = []
        for sc in sorted_centers:
            best_circle = min(selected_circles, key=lambda c: np.hypot(c[0] - sc[0], c[1] - sc[1]))
            sorted_circles.append(best_circle)
        selected_circles = sorted_circles
    
    # 处理检测到的圆
    diameters = []
    for i, (cx, cy, r) in enumerate(selected_circles[:4]):
        diameter = 2 * r
        diameters.append(diameter)
        
        pt_result = CirclePointResult(
            index=i,
            label=CIRCLE_POINT_LABELS[i] if i < len(CIRCLE_POINT_LABELS) else f"P{i + 1}",
            center=(cx, cy),
            radius=r,
            diameter=diameter,
            detected=True
        )
        result.circles.append(pt_result)
    
    # 判断对焦状态
    if len(diameters) < 4:
        result.status = "insufficient"
        result.message = f"未检测到足够的圆（仅 {len(diameters)}/4）"
        result.status_color = (0, 165, 255)
    else:
        avg_diameter = float(np.mean(diameters))
        std_diameter = float(np.std(diameters))
        result.avg_diameter = avg_diameter
        result.std_diameter = std_diameter
        
        lower = ideal_diameter - tolerance
        upper = ideal_diameter + tolerance
        
        if lower <= avg_diameter <= upper and std_diameter < std_threshold:
            result.status = "focus_ok"
            result.message = f"对焦完成 (平均直径: {avg_diameter:.1f}px, 标准差: {std_diameter:.2f}px)"
            result.status_color = (0, 255, 0)
        elif avg_diameter < lower:
            result.status = "too_near"
            result.message = f"离焦太近 (平均直径: {avg_diameter:.1f}px, 目标: {ideal_diameter}±{tolerance}px)"
            result.status_color = (0, 0, 255)
        else:
            result.status = "too_far"
            result.message = f"离焦太远 (平均直径: {avg_diameter:.1f}px, 目标: {ideal_diameter}±{tolerance}px)"
            result.status_color = (0, 0, 255)
    
    # 绘制结果
    output = draw_circle_focus_results(output, result)
    
    return output, result



def draw_circle_focus_results(image: np.ndarray, result: CircleFocusResult) -> np.ndarray:
    # 先绘制所有图形（圆、点等）仍用 OpenCV（速度快）
    output = image.copy()
    for circle_pt in result.circles:
        cx, cy = int(circle_pt.center[0]), int(circle_pt.center[1])
        r = int(circle_pt.radius)
        cv2.circle(output, (cx, cy), r, result.status_color, 2, cv2.LINE_AA)
        cv2.circle(output, (cx, cy), 4, result.status_color, -1, cv2.LINE_AA)

    # ---- 中文文本改用 Pillow 绘制 ----
    # 转换为 RGB PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 加载一个支持中文的字体文件（请根据系统实际路径修改）
    # Windows 示例：C:/Windows/Fonts/simhei.ttf
    # Linux 示例：/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf
    try:
        font = ImageFont.truetype("simhei.ttf", 20)   # 字号 20
    except IOError:
        # 如果找不到字体，回退到默认字体（可能仍不支持中文）
        font = ImageFont.load_default()

    for circle_pt in result.circles:
        cx, cy = int(circle_pt.center[0]), int(circle_pt.center[1])
        r = int(circle_pt.radius)

        # 绘制标签（中文）
        label = circle_pt.label          # 例如 "P1-左上"
        draw.text((cx - 30, cy - r - 25), label, font=font, fill=result.status_color)

        # 绘制直径数值（可继续用 cv2.putText，因为数值不含中文）
        cv2.putText(output, f"{circle_pt.diameter:.1f}px", (cx - 25, cy + r + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, result.status_color, 1, cv2.LINE_AA)

    # 状态栏文本（可能含中文）
    if result.avg_diameter is not None:
        status_text = f"平均直径: {result.avg_diameter:.1f}px | 标准差: {result.std_diameter:.2f}px | {result.message.split('(')[0]}"
    else:
        status_text = result.message
    draw.text((10, 10), status_text, font=font, fill=result.status_color)

    # 转回 OpenCV BGR 格式
    output = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return output
