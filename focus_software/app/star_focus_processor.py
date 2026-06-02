# -*- coding: utf-8 -*-
"""
四角同心圆对焦检测处理模块 - 工业沙姆角相机优化版
专门用于从白底黑圈的图纸中，过滤嵌套同心圆，只提取 4 个角落最外圈圆的直径
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

@dataclass
class MultiCirclePointResult:
    """单个圆检测结果"""
    index: int
    label: str
    center: Tuple[float, float]
    radius: float
    diameter: float
    detected: bool = True

@dataclass
class MultiCircleFocusResult:
    """整帧四角圆直径对焦检测结果"""
    circles: List[MultiCirclePointResult] = field(default_factory=list)
    avg_diameter: Optional[float] = None
    std_diameter: Optional[float] = None
    status: str = "insufficient"
    message: str = ""
    status_color: Tuple[int, int, int] = (128, 128, 128)

MULTI_CIRCLE_LABELS = ("P1-左上", "P2-右上", "P3-左下", "P4-右下")

def process_multi_circle_frame(
    frame: np.ndarray,
    ideal_diameter: float = 120.0,
    tolerance: float = 5.0,
    std_threshold: float = 3.0
) -> tuple[np.ndarray, MultiCircleFocusResult]:
    """
    通过轮廓拓扑树过滤内圈，仅提取4个角落最外层圆环的直径进行对焦评估（专为沙姆角畸变优化）
    """
    output = frame.copy()
    h, w = frame.shape[:2]
    result = MultiCircleFocusResult()
    
    # 1. 图像预处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 白底黑圈：使用 BINARY_INV 使得黑色的圆环在二值化后变成高亮的白色轮廓
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. 寻找轮廓并建立树状拓扑关系 (RETR_TREE)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_circles = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # 根据全图分辨率自适应面积筛选，防止过滤掉大图中的圆（下限100像素，上限占全图15%）
            if area < 100 or area > (w * h * 0.15):
                continue
                
            # 【核心过滤】：如果当前轮廓有父轮廓，说明它是同心圆的内圈线，直接跳过，只要最外圈！
            if hierarchy[i][3] != -1:
                continue
                
            # 拟合最小外接圆
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            diameter = radius * 2
            
            # 过滤长宽比：【重要修改】因为沙姆角相机的倾斜，正圆在画面里会变成椭圆。
            # 将原本严格的 0.85-1.15 放宽到 0.5-2.0，允许明显的椭圆通过，极大地提升识别率！
            x, y, r_w, r_h = cv2.boundingRect(contour)
            aspect_ratio = float(r_w) / r_h
            if not (0.5 <= aspect_ratio <= 2.0):
                continue
                
            # 排除处于画面正中央的元素（防止误抓中间的 Siemens 星图边缘）
            if (0.4 * w < cx < 0.6 * w) and (0.4 * h < cy < 0.6 * h):
                continue
                
            valid_circles.append(((cx, cy), radius, diameter))

    # 3. 象限划分定位 (确保准确分流到左上、右上、左下、右下)
    mid_x, mid_y = w / 2, h / 2
    quadrants = { "TL": [], "TR": [], "BL": [], "BR": [] }
    
    for item in valid_circles:
        cx, cy = item[0]
        if cx < mid_x and cy < mid_y: quadrants["TL"].append(item)
        elif cx >= mid_x and cy < mid_y: quadrants["TR"].append(item)
        elif cx < mid_x and cy >= mid_y: quadrants["BL"].append(item)
        else: quadrants["BR"].append(item)
        
    # 如果任意一个角落没找到圆，返回失败，并在图上把所有找到的嫌疑目标画出来方便调试
    if not (quadrants["TL"] and quadrants["TR"] and quadrants["BL"] and quadrants["BR"]):
        result.status = "insufficient"
        result.message = f"未找齐四角圆 (共识别到 {len(valid_circles)} 个外圈)"
        result.status_color = (0, 0, 255)
        
        # 调试辅助线：把所有捕获到的外圈用橙色标出来，哪怕数量不够
        for (center, radius, diameter) in valid_circles:
            cv2.circle(output, (int(center[0]), int(center[1])), int(radius), (0, 165, 255), 2)
            cv2.putText(output, f"{diameter:.1f}px", (int(center[0])-20, int(center[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                        
        output = _draw_chinese_text(output, result.message, (20, 40), (0, 0, 255))
        return output, result

    # 每个象限中，选择最靠近边缘（离画面中心最远）的那个轮廓作为最终标靶
    final_4_circles = []
    for key in ["TL", "TR", "BL", "BR"]:
        q_list = quadrants[key]
        best_circle = max(q_list, key=lambda c: (c[0][0] - mid_x)**2 + (c[0][1] - mid_y)**2)
        final_4_circles.append(best_circle)
    
    # 4. 数据统计与对焦状态分析
    diameters = [c[2] for c in final_4_circles]
    avg_d = float(np.mean(diameters))
    std_d = float(np.std(diameters))
    
    result.avg_diameter = avg_d
    result.std_diameter = std_d
    
    dev_from_ideal = abs(avg_d - ideal_diameter)
    size_ok = dev_from_ideal <= tolerance
    std_ok = std_d <= std_threshold
    
    if size_ok and std_ok:
        result.status = "focused"
        result.message = f"四角对焦成功! 平均直径: {avg_d:.1f}px, 标准差: {std_d:.2f}"
        result.status_color = (0, 220, 0) # 绿色
    else:
        result.status = "unfocused"
        reasons = []
        if not size_ok: reasons.append(f"直径偏差:{dev_from_ideal:.1f}px")
        if not std_ok: reasons.append(f"不平整(标准差:{std_d:.1f})")
        result.message = f"未准焦 ({', '.join(reasons)})"
        result.status_color = (0, 0, 255) # 红色

    # 5. 可视化绘制
    for i, (center, radius, diameter) in enumerate(final_4_circles):
        cx, cy = int(center[0]), int(center[1])
        pt_res = MultiCirclePointResult(
            index=i,
            label=MULTI_CIRCLE_LABELS[i],
            center=center,
            radius=radius,
            diameter=diameter
        )
        result.circles.append(pt_res)
        
        cv2.circle(output, (cx, cy), int(radius), result.status_color, 2, cv2.LINE_AA)
        cv2.circle(output, (cx, cy), 3, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.putText(output, f"{diameter:.1f}px", (cx - 25, cy + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, result.status_color, 1, cv2.LINE_AA)

    output = _draw_chinese_text(output, result.message, (20, 40), result.status_color)
    return output, result

def _draw_chinese_text(img: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)