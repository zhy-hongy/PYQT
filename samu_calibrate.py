#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
沙姆相机标定（高精度版本）- 增强畸变模型 + 角点几何质量筛选
"""

import numpy as np
import cv2
from scipy.optimize import least_squares
import glob
import os
from typing import List, Tuple, Dict, Optional
import json
import datetime
import matplotlib.pyplot as plt

# ============================================================
# 标定板类型
# ============================================================

CALIB_BOARD_CHESSBOARD = 0      #棋盘格标定板
CALIB_BOARD_CIRCLE_SYM = 1      #对称圆标定板
CALIB_BOARD_CIRCLE_ASYM = 2     #非对称圆标定板

CALIB_BOARD_TYPE = CALIB_BOARD_CHESSBOARD

# ============================================================
#  辅助函数
# ============================================================
def rodrigues_to_rotation_matrix(r_vec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(r_vec)
    if theta < 1e-8:
        return np.eye(3)
    r_hat = r_vec / theta
    K = np.array([[0, -r_hat[2], r_hat[1]],
                  [r_hat[2], 0, -r_hat[0]],
                  [-r_hat[1], r_hat[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

def compute_tilt_homography(tau: float, rho: float, d: float, lens_type: str = 'perspective') -> np.ndarray:
    c_tau, s_tau = np.cos(tau), np.sin(tau)
    c_rho, s_rho = np.cos(rho), np.sin(rho)
    Rt = np.array([
        [c_rho**2*(1-c_tau) + c_tau, c_rho*s_rho*(1-c_tau),     s_rho*s_tau],
        [c_rho*s_rho*(1-c_tau),      s_rho**2*(1-c_tau) + c_tau, -c_rho*s_tau],
        [-s_rho*s_tau,               c_rho*s_tau,                c_tau]
    ])
    r11, r12, r13 = Rt[0,0], Rt[0,1], Rt[0,2]
    r21, r22, r23 = Rt[1,0], Rt[1,1], Rt[1,2]
    r31, r32, r33 = Rt[2,0], Rt[2,1], Rt[2,2]
    if lens_type == 'telecentric':
        delta = r11*r22 - r12*r21
        H = np.array([[r22/delta, -r12/delta, 0],
                      [-r21/delta, r11/delta, 0],
                      [0, 0, 1]])
    else:
        H = np.array([
            [r11*r33 - r13*r31, r21*r33 - r23*r31, 0],
            [r12*r33 - r13*r32, r22*r33 - r23*r32, 0],
            [r13/d,             r23/d,             r33]
        ])
    return H

def apply_distortion(xu: np.ndarray, yu: np.ndarray,
                     k1: float, k2: float, k3: float,
                     p1: float, p2: float) -> Tuple[np.ndarray, np.ndarray]:
    """在未倾斜虚拟平面上施加径向和切向畸变（论文公式10-12）
    输入输出均为归一化坐标（无量纲）"""
    r2 = xu*xu + yu*yu
    r4 = r2*r2
    r6 = r2*r4
    radial = 1.0 + k1*r2 + k2*r4 + k3*r6
    x_rad = xu * radial
    y_rad = yu * radial
    # 切向畸变
    x_tan = 2.0*p1*xu*yu + p2*(r2 + 2.0*xu*xu)
    y_tan = p1*(r2 + 2.0*yu*yu) + 2.0*p2*xu*yu
    # xd = x_rad + x_tan
    # yd = y_rad + y_tan
    # x_tilt = alpha1 * tau * xu
    # y_tilt = alpha2 * tau * yu
    xd = x_rad + x_tan # + x_tilt
    yd = y_rad + y_tan # + y_tilt
    return xd, yd

def project_points_scheimpflug(P_w: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                               c: float, d: float, tau: float, rho: float,
                               k1: float, k2: float, k3: float,
                               p1: float, p2: float,
                               sx: float, sy: float, cx: float, cy: float) -> np.ndarray:
    """
    完整的透视倾斜相机投影（论文模型）
    流程：世界 -> 相机 -> 未倾斜理想点 -> 畸变(未倾斜) -> 倾斜单应Hp -> 像素
    """
    N = P_w.shape[0]
    R = rodrigues_to_rotation_matrix(rvec)
    P_c = (R @ P_w.T).T + tvec
    Xc, Yc, Zc = P_c[:,0], P_c[:,1], P_c[:,2]
    Zc = np.maximum(Zc, 1e-8)
    # 未倾斜虚拟平面上的理想针孔投影（距离 c）
    xu = c * Xc / Zc
    yu = c * Yc / Zc
     # 在未倾斜平面上施加畸变
    xd, yd = apply_distortion(xu, yu, k1, k2, k3, p1, p2)
    # 倾斜单应映射到实际传感器平面
    H = compute_tilt_homography(tau, rho, d, lens_type='perspective')
    pts_homo = H @ np.vstack((xd, yd, np.ones(N)))
    xt = pts_homo[0, :] / pts_homo[2, :]
    yt = pts_homo[1, :] / pts_homo[2, :]

    # 像素坐标
    u = xt / sx + cx
    v = yt / sy + cy
    return np.vstack((u, v)).T

# ============================================================
#  角点几何质量筛选函数（调试版）
# ============================================================

def filter_chessboard_corners(corners, pattern_size, max_line_error_px=2.0, max_dist_ratio=0.5, debug=False):
    """
    棋盘格角点质量筛选：间距一致性 + 行列直线拟合（基于垂直距离）
    
    参数:
        corners: 检测到的角点坐标，形状 (N,2) 或 (N,1,2)
        pattern_size: 棋盘格内角点尺寸 (cols, rows)
        max_line_error_px: 点到拟合直线的最大允许垂直距离（像素）
        max_dist_ratio: 间距标准差/均值的最大比例，超则整图无效
        debug: 是否打印调试信息
    返回:
        valid_mask: 布尔数组，长度为 N，True 表示角点有效
    """
    # 将角点转换为标准形状 (N,2)
    corners = np.array(corners)
    if corners.ndim == 3 and corners.shape[1] == 1:
        corners = corners.reshape(-1, 2)
    elif corners.ndim != 2 or corners.shape[1] != 2:
        return np.zeros(len(corners), dtype=bool)

    cols, rows = pattern_size
    N = len(corners)
    valid_mask = np.ones(N, dtype=bool)

    # 1. 数量检查
    if N != cols * rows:
        valid_mask[:] = False
        return valid_mask

    # 2. 间距一致性检验（水平与垂直方向）
    h_distances = []
    v_distances = []
    for r in range(rows):
        for c in range(cols - 1):
            idx1 = r * cols + c
            idx2 = r * cols + c + 1
            h_distances.append(np.linalg.norm(corners[idx2] - corners[idx1]))
    for c in range(cols):
        for r in range(rows - 1):
            idx1 = r * cols + c
            idx2 = (r + 1) * cols + c
            v_distances.append(np.linalg.norm(corners[idx2] - corners[idx1]))

    if len(h_distances) == 0 or len(v_distances) == 0:
        return valid_mask

    mean_h = np.mean(h_distances)
    std_h = np.std(h_distances)
    mean_v = np.mean(v_distances)
    std_v = np.std(v_distances)

    if debug:
        print(f"水平距离: 均值={mean_h:.2f}, 标准差={std_h:.2f}, CV={std_h/mean_h:.3f}")
        print(f"垂直距离: 均值={mean_v:.2f}, 标准差={std_v:.2f}, CV={std_v/mean_v:.3f}")

    if std_h / mean_h > max_dist_ratio or std_v / mean_v > max_dist_ratio:
        if debug:
            print("间距一致性检验失败")
        valid_mask[:] = False
        return valid_mask

    # 3. 辅助函数：拟合直线并返回点到直线的垂直距离
    def fit_line_and_distances(pts):
        """
        pts: (M,2) 点集
        返回: (M,) 每个点到拟合直线的垂直距离；拟合失败返回 None
        """
        if pts.shape[0] < 2:
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        var_x = np.var(x)
        var_y = np.var(y)

        # 所有点共线且与坐标轴平行的情况
        if var_x < 1e-8 and var_y < 1e-8:
            return np.zeros_like(x)   # 所有点重合（理论上不会发生）
        if var_x < 1e-8:   # x 几乎不变，拟合垂直线 x = constant
            x_mean = np.mean(x)
            distances = np.abs(x - x_mean)
            return distances
        if var_y < 1e-8:   # y 几乎不变，拟合水平线 y = constant
            y_mean = np.mean(y)
            distances = np.abs(y - y_mean)
            return distances

        # 一般情况：选择变化大的坐标作为自变量
        if var_x >= var_y:
            # 拟合 y = kx + b
            A = np.vstack([x, np.ones(len(x))]).T
            try:
                k, b = np.linalg.lstsq(A, y, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            # 点到直线 kx - y + b = 0 的垂直距离
            norm = np.sqrt(k * k + 1.0)
            distances = np.abs(k * x - y + b) / norm
        else:
            # 拟合 x = k' y + b'
            A = np.vstack([y, np.ones(len(y))]).T
            try:
                k_prime, b_prime = np.linalg.lstsq(A, x, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            # 直线方程: x - k'*y - b' = 0
            norm = np.sqrt(1.0 + k_prime * k_prime)
            distances = np.abs(x - k_prime * y - b_prime) / norm
        return distances

    # 4. 行拟合（垂直距离）
    for r in range(rows):
        row_indices = list(range(r * cols, (r + 1) * cols))
        pts = corners[row_indices]
        distances = fit_line_and_distances(pts)
        if distances is None:
            valid_mask[row_indices] = False
            if debug:
                print(f"行 {r} 拟合失败，全部标记为无效")
            continue
        invalid = distances > max_line_error_px
        if np.any(invalid):
            valid_mask[np.array(row_indices)[invalid]] = False
            if debug:
                print(f"行 {r} 最大垂直距离误差 {np.max(distances):.2f} px")

    # 5. 列拟合（垂直距离）
    for c in range(cols):
        col_indices = [r * cols + c for r in range(rows)]
        pts = corners[col_indices]
        distances = fit_line_and_distances(pts)
        if distances is None:
            valid_mask[col_indices] = False
            if debug:
                print(f"列 {c} 拟合失败，全部标记为无效")
            continue
        invalid = distances > max_line_error_px
        if np.any(invalid):
            valid_mask[np.array(col_indices)[invalid]] = False
            if debug:
                print(f"列 {c} 最大垂直距离误差 {np.max(distances):.2f} px")

    return valid_mask


def load_and_detect_calibration_images(
    image_dir: str,
    pattern_size: Tuple[int, int],
    square_size_mm: float,
    show_corners: bool = False,
    max_line_error_px=2.0,
    max_reject_ratio=0.5,
    debug=False
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise FileNotFoundError(f"在路径 {image_dir} 中没有找到图片！")
    print(f"找到 {len(image_paths)} 张图片，正在提取角点/圆点...")
    cols, rows = pattern_size
    if CALIB_BOARD_TYPE in (CALIB_BOARD_CHESSBOARD, CALIB_BOARD_CIRCLE_SYM):
        objp = np.zeros((cols * rows, 3), np.float64)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    elif CALIB_BOARD_TYPE == CALIB_BOARD_CIRCLE_ASYM:
        objp = np.zeros((cols * rows, 3), np.float64)
        idx_pt = 0
        for r in range(rows):
            for c in range(cols):
                objp[idx_pt, 0] = (2 * c + (r % 2)) * square_size_mm
                objp[idx_pt, 1] = r * square_size_mm
                idx_pt += 1
    else:
        raise ValueError("未知标定板类型")
    P_w_list = []
    P_img_list = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    show_flag = show_corners   # 初始为传入参数
    
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"{os.path.basename(path)}: 图片读取失败，跳过")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret = False
        corners_subpix = None
        if CALIB_BOARD_TYPE == CALIB_BOARD_CHESSBOARD:
            ret, corners = cv2.findChessboardCorners(
                gray, pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                if corners_subpix is None or len(corners_subpix) == 0:
                    print(f"{os.path.basename(path)}: 亚像素角点提取失败，跳过")
                    continue
                corners_subpix = corners_subpix.reshape(-1,2)
                debug_this = debug and idx == 0
                valid_mask = filter_chessboard_corners(
                    corners_subpix, pattern_size,
                    max_line_error_px, max_dist_ratio=0.5, debug=debug_this
                )
                reject_ratio = 1.0 - np.sum(valid_mask) / len(valid_mask)
                if reject_ratio > max_reject_ratio:
                    print(f"{os.path.basename(path)}: 角点几何质量不合格 (剔除比例 {reject_ratio:.1%})，跳过")
                    continue
            else:
                print(f"{os.path.basename(path)}: 未检测到棋盘格，跳过")
        elif CALIB_BOARD_TYPE in (CALIB_BOARD_CIRCLE_SYM, CALIB_BOARD_CIRCLE_ASYM):
            flags = cv2.CALIB_CB_SYMMETRIC_GRID if CALIB_BOARD_TYPE == CALIB_BOARD_CIRCLE_SYM else cv2.CALIB_CB_ASYMMETRIC_GRID
            ret, corners = cv2.findCirclesGrid(gray, pattern_size, flags=flags)
            if ret:
                corners_subpix = corners.reshape(-1,2)
            else:
                print(f"{os.path.basename(path)}: 未检测到圆点网格，跳过")
        else:
            raise ValueError("未知标定板类型")
        if ret and corners_subpix is not None:
            P_w_list.append(objp.copy())
            P_img_list.append(corners_subpix)
            print(f"{os.path.basename(path)}: 检测到 {len(corners_subpix)} 个特征点")
            if show_flag:
                cols, rows = pattern_size
                img_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                colors = [
                    (0, 255, 0),     # 绿
                    (0, 0, 255),     # 红
                    (255, 0, 0),     # 蓝
                    (0, 255, 255),   # 黄
                    (255, 0, 255),   # 品红
                    (255, 255, 0),   # 青
                    (0, 128, 255),   # 橙
                    (128, 0, 255),   # 紫
                    (255, 128, 0),   # 橙
                    (128, 255, 0),   # 浅绿
                    (255, 128, 128), # 粉
                    (128, 255, 255), # 浅蓝
                ]
                pts = corners_subpix  # shape (N, 2)
                # 每行不同颜色 + 折线连接
                for r in range(rows):
                    color = colors[r % len(colors)]
                    row_pts = []
                    for c in range(cols):
                        idx = r * cols + c
                        center = tuple(pts[idx].astype(int))
                        cv2.circle(img_draw, center, 3, color, -1)
                        row_pts.append(center)
                    cv2.polylines(img_draw, [np.array(row_pts)], False, color, 1)
                # 每列也用淡色折线连接
                for c in range(cols):
                    col_pts = []
                    for r in range(rows):
                        idx = r * cols + c
                        col_pts.append(tuple(pts[idx].astype(int)))
                    cv2.polylines(img_draw, [np.array(col_pts)], False, (180, 180, 180), 1)
                cv2.imshow('calibrate', img_draw)
                key = cv2.waitKey(0) & 0xFF   # 获取按键
                if key == 27:                 # ESC 键
                    show_flag = False
                    cv2.destroyAllWindows()
                    print("按下 ESC，后续图片将不再显示（但仍会继续检测角点）")
    cv2.destroyAllWindows()
    if len(P_w_list) < 3:
        raise ValueError(f"有效图片仅 {len(P_w_list)} 张，至少需要 3 张。")
    print(f"有效图片数量: {len(P_w_list)}")
    return P_w_list, P_img_list

# ============================================================
#  初始外参估计（solvePnP）
# ============================================================

def estimate_initial_extrinsics_scheimpflug(P_w_list, P_img_list, cam_params):
    """
    使用当前投影模型（固定内参）为每个视图估计外参初值。
    
    参数:
        P_w_list: 世界坐标点列表
        P_img_list: 图像点列表
        cam_params: 包含 c, d, tau, rho, sx, sy, cx, cy, k1, k2, k3, p1, p2 的字典
                    （畸变系数可为0，因为初值估计对畸变不敏感）
    
    返回:
        extrinsics: 列表，每个元素为 (rvec, tvec)
    """
    extrinsics = []
    for P_w, P_img in zip(P_w_list, P_img_list):
        # ---- 步骤1：用针孔模型获得粗略初值 ----
        fx = cam_params['c'] / cam_params['sx']
        fy = cam_params['c'] / cam_params['sy']
        K_pinhole = np.array([[fx, 0, cam_params['cx']],
                              [0, fy, cam_params['cy']],
                              [0, 0, 1]], dtype=np.float64)
        dist_pinhole = np.zeros(4)  # 无畸变
        _, rvec, tvec = cv2.solvePnP(P_w, P_img, K_pinhole, dist_pinhole,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        # ---- 步骤2：用完整倾斜模型优化外参（固定内参） ----
        def residual(x):
            r = x[:3]
            t = x[3:6]
            proj = project_points_scheimpflug(
                P_w, r, t,
                cam_params['c'], cam_params['d'], cam_params['tau'], cam_params['rho'],
                cam_params['k1'], cam_params['k2'], cam_params.get('k3', 0.0),
                cam_params['p1'], cam_params['p2'],
                cam_params['sx'], cam_params['sy'],
                cam_params['cx'], cam_params['cy']
            )
            return (proj - P_img).ravel()

        x0 = np.hstack([rvec.ravel(), tvec.ravel()])
        # 使用 Levenberg-Marquardt 进行快速优化
        res = least_squares(residual, x0, method='lm', max_nfev=500)
        opt_rvec = res.x[:3]
        opt_tvec = res.x[3:6]
        extrinsics.append((opt_rvec, opt_tvec))
        # print(f"视图外参优化: 初始重投影误差 = {np.sqrt(np.mean(residual(x0)**2)):.3f} px -> "
        #       f"最终 = {np.sqrt(np.mean(residual(res.x)**2)):.3f} px")
    return extrinsics

# ============================================================
#  残差函数
# ============================================================
def calibration_residuals(params, P_w_list, P_img_list, sx, sy,
                          distortion_model: str, lens_type: str = 'perspective'):
    num_views = len(P_w_list)
    if distortion_model == 'none':
        c, d, tau, rho, cx, cy = params[0:6]
        k1=k2=k3=p1=p2=0.0
        int_len = 6
    elif distortion_model == 'k1':
        c, d, tau, rho, cx, cy, k1 = params[0:7]
        k2=k3=p1=p2=0.0
        int_len = 7
    elif distortion_model == 'k2':
        c, d, tau, rho, cx, cy, k1, k2 = params[0:8]
        k3=p1=p2=0.0
        int_len = 8
    elif distortion_model == 'full':
        c, d, tau, rho, cx, cy, k1, k2, p1, p2 = params[0:10]
        k3=0.0
        int_len = 10
    elif distortion_model == 'full_k3':
        c, d, tau, rho, cx, cy, k1, k2, k3, p1, p2 = params[0:11]
        int_len = 11
    else:
        raise ValueError(f"未知的畸变模型: {distortion_model}")
    
    residuals = []
    for i in range(num_views):
        idx = int_len + i * 6
        rvec = params[idx:idx+3]
        tvec = params[idx+3:idx+6]
        P_pred = project_points_scheimpflug(P_w_list[i], rvec, tvec,
                                            c, d, tau, rho,
                                            k1, k2, k3, p1, p2,
                                            sx, sy, cx, cy)
        diff = P_pred - P_img_list[i]
        residuals.extend(diff.flatten())
    return np.array(residuals)

# ============================================================
#  主标定函数
# ============================================================
def calibrate_scheimpflug(P_w_list, P_img_list, image_shape: Tuple[int, int],
                          sx: float, sy: float,
                          init_intrinsic: Dict,
                          distortion_model: str = 'full_tilt',
                          lens_type: str = 'perspective') -> Dict:
    h, w = image_shape
    c0 = init_intrinsic.get('c', 8.0)
    d0 = init_intrinsic.get('d', c0)
    tau0 = init_intrinsic.get('tau', 0.0)
    rho0 = init_intrinsic.get('rho', 0.0)
    cx0 = init_intrinsic.get('cx', w / 2.0)
    cy0 = init_intrinsic.get('cy', h / 2.0)
    
    if distortion_model == 'none':
        init_int_vals = [c0, d0, tau0, rho0, cx0, cy0]
        lower_bounds = [5.0, 5.0, -np.radians(30), -np.pi, 600, 450]
        upper_bounds = [15.0, 15.0, np.radians(30), np.pi, 680, 530]
    elif distortion_model == 'k1':
        init_int_vals = [c0, d0, tau0, rho0, cx0, cy0, 0.0]
        lower_bounds = [5.0, 5.0, -np.radians(30), -np.pi, 600, 450, -1.0]
        upper_bounds = [15.0, 15.0, np.radians(30), np.pi, 680, 530,  1.0]
    elif distortion_model == 'k2':
        init_int_vals = [c0, d0, tau0, rho0, cx0, cy0, 0.0, 0.0]
        lower_bounds = [5.0, 7.5, -np.radians(30), -np.pi, 600, 450, -1.0, -1.0]
        upper_bounds = [15.0, 9.5, np.radians(30),  np.pi, 680, 530, 1.0, 1.0]
    elif distortion_model == 'full':
        init_int_vals = [c0, d0, tau0, rho0, cx0, cy0, 0.0, 0.0, 0.0, 0.0]
        lower_bounds = [ 5.0,  5.0, -np.radians(30), -np.pi, 600, 450, -1.0, -1.0, -0.5, -0.5]
        upper_bounds = [15.0, 25.0,  np.radians(30),  np.pi, 680, 530, 1.0, 1.0, 0.5, 0.5]
    elif distortion_model == 'full_k3':
        init_int_vals = [c0, d0, tau0, rho0, cx0, cy0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lower_bounds = [5.0,   5.0, -np.radians(15), -np.pi, 600, 450, -1.0, -1.0, -1.0, -0.5, -0.5]
        upper_bounds = [15.0, 15.0,  np.radians(15),  np.pi, 680, h-1,  1.0,  1.0,  1.0,  0.5,  0.5]

    # 构建一个临时参数字典（用于初值估计）
    tmp_params = {
        'c': c0, 'd': d0, 'tau': tau0, 'rho': rho0,
        'cx': cx0, 'cy': cy0,
        'k1': 0.0, 'k2': 0.0, 'k3': 0.0, 'p1': 0.0, 'p2': 0.0,
        'sx': sx, 'sy': sy
    }
    extrinsics_list = estimate_initial_extrinsics_scheimpflug(P_w_list, P_img_list, tmp_params)

    ext_params = []
    for rvec, tvec in extrinsics_list:
        ext_params.extend(rvec)
        ext_params.extend(tvec)
    
    init_params = np.array(init_int_vals + ext_params, dtype=np.float64)
    
    for _ in range(len(P_w_list)):
        lower_bounds.extend([-np.pi, -np.pi, -np.pi, -1e3, -1e3, 10.0])
        upper_bounds.extend([np.pi, np.pi, np.pi, 1e3, 1e3, 1e3])
    
    for i, (low, high) in enumerate(zip(lower_bounds, upper_bounds)):
        if init_params[i] < low:
            init_params[i] = low
        elif init_params[i] > high:
            init_params[i] = high
    
    print("\n正在启动非线性最小二乘优化...")
    result = least_squares(
        calibration_residuals,
        init_params,
        bounds=(lower_bounds, upper_bounds),
        args=(P_w_list, P_img_list, sx, sy, distortion_model, lens_type),
        method='trf',
        loss='soft_l1',
        # loss='linear',
        f_scale=1.0,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=5000,
        verbose=1
    )
    
    if not result.success:
        print("警告：优化未完全收敛。")
    
    if distortion_model == 'none':
        c, d, tau, rho, cx, cy = result.x[0:6]
        k1 = k2 = k3 = p1 = p2 = alpha1 = alpha2 = 0.0
        int_len = 6
    elif distortion_model == 'k1':
        c, d, tau, rho, cx, cy, k1 = result.x[0:7]
        k2 = k3 = p1 = p2 = alpha1 = alpha2 = 0.0
        int_len = 7
    elif distortion_model == 'k2':
        c, d, tau, rho, cx, cy, k1, k2 = result.x[0:8]
        k3 = p1 = p2 = alpha1 = alpha2 = 0.0
        int_len = 8
    elif distortion_model == 'full':
        c, d, tau, rho, cx, cy, k1, k2, p1, p2 = result.x[0:10]
        k3 = alpha1 = alpha2 = 0.0
        int_len = 10
    elif distortion_model == 'full_k3':
        c, d, tau, rho, cx, cy, k1, k2, k3, p1, p2 = result.x[0:11]
        alpha1 = alpha2 = 0.0
        int_len = 11
    
    total_err_sq = 0.0
    total_pts = 0
    for i in range(len(P_w_list)):
        idx = int_len + i * 6
        rvec = result.x[idx:idx+3]
        tvec = result.x[idx+3:idx+6]
        P_pred = project_points_scheimpflug(P_w_list[i], rvec, tvec,
                                            c, d, tau, rho,
                                            k1, k2, k3, p1, p2,
                                            sx, sy, cx, cy)
        diff = P_pred - P_img_list[i]
        total_err_sq += np.sum(diff**2)
        total_pts += len(P_img_list[i])
    rmse = np.sqrt(total_err_sq / (2 * total_pts))
    
    print(f"\n优化完成。最终重投影误差 RMSE = {rmse:.4f} 像素")
    print(f"  c = {c:.4f} mm")
    print(f"  d = {d:.4f} mm")
    print(f"  tau = {np.degrees(tau):.4f} 度")
    print(f"  rho = {np.degrees(rho):.4f} 度")
    print(f"  cx = {cx:.2f} 像素, cy = {cy:.2f} 像素")
    if distortion_model in ('k1','k2','full','full_tilt'):
        print(f"  k1 = {k1:.6f}, k2 = {k2:.6f}")
    if distortion_model in ('full','full_tilt'):
        print(f"  p1 = {p1:.6f}, p2 = {p2:.6f}")
    if distortion_model == 'full_tilt':
        print(f"  k3 = {k3:.6f}, alpha1 = {alpha1:.6f}, alpha2 = {alpha2:.6f}")
    if distortion_model in ('full_k3'):
        print(f"  k1 = {k1:.6f}, k2 = {k2:.6f}, k3 = {k3:.6f}, p1 = {p1:.6f}, p2 = {p2:.6f}")
    
    cam_params = {
        'c': c, 'd': d, 'tau': tau, 'rho': rho,
        'cx': cx, 'cy': cy,
        'k1': k1, 'k2': k2, 'k3': k3,
        'p1': p1, 'p2': p2,
        # 'alpha1': alpha1, 'alpha2': alpha2,
        'sx': sx, 'sy': sy,
        'image_shape': image_shape,
        'rmse': rmse,
        'distortion_model': distortion_model,
        'optim_success': result.success,
        'raw_x': result.x
    }
    return cam_params

# ============================================================
#  图像校正
# ============================================================

def rectify_tilt_image(img: np.ndarray, cam_params: Dict, output_size: Optional[Tuple[int,int]] = None) -> np.ndarray:
    h_in, w_in = img.shape[:2]
    if output_size is None:
        out_w, out_h = w_in, h_in
    else:
        out_w, out_h = output_size
    
    sx = cam_params['sx']
    sy = cam_params['sy']
    c = cam_params['c']
    d = cam_params['d']
    tau = cam_params['tau']
    rho = cam_params['rho']
    cx = cam_params['cx']
    cy = cam_params['cy']
    k1 = cam_params['k1']
    k2 = cam_params['k2']
    k3 = cam_params.get('k3', 0.0)
    p1 = cam_params['p1']
    p2 = cam_params['p2']
    alpha1 = cam_params.get('alpha1', 0.0)
    alpha2 = cam_params.get('alpha2', 0.0)
    
    rectified = np.zeros((out_h, out_w, 3), dtype=np.uint8) if img.ndim == 3 else np.zeros((out_h, out_w), dtype=np.uint8)
    H_tilt = compute_tilt_homography(tau, rho, d, lens_type='perspective')
    H_inv = np.linalg.inv(H_tilt)
    
    for v_out in range(out_h):
        for u_out in range(out_w):
            x_ideal = (u_out - cx) * sx
            y_ideal = (v_out - cy) * sy
            pt = H_inv @ np.array([x_ideal, y_ideal, 1.0])
            x_tilt = pt[0] / pt[2]
            y_tilt = pt[1] / pt[2]
            r2 = x_tilt*x_tilt + y_tilt*y_tilt
            r4 = r2*r2
            r6 = r2*r4
            radial = 1.0 + k1*r2 + k2*r4 + k3*r6
            x_rad = x_tilt * radial
            y_rad = y_tilt * radial
            x_tan = 2.0*p1*x_tilt*y_tilt + p2*(r2 + 2.0*x_tilt*x_tilt)
            y_tan = p1*(r2 + 2.0*y_tilt*y_tilt) + 2.0*p2*x_tilt*y_tilt
            x_tilt_comp = alpha1 * tau * x_tilt
            y_tilt_comp = alpha2 * tau * y_tilt
            x_d = x_rad + x_tan + x_tilt_comp
            y_d = y_rad + y_tan + y_tilt_comp
            u_orig = x_d / sx + cx
            v_orig = y_d / sy + cy
            if 0 <= u_orig < w_in-1 and 0 <= v_orig < h_in-1:
                u0, u1 = int(np.floor(u_orig)), int(np.ceil(u_orig))
                v0, v1 = int(np.floor(v_orig)), int(np.ceil(v_orig))
                du = u_orig - u0
                dv = v_orig - v0
                if img.ndim == 3:
                    for ch in range(3):
                        val = (1-du)*(1-dv)*img[v0,u0,ch] + du*(1-dv)*img[v0,u1,ch] + \
                              (1-du)*dv*img[v1,u0,ch] + du*dv*img[v1,u1,ch]
                        rectified[v_out, u_out, ch] = np.clip(val, 0, 255)
                else:
                    val = (1-du)*(1-dv)*img[v0,u0] + du*(1-dv)*img[v0,u1] + \
                          (1-du)*dv*img[v1,u0] + du*dv*img[v1,u1]
                    rectified[v_out, u_out] = np.clip(val, 0, 255)
    return rectified

def validate_scheimpflug_calibration(
    image_paths,  # 可以是字符串(单张图片路径)或列表(多张图片路径)
    cam_params,
    pattern_size,
    square_size_mm,
    show=False
):
    """
    验证沙姆相机标定精度，支持多张图像，输出每张图的详细误差统计。

    参数:
        image_paths: str 或 List[str] - 图像路径或路径列表
        cam_params: dict - 标定参数
        pattern_size: tuple (cols, rows) - 棋盘格内角点数量
        square_size_mm: float - 棋盘格方格边长 (mm)
        show: bool - 是否显示第一张图像的可视化结果

    返回:
        dict: {
            'per_image': [{'path': str, 'mean': float, 'rms': float, 'max': float,
                           'dx_mean': float, 'dx_std': float,
                           'dy_mean': float, 'dy_std': float, ...}, ...],
            'overall': {'mean': float, 'rms': float, 'max': float,
                        'dx_mean': float, 'dx_std': float,
                        'dy_mean': float, 'dy_std': float}
        }
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    cols, rows = pattern_size
    # 世界坐标点 (Z=0)
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    all_errors = []  # 存储每张图像的所有角点误差（用于整体统计）
    per_image_stats = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图像 {img_path}，跳过")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测角点
        ret, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ret:
            print(f"警告: {img_path} 角点检测失败，跳过")
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ).reshape(-1, 2)

        # 使用优化后的外参（调用 least_squares 进行单图外参优化）
        # 注意：这里为了每张图独立验证，会重新估计外参（固定内参）
        # 这样得到的是该图像的最佳拟合外参下的误差，而不是全局外参。
        # 如果希望使用全局外参，可以跳过此步，但需要事先有全局外参。
        # 为简单起见，这里使用 solvePnP 初值 + 优化外参（与之前 validation 逻辑一致）

        fx = cam_params['c'] / cam_params['sx']
        fy = cam_params['c'] / cam_params['sy']
        K_pinhole = np.array([
            [fx, 0, cam_params['cx']],
            [0, fy, cam_params['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_pinhole = np.array([cam_params['k1'], cam_params['k2'],
                                 cam_params['p1'], cam_params['p2']], dtype=np.float64)

        _, rvec_init, tvec_init = cv2.solvePnP(objp, corners, K_pinhole, dist_pinhole,
                                               flags=cv2.SOLVEPNP_ITERATIVE)

        def loss_function(ext_params):
            rvec_opt = ext_params[0:3]
            tvec_opt = ext_params[3:6]
            proj_pts = project_points_scheimpflug(
                objp,
                rvec_opt,
                tvec_opt,
                cam_params['c'],
                cam_params['d'],
                cam_params['tau'],
                cam_params['rho'],
                cam_params['k1'],
                cam_params['k2'],
                cam_params.get('k3', 0.0),
                cam_params['p1'],
                cam_params['p2'],
                cam_params['sx'],
                cam_params['sy'],
                cam_params['cx'],
                cam_params['cy']
            )
            return (proj_pts - corners).ravel()

        x0 = np.hstack((rvec_init.ravel(), tvec_init.ravel()))
        res = least_squares(loss_function, x0, method='lm')
        rvec_opt = res.x[0:3]
        tvec_opt = res.x[3:6]

        # 计算投影和误差
        proj = project_points_scheimpflug(
            objp,
            rvec_opt,
            tvec_opt,
            cam_params['c'],
            cam_params['d'],
            cam_params['tau'],
            cam_params['rho'],
            cam_params['k1'],
            cam_params['k2'],
            cam_params.get('k3', 0.0),
            cam_params['p1'],
            cam_params['p2'],
            cam_params['sx'],
            cam_params['sy'],
            cam_params['cx'],
            cam_params['cy']
        )

        errors = np.linalg.norm(proj - corners, axis=1)
        dx = proj[:, 0] - corners[:, 0]
        dy = proj[:, 1] - corners[:, 1]

        stats = {
            'path': img_path,
            'mean': float(np.mean(errors)),
            'rms': float(np.sqrt(np.mean(errors**2))),
            'max': float(np.max(errors)),
            'dx_mean': float(np.mean(dx)),
            'dx_std': float(np.std(dx)),
            'dy_mean': float(np.mean(dy)),
            'dy_std': float(np.std(dy)),
            'num_corners': len(corners)
        }
        per_image_stats.append(stats)
        all_errors.extend(errors)
        all_dx = dx if 'all_dx' not in locals() else np.concatenate((all_dx, dx))
        all_dy = dy if 'all_dy' not in locals() else np.concatenate((all_dy, dy))

        # 可视化第一张图像（如果show=True）
        if show and img_path == image_paths[0]:
            vis = img.copy()
            for p_det, p_proj in zip(corners, proj):
                p_det = tuple(p_det.astype(int))
                p_proj = tuple(p_proj.astype(int))
                cv2.circle(vis, p_det, 3, (0, 255, 0), -1)
                cv2.circle(vis, p_proj, 3, (0, 0, 255), -1)
                cv2.line(vis, p_det, p_proj, (255, 0, 0), 1)
            cv2.imshow("Validation (Green: detected, Red: projected)", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if not per_image_stats:
        raise RuntimeError("没有成功处理任何图像")

    # 整体统计
    overall_errors = np.array(all_errors)
    overall_dx = np.array(all_dx)
    overall_dy = np.array(all_dy)

    overall = {
        'mean': float(np.mean(overall_errors)),
        'rms': float(np.sqrt(np.mean(overall_errors**2))),
        'max': float(np.max(overall_errors)),
        'dx_mean': float(np.mean(overall_dx)),
        'dx_std': float(np.std(overall_dx)),
        'dy_mean': float(np.mean(overall_dy)),
        'dy_std': float(np.std(overall_dy))
    }

    # 打印每张图的统计
    # print("\n==================== 逐图像验证统计 ====================")
    # for s in per_image_stats:
    #     print(f"图像: {os.path.basename(s['path'])}")
    #     print(f"  平均误差: {s['mean']:.4f} px, RMS: {s['rms']:.4f} px, 最大: {s['max']:.4f} px")
    #     print(f"  dx: 均值={s['dx_mean']:.4f}, 标准差={s['dx_std']:.4f}")
    #     print(f"  dy: 均值={s['dy_mean']:.4f}, 标准差={s['dy_std']:.4f}")
    #     print("")
    print("==================== 整体统计 ====================")
    print(f"整体平均误差: {overall['mean']:.4f} px")
    print(f"整体RMS: {overall['rms']:.4f} px")
    print(f"整体最大误差: {overall['max']:.4f} px")
    print(f"整体dx: 均值={overall['dx_mean']:.4f}, 标准差={overall['dx_std']:.4f}")
    print(f"整体dy: 均值={overall['dy_mean']:.4f}, 标准差={overall['dy_std']:.4f}")
    print("===================================================\n")

    return {
        'per_image': per_image_stats,
        'overall': overall
    }

def analyze_overall_coverage(P_img_list, image_shape, grid=(4,4), verbose=True):
    """
    分析多张标定图像中所有角点在图像上的整体覆盖情况。
    
    参数:
        P_img_list: list of (N_i, 2) 角点坐标（像素）列表
        image_shape: (height, width)
        grid: (rows, cols) 将图像划分的网格数
        verbose: 是否打印详细统计
    
    返回:
        coverage_ok: bool，是否满足均匀覆盖（可根据实际情况设定判断标准）
        stats: dict，包含网格计数、比例等
    """
    h, w = image_shape
    rows, cols = grid
    cell_h = h / rows
    cell_w = w / cols
    
    # 统计每个网格的总角点数
    grid_counts = np.zeros((rows, cols), dtype=int)
    total_corners = 0
    
    for corners in P_img_list:
        for (x, y) in corners:
            # 边界处理
            col = int(x // cell_w) if x < w else cols-1
            row = int(y // cell_h) if y < h else rows-1
            col = min(col, cols-1)
            row = min(row, rows-1)
            grid_counts[row, col] += 1
            total_corners += 1
    
    if total_corners == 0:
        print("无角点数据！")
        return False, {}
    
    grid_ratios = grid_counts / total_corners
    
    # 统计覆盖指标
    min_ratio = np.min(grid_ratios)
    max_ratio = np.max(grid_ratios)
    mean_ratio = np.mean(grid_ratios)
    std_ratio = np.std(grid_ratios)
    non_zero_grids = np.sum(grid_counts > 0)
    total_grids = rows * cols
    
    if verbose:
        print("\n========== 整体角点覆盖分析 ==========")
        print(f"总角点数: {total_corners}")
        print(f"图像区域: {w} x {h} 像素，划分 {rows}x{cols} 网格")
        print(f"每个网格的角点占比:")
        for r in range(rows):
            row_str = " ".join([f"{grid_ratios[r,c]:.2%}" for c in range(cols)])
            print(f"  行{r}: {row_str}")
        print(f"最小网格占比: {min_ratio:.2%}")
        print(f"最大网格占比: {max_ratio:.2%}")
        print(f"平均网格占比: {mean_ratio:.2%} ± {std_ratio:.2%}")
        print(f"覆盖网格数: {non_zero_grids}/{total_grids} ({non_zero_grids/total_grids:.1%})")
        print("====================================\n")
    
    # 判断覆盖是否合格（可根据需要调整阈值）
    # 例如：要求覆盖所有网格，且最小占比不低于 2%（保证每个区域都有足够数据）
    coverage_ok = (non_zero_grids == total_grids) and (min_ratio >= 0.02)
    if not coverage_ok:
        print("警告：整体角点覆盖不足！建议增加标定板在不同区域（边缘、角落）的图像。")
    
    stats = {
        'total_corners': total_corners,
        'grid_counts': grid_counts,
        'grid_ratios': grid_ratios,
        'min_ratio': min_ratio,
        'max_ratio': max_ratio,
        'mean_ratio': mean_ratio,
        'std_ratio': std_ratio,
        'non_zero_grids': non_zero_grids,
        'total_grids': total_grids,
        'coverage_ok': coverage_ok
    }
    return coverage_ok, stats



def plot_coverage_heatmap(stats, image_shape, save_path=None):
    grid_counts = stats['grid_counts']
    plt.imshow(grid_counts, cmap='hot', interpolation='nearest')
    plt.colorbar(label='角点数量')
    plt.title('角点覆盖热力图')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_all_corners_scatter(P_img_list, image_shape, save_path=None):
    plt.figure(figsize=(8, 6))
    for corners in P_img_list:
        plt.scatter(corners[:,0], corners[:,1], s=1, alpha=0.5, c='blue')
    plt.xlim(0, image_shape[1])
    plt.ylim(image_shape[0], 0)   # 图像原点在左上角，需要翻转y轴
    plt.gca().invert_yaxis()
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.title('All corners from all images')
    # if save_path:
        # plt.savefig(save_path)
    plt.show()
    
# ============================================================
#  主程序
# ============================================================
if __name__ == "__main__":
    IMAGE_DIR = r"./saved_images"  # 请替换为实际的标定图片目录
    PATTERN_SIZE = (11, 8)         # 注意：请根据实际棋盘格内角点行列数设置
    SQUARE_SIZE_MM = 5
    PIXEL_SIZE_X_MM = 0.004
    PIXEL_SIZE_Y_MM = 0.004
    INIT_INTRINSIC = {
        'c': 8.7, 'd': 15, 'tau': np.radians(9.3), 'rho': np.radians(0.0),
        'cx': 640.0, 'cy': 512.0
    }
    DISTORTION_MODEL = 'full_k3'# 'none', 'k1', 'k2', 'full', 'full_k3'
    LENS_TYPE = 'perspective'
    SHOW_CORNERS = True
    # 放宽筛选参数 
    MAX_LINE_ERROR_PX = 0.5
    MAX_REJECT_RATIO = 0.6
    DEBUG_CORNERS = True
    
    # 验证用测试图片路径
    TEST_IMAGE_PATH = r"./saved_images"
    test_images = glob.glob(f"{TEST_IMAGE_PATH}/*.png") + glob.glob(f"{TEST_IMAGE_PATH}/*.jpg") \
                + glob.glob(f"{TEST_IMAGE_PATH}/*.jpeg") + glob.glob(f"{TEST_IMAGE_PATH}/*.bmp")\
                + glob.glob(f"{TEST_IMAGE_PATH}/*.tif") + glob.glob(f"{TEST_IMAGE_PATH}/*.tiff")
    # -------------------- 创建结果保存目录 --------------------
    save_dir = "./calibration_data"
    os.makedirs(save_dir, exist_ok=True)
    
    # 检测角点（带几何筛选）
    P_w_list, P_img_list = load_and_detect_calibration_images(
        IMAGE_DIR, PATTERN_SIZE, SQUARE_SIZE_MM, show_corners=SHOW_CORNERS,
        max_line_error_px=MAX_LINE_ERROR_PX, max_reject_ratio=MAX_REJECT_RATIO,
        debug=DEBUG_CORNERS
    )
        
    # ===== 自动识别图片类型，获取第一张图片的尺寸 =====
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    sample_path = None

    for ext in extensions:
        # 小写扩展名
        files = glob.glob(os.path.join(IMAGE_DIR, ext))
        if files:
            sample_path = files[0]
            break
        # 大写扩展名
        files = glob.glob(os.path.join(IMAGE_DIR, ext.upper()))
        if files:
            sample_path = files[0]
            break

    if sample_path is None:
        raise FileNotFoundError(f"在目录 {IMAGE_DIR} 中没有找到任何图片文件（支持格式：jpg/jpeg/png/bmp/tif/tiff）")

    sample_img = cv2.imread(sample_path)
    img_h, img_w = sample_img.shape[:2]
    # print(f"使用图片 {os.path.basename(sample_path)} 获取图像尺寸: {img_w} x {img_h}")
    
    if P_img_list:
        coverage_ok, stats = analyze_overall_coverage(P_img_list, (img_h, img_w), grid=(4,4))
        # plot_coverage_heatmap(stats, (img_h, img_w), save_path=os.path.join(save_dir, "coverage_heatmap.png"))
        # plot_all_corners_scatter(P_img_list, (img_h, img_w), save_path=os.path.join(save_dir, "corners_scatter.png"))
        if not coverage_ok:
            print("覆盖不均匀警告...")
                # exit(1)
    # 标定
    cam_params = calibrate_scheimpflug(
        P_w_list, P_img_list, (img_h, img_w),
        PIXEL_SIZE_X_MM, PIXEL_SIZE_Y_MM,
        INIT_INTRINSIC,
        distortion_model=DISTORTION_MODEL,
        lens_type=LENS_TYPE
    )
    

    result = validate_scheimpflug_calibration(
        test_images,
        cam_params,
        PATTERN_SIZE,
        SQUARE_SIZE_MM,
        show=True
    )

    # 访问每张图像统计
    # for img_stat in result['per_image']:
    #     print(img_stat['path'], img_stat['dx_mean'], img_stat['dy_mean'])

    # 访问整体统计
    overall = result['overall']
        
    # -------------------- 保存结果（JSON格式） --------------------
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"calibration_{timestamp}.json")

    # 准备要保存的数据（将numpy类型转换为Python原生类型）
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        return obj

    # 拷贝参数字典并转换
    saved_params = {}
    for key, value in cam_params.items():
        # raw_x 可能非常大，建议不保存（或者保存为列表）
        if key == 'raw_x':
            # 可以选择保存为列表，但会占用大量空间
            # saved_params[key] = value.tolist()
            continue   # 跳过 raw_x，减小文件体积
        if key in ('tau', 'rho'):
            # 将弧度转换为度数后保存
            saved_params[key] = float(np.degrees(value))
        else:
            saved_params[key] = serialize(value)

    # 加入附加信息
    saved_params['timestamp'] = timestamp
    saved_params['config'] = {
        'pattern_size': PATTERN_SIZE,
        'square_size_mm': SQUARE_SIZE_MM,
        'pixel_size_x_mm': PIXEL_SIZE_X_MM,
        'pixel_size_y_mm': PIXEL_SIZE_Y_MM,
        'distortion_model': DISTORTION_MODEL,
        'lens_type': LENS_TYPE,
        'max_line_error_px': MAX_LINE_ERROR_PX,
        'max_reject_ratio': MAX_REJECT_RATIO,
        'image_dir': IMAGE_DIR,
        'test_image': TEST_IMAGE_PATH
    }
    if result is not None:
        saved_params['validation'] = serialize(result)

    # 写入JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(saved_params, f, indent=4, ensure_ascii=False)

    print(f"\n标定结果已保存至: {save_path}")    
    
    # 校正一张示例图像
    test_img = cv2.imread(sample_path)
    if test_img is not None:
        rectified = rectify_tilt_image(test_img, cam_params)
        cv2.imshow("rectified", rectified)
        cv2.waitKey(0)
    #     cv2.imwrite("rectified_example.jpg", rectified)
    #     print("\n校正后的图像已保存至 rectified_example.jpg")
    
    print("\n" + "="*60)
    print("高精度标定提醒：...")
    print("="*60)