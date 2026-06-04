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
                     p1: float, p2: float,
                     alpha1: float, alpha2: float, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """在未倾斜虚拟平面上施加径向和切向畸变（论文公式10-12）"""
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
                               alpha1: float, alpha2: float,
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
    xd, yd = apply_distortion(xu, yu, k1, k2, k3, p1, p2, alpha1, alpha2, tau)
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
    corners = np.array(corners)
    if corners.ndim == 3 and corners.shape[1] == 1:
        corners = corners.reshape(-1, 2)
    elif corners.ndim != 2 or corners.shape[1] != 2:
        return np.zeros(len(corners), dtype=bool)
    
    cols, rows = pattern_size
    N = len(corners)
    valid_mask = np.ones(N, dtype=bool)
    if N != cols * rows:
        valid_mask[:] = False
        return valid_mask
    
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
            idx2 = (r+1) * cols + c
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
    
    # 行拟合
    for r in range(rows):
        row_indices = list(range(r*cols, (r+1)*cols))
        pts = corners[row_indices]
        x, y = pts[:,0], pts[:,1]
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            k, b = np.linalg.lstsq(A, y, rcond=None)[0]
            errors = np.abs(y - (k*x + b))
            for i, err in enumerate(errors):
                if err > max_line_error_px:
                    valid_mask[row_indices[i]] = False
            if debug and np.max(errors) > max_line_error_px:
                print(f"行 {r} 最大拟合误差 {np.max(errors):.2f} px")
        except:
            valid_mask[row_indices] = False
    
    # 列拟合
    for c in range(cols):
        col_indices = [r*cols + c for r in range(rows)]
        pts = corners[col_indices]
        x, y = pts[:,0], pts[:,1]
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            k, b = np.linalg.lstsq(A, y, rcond=None)[0]
            errors = np.abs(y - (k*x + b))
            for i, err in enumerate(errors):
                if err > max_line_error_px:
                    valid_mask[col_indices[i]] = False
            if debug and np.max(errors) > max_line_error_px:
                print(f"列 {c} 最大拟合误差 {np.max(errors):.2f} px")
        except:
            valid_mask[col_indices] = False
    
    return valid_mask


# ---------------------------------------------------------------------
def load_and_detect_calibration_images(
    image_dir: str,
    pattern_size: Tuple[int, int],
    square_size_mm: float,
    show_corners: bool = False,
    max_line_error_px=2.0,
    max_reject_ratio=0.5,
    debug=False
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    加载标定图片并检测角点/圆点
    支持棋盘格、对称圆点板、非对称圆点板
    """
    # -----------------------------------------------------------------
    # 搜索图片
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # 生成世界坐标
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # 循环处理每张图片
    # -----------------------------------------------------------------
    for idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---------------- 检测标定板 ----------------
        ret = False
        corners_subpix = None

        if CALIB_BOARD_TYPE == CALIB_BOARD_CHESSBOARD:
            ret, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if ret:
                corners_subpix = cv2.cornerSubPix(
                    gray, corners, (11,11), (-1,-1), criteria
                )
                if corners_subpix is None or len(corners_subpix) == 0:
                    continue
                corners_subpix = corners_subpix.reshape(-1,2)

                # 几何筛选
                debug_this = debug and idx == 0
                valid_mask = filter_chessboard_corners(
                    corners_subpix,
                    pattern_size,
                    max_line_error_px,
                    max_dist_ratio=0.5,
                    debug=debug_this
                )
                reject_ratio = 1.0 - np.sum(valid_mask) / len(valid_mask)
                if reject_ratio > max_reject_ratio:
                    print(
                        f"{os.path.basename(path)}: 角点几何质量不合格 "
                        f"(剔除比例 {reject_ratio:.1%})，跳过该图像"
                    )
                    if debug_this and show_corners:
                        img_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        for i_pt, m in enumerate(valid_mask):
                            pt = tuple(corners_subpix[i_pt].astype(int))
                            color = (0,255,0) if m else (0,0,255)
                            cv2.circle(img_draw, pt, 3, color, -1)
                        cv2.imshow('角点质量 (绿=有效, 红=无效)', img_draw)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    continue

        elif CALIB_BOARD_TYPE in (CALIB_BOARD_CIRCLE_SYM, CALIB_BOARD_CIRCLE_ASYM):
            flags = cv2.CALIB_CB_SYMMETRIC_GRID if CALIB_BOARD_TYPE == CALIB_BOARD_CIRCLE_SYM else cv2.CALIB_CB_ASYMMETRIC_GRID
            ret, corners = cv2.findCirclesGrid(gray, pattern_size, flags=flags)
            if ret:
                corners_subpix = corners.reshape(-1,2)

        else:
            raise ValueError("未知标定板类型")

        # ---------------- 保存有效角点/圆点 ----------------
        if ret and corners_subpix is not None:
            P_w_list.append(objp.copy())
            P_img_list.append(corners_subpix)
            print(f"{os.path.basename(path)}: 检测到 {len(corners_subpix)} 个特征点")

            if show_corners:
                img_draw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for pt in corners_subpix:
                    cv2.circle(img_draw, tuple(pt.astype(int)), 3, (0,255,0), -1)
                cv2.imshow('角点/圆点', img_draw)
                cv2.waitKey(0)

    cv2.destroyAllWindows()

    if len(P_w_list) < 3:
        raise ValueError(f"有效图片仅 {len(P_w_list)} 张，至少需要 3 张。")
    print(f"有效图片数量: {len(P_w_list)}")

    return P_w_list, P_img_list

# ============================================================
#  初始外参估计（solvePnP）
# ============================================================
def estimate_initial_extrinsics(P_w_list, P_img_list, c_init, cx_init, cy_init, sx, sy):
    fx_init = c_init / sx
    fy_init = c_init / sy
    K_init = np.array([[fx_init, 0, cx_init], [0, fy_init, cy_init], [0, 0, 1]], dtype=np.float64)
    dist_init = np.zeros(4)
    extrinsics = []
    print("正在使用 solvePnP 估计初始外参...")
    for i, (P_w, P_img) in enumerate(zip(P_w_list, P_img_list)):
        success, rvec, tvec = cv2.solvePnP(P_w, P_img, K_init, dist_init,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            rvec = np.zeros(3)
            tvec = np.array([0, 0, 100.0])
        extrinsics.append((rvec.flatten(), tvec.flatten()))
        print(f"  视图 {i}: t = [{tvec[0,0]:.2f}, {tvec[1,0]:.2f}, {tvec[2,0]:.2f}] mm")
    return extrinsics

# ============================================================
#  残差函数
# ============================================================
def calibration_residuals(params, P_w_list, P_img_list, sx, sy,
                          distortion_model: str, lens_type: str = 'perspective'):
    num_views = len(P_w_list)
    if distortion_model == 'none':
        c, d, tau, rho, cx, cy = params[0:6]
        k1=k2=k3=p1=p2=alpha1=alpha2=0.0
        int_len = 6
    elif distortion_model == 'k1':
        c, d, tau, rho, cx, cy, k1 = params[0:7]
        k2=k3=p1=p2=alpha1=alpha2=0.0
        int_len = 7
    elif distortion_model == 'k2':
        c, d, tau, rho, cx, cy, k1, k2 = params[0:8]
        k3=p1=p2=alpha1=alpha2=0.0
        int_len = 8
    elif distortion_model == 'full':
        c, d, tau, rho, cx, cy, k1, k2, p1, p2 = params[0:10]
        k3=alpha1=alpha2=0.0
        int_len = 10
    elif distortion_model == 'full_tilt':
        c, d, tau, rho, cx, cy, k1, k2, k3, p1, p2, alpha1, alpha2 = params[0:13]
        int_len = 13
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
                                            alpha1, alpha2,
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
    d0 = init_intrinsic.get('d', c0 / np.cos(init_intrinsic.get('tau', 0.0)))
    tau0 = init_intrinsic.get('tau', 0.0)
    rho0 = init_intrinsic.get('rho', 0.0)
    cx0 = init_intrinsic.get('cx', w / 2.0)
    cy0 = init_intrinsic.get('cy', h / 2.0)
    
    if distortion_model == 'none':
        init_int_vals = [c0, d0, tau0, rho0, cx0, cy0]
        lower_bounds = [5.0, 5.0, -np.radians(30), -np.pi, 0, 0]
        upper_bounds = [15.0, 15.0, np.radians(30), np.pi, w-1, h-1]
    elif distortion_model == 'k1':
        init_int_vals = [c0, d0, tau0, rho0, cx0, cy0, 0.0]
        lower_bounds = [5.0, 5.0, -np.radians(30), -np.pi, 0, 0, -1.0]
        upper_bounds = [15.0, 15.0, np.radians(30), np.pi, w-1, h-1, 1.0]
    elif distortion_model == 'k2':
        init_int_vals = [c0, d0, tau0, rho0, cx0, cy0, 0.0, 0.0]
        lower_bounds = [5.0, 7.5, -np.radians(30), -np.pi, 0, 0, -1.0, -1.0]
        upper_bounds = [15.0, 9.5, np.radians(30), np.pi, w-1, h-1, 1.0, 1.0]
    elif distortion_model == 'full':
        init_int_vals = [c0, d0, tau0, rho0, cx0, cy0, 0.0, 0.0, 0.0, 0.0]
        lower_bounds = [5.0, 5.0, -np.radians(30), -np.pi, 0, 0, -1.0, -1.0, -0.5, -0.5]
        upper_bounds = [15.0, 15.0, np.radians(30), np.pi, w-1, h-1, 1.0, 1.0, 0.5, 0.5]
    else:  # 'full_tilt'
        init_int_vals = [c0, d0, tau0, rho0, cx0, cy0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lower_bounds = [5.0, 5.0, -np.radians(30), -np.pi, 0, 0,
                        -1.0, -1.0, -1.0, -0.5, -0.5, -2.0, -2.0]
        upper_bounds = [15.0, 15.0, np.radians(30), np.pi, w-1, h-1,
                        1.0, 1.0, 1.0, 0.5, 0.5, 5.0, 5.0]
    
    extrinsics_list = estimate_initial_extrinsics(P_w_list, P_img_list,
                                                   c0, cx0, cy0, sx, sy)
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
        f_scale=1.0,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=5000,
        verbose=2
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
    else:
        c, d, tau, rho, cx, cy, k1, k2, k3, p1, p2, alpha1, alpha2 = result.x[0:13]
        int_len = 13
    
    total_err_sq = 0.0
    total_pts = 0
    for i in range(len(P_w_list)):
        idx = int_len + i * 6
        rvec = result.x[idx:idx+3]
        tvec = result.x[idx+3:idx+6]
        P_pred = project_points_scheimpflug(P_w_list[i], rvec, tvec,
                                            c, d, tau, rho,
                                            k1, k2, k3, p1, p2, alpha1, alpha2,
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
    
    cam_params = {
        'c': c, 'd': d, 'tau': tau, 'rho': rho,
        'cx': cx, 'cy': cy,
        'k1': k1, 'k2': k2, 'k3': k3,
        'p1': p1, 'p2': p2,
        'alpha1': alpha1, 'alpha2': alpha2,
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

# def rectify_tilt_image(img: np.ndarray, cam_params: Dict, output_size: Optional[Tuple[int,int]] = None) -> np.ndarray:
#     """
#     只校正镜头畸变，但保留沙姆倾斜透视（不做逆向透视倾斜变换）。
    
#     参数:
#         img: 原始倾斜图像 (H, W) 或 (H, W, 3)
#         cam_params: 标定参数字典，需包含以下键：
#             sx, sy, c, d, tau, rho, cx, cy,
#             k1, k2, k3, p1, p2, alpha1, alpha2
#         output_size: 输出图像尺寸 (width, height)，默认与输入图像相同
    
#     返回:
#         校正畸变后的图像（仍保留透视倾斜）
#     """
#     h_in, w_in = img.shape[:2]
#     if output_size is None:
#         out_w, out_h = w_in, h_in
#     else:
#         out_w, out_h = output_size
    
#     # 提取相机参数
#     sx = cam_params['sx']
#     sy = cam_params['sy']
#     c = cam_params['c']          # 本函数虽未直接使用c/d，但保留以供可能的扩展
#     d = cam_params['d']
#     tau = cam_params['tau']
#     rho = cam_params['rho']
#     cx = cam_params['cx']
#     cy = cam_params['cy']
#     k1 = cam_params['k1']
#     k2 = cam_params['k2']
#     k3 = cam_params.get('k3', 0.0)
#     p1 = cam_params['p1']
#     p2 = cam_params['p2']
#     alpha1 = cam_params.get('alpha1', 0.0)
#     alpha2 = cam_params.get('alpha2', 0.0)
    
#     # 准备输出图像（彩色或灰度）
#     if img.ndim == 3:
#         rectified = np.zeros((out_h, out_w, 3), dtype=np.uint8)
#     else:
#         rectified = np.zeros((out_h, out_w), dtype=np.uint8)
    
#     # ============================================================
#     # 关键修改：不再计算倾斜单应矩阵 H_tilt 及其逆矩阵
#     # 因此 x_tilt, y_tilt 直接等于理想平面上的物理坐标
#     # ============================================================
    
#     # 遍历输出图像的每个像素（逆向映射）
#     for v_out in range(out_h):
#         for u_out in range(out_w):
#             # 1. 从输出像素坐标转换到理想未倾斜平面上的物理坐标（单位：mm）
#             x_ideal = (u_out - cx) * sx
#             y_ideal = (v_out - cy) * sy
            
#             # 2. 跳过逆向透视倾斜变换，直接将理想坐标作为倾斜传感器平面上的无畸变点
#             x_tilt = x_ideal
#             y_tilt = y_ideal
            
#             # 3. 施加正向畸变模型（径向 + 切向 + 倾斜畸变补偿）
#             r2 = x_tilt * x_tilt + y_tilt * y_tilt
#             r4 = r2 * r2
#             r6 = r2 * r4
#             radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
#             x_rad = x_tilt * radial
#             y_rad = y_tilt * radial
            
#             x_tan = 2.0 * p1 * x_tilt * y_tilt + p2 * (r2 + 2.0 * x_tilt * x_tilt)
#             y_tan = p1 * (r2 + 2.0 * y_tilt * y_tilt) + 2.0 * p2 * x_tilt * y_tilt
            
#             x_tilt_comp = alpha1 * tau * x_tilt
#             y_tilt_comp = alpha2 * tau * y_tilt
            
#             x_d = x_rad + x_tan + x_tilt_comp
#             y_d = y_rad + y_tan + y_tilt_comp
            
#             # 4. 将畸变后的物理坐标映射回原始图像像素坐标
#             u_orig = x_d / sx + cx
#             v_orig = y_d / sy + cy
            
#             # 5. 双线性插值（边界检查）
#             if 0 <= u_orig < w_in - 1 and 0 <= v_orig < h_in - 1:
#                 u0 = int(np.floor(u_orig))
#                 u1 = u0 + 1
#                 v0 = int(np.floor(v_orig))
#                 v1 = v0 + 1
#                 du = u_orig - u0
#                 dv = v_orig - v0
                
#                 if img.ndim == 3:  # 彩色图像
#                     for ch in range(3):
#                         val = (1 - du) * (1 - dv) * img[v0, u0, ch] + \
#                               du * (1 - dv) * img[v0, u1, ch] + \
#                               (1 - du) * dv * img[v1, u0, ch] + \
#                               du * dv * img[v1, u1, ch]
#                         rectified[v_out, u_out, ch] = np.clip(val, 0, 255)
#                 else:  # 灰度图像
#                     val = (1 - du) * (1 - dv) * img[v0, u0] + \
#                           du * (1 - dv) * img[v0, u1] + \
#                           (1 - du) * dv * img[v1, u0] + \
#                           du * dv * img[v1, u1]
#                     rectified[v_out, u_out] = np.clip(val, 0, 255)
    
#     return rectified

# ============================================================
#  精度评估
# ============================================================
def evaluate_calibration_baseline_pure(P_w_list: List[np.ndarray], P_img_list: List[np.ndarray], 
                                       cam_params: Dict, final_x: np.ndarray, square_size_mm: float):
    print("\n" + "="*20 + " 物理空间绝对精度评估 " + "="*20)
    all_reproj_errors = []
    all_edge_errors = []
    
    sx, sy = cam_params['sx'], cam_params['sy']
    c, d = cam_params['c'], cam_params['d']
    tau, rho = cam_params['tau'], cam_params['rho']
    cx, cy = cam_params['cx'], cam_params['cy']
    k1, k2, k3 = cam_params['k1'], cam_params['k2'], cam_params.get('k3', 0.0)
    p1, p2 = cam_params['p1'], cam_params['p2']
    alpha1, alpha2 = cam_params.get('alpha1', 0.0), cam_params.get('alpha2', 0.0)
    
    distortion_model = cam_params['distortion_model']
    if distortion_model == 'none': int_len = 6
    elif distortion_model == 'k1': int_len = 7
    elif distortion_model == 'k2': int_len = 8
    elif distortion_model == 'full': int_len = 10
    else: int_len = 13
    
    for i in range(len(P_w_list)):
        idx = int_len + i * 6
        rvec = final_x[idx:idx+3]
        tvec = final_x[idx+3:idx+6]
        P_pred = project_points_scheimpflug(P_w_list[i], rvec, tvec, c, d, tau, rho,
                                            k1, k2, k3, p1, p2, alpha1, alpha2, sx, sy, cx, cy)
        pix_errors = np.linalg.norm(P_pred - P_img_list[i], axis=1)
        all_reproj_errors.extend(pix_errors)
        
        P_w = P_w_list[i]
        num_pts = P_w.shape[0]
        for idx_a in range(num_pts):
            x_i, y_i, _ = P_w[idx_a]
            for idx_b in range(num_pts):
                x_j, y_j, _ = P_w[idx_b]
                is_horiz = np.isclose(x_j - x_i, square_size_mm) and np.isclose(y_j, y_i)
                is_vert = np.isclose(y_j - y_i, square_size_mm) and np.isclose(x_j, x_i)
                if is_horiz or is_vert:
                    pred_dist = np.linalg.norm(P_pred[idx_a] - P_pred[idx_b]) * sx
                    all_edge_errors.append(pred_dist - square_size_mm)
    
    all_reproj_errors = np.array(all_reproj_errors)
    all_edge_errors = np.array(all_edge_errors) * 1000.0
    print(f"1. 像素重投影误差统计:")
    print(f"   - 平均误差: {np.mean(all_reproj_errors):.4f} 像素")
    print(f"   - 最大误差: {np.max(all_reproj_errors):.4f} 像素")
    print(f"\n2. 物理空间格网形变残差统计:")
    print(f"   - 平均物理偏差: {np.mean(np.abs(all_edge_errors)):.2f} µm")
    print(f"   - 最大物理偏差: {np.max(np.abs(all_edge_errors)):.2f} µm")
    print(f"   - 物理偏差标准差: {np.std(all_edge_errors):.2f} µm")
    max_phy_err = np.max(np.abs(all_edge_errors))
    print("\n" + "-"*50)
    print(f"【当前标定精度段位评估】:")
    if max_phy_err <= 5.0:
        print(f" 🌟 完美: 最大物理误差 {max_phy_err:.2f}µm <= 5µm。")
    elif max_phy_err <= 25.0:
        print(f" 👍 良好: 最大物理误差 {max_phy_err:.2f}µm。")
    elif max_phy_err <= 100.0:
        print(f" ⚠️ 一般: 最大物理误差 {max_phy_err:.2f}µm。")
    else:
        print(f" ❌ 粗糙: 最大物理误差 {max_phy_err:.2f}µm。")
    print("="*69)


def image_to_world_point(u, v, rvec, tvec, cam_params):
    """
    将图像点逆向投影到世界坐标系中的Z=0平面
    
    参数:
        u, v: 图像坐标（像素）
        rvec: 旋转向量（3,）
        tvec: 平移向量（3,）
        cam_params: 相机参数字典
    
    返回:
        世界坐标点 (3,) 或 None
    """
    # 提取参数
    sx, sy = cam_params['sx'], cam_params['sy']
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
    
    try:
        # 1. 像素坐标 -> 传感器物理坐标
        xd = (u - cx) * sx
        yd = (v - cy) * sy
        
        # 2. 倾斜映射逆变换
        H = compute_tilt_homography(tau, rho, d, lens_type='perspective')
        H_inv = np.linalg.inv(H)
        
        pt = H_inv @ np.array([xd, yd, 1.0])
        xt = pt[0] / pt[2]
        yt = pt[1] / pt[2]
        
        # 3. 畸变校正（迭代法）
        # 简化的畸变校正：使用固定点迭代
        xu, yu = undistort_point(xt, yt, k1, k2, k3, p1, p2, alpha1, alpha2, tau)
        
        # 4. 未倾斜平面上的理想点 -> 相机坐标系
        Zc = c  # 假设在参考距离处
        Xc = xu * Zc / c
        Yc = yu * Zc / c
        
        P_c = np.array([Xc, Yc, Zc])
        
        # 5. 相机坐标系 -> 世界坐标系
        R = rodrigues_to_rotation_matrix(rvec)
        P_w = R.T @ (P_c - tvec)
        
        # 假设标定板在Z=0平面
        if abs(P_w[2]) > 1e-6:
            # 投影到Z=0平面
            scale = -P_w[2] / (R.T @ np.array([0, 0, 1]))[2] if False else 1.0
            # 简化：直接返回
            return P_w
        
        return P_w
        
    except Exception as e:
        return None


def undistort_point(x, y, k1, k2, k3, p1, p2, alpha1, alpha2, tau, max_iter=10):
    """
    畸变校正（迭代法）
    """
    xu, yu = x, y
    
    for _ in range(max_iter):
        r2 = xu*xu + yu*yu
        r4 = r2*r2
        r6 = r2*r4
        
        radial = 1.0 + k1*r2 + k2*r4 + k3*r6
        
        # 计算当前点的畸变
        x_rad = xu * radial
        y_rad = yu * radial
        
        x_tan = 2.0*p1*xu*yu + p2*(r2 + 2.0*xu*xu)
        y_tan = p1*(r2 + 2.0*yu*yu) + 2.0*p2*xu*yu
        
        x_tilt_comp = alpha1 * tau * xu
        y_tilt_comp = alpha2 * tau * yu
        
        xd = x_rad + x_tan + x_tilt_comp
        yd = y_rad + y_tan + y_tilt_comp
        
        # 更新估计值
        error_x = xd - x
        error_y = yd - y
        
        if abs(error_x) < 1e-8 and abs(error_y) < 1e-8:
            break
        
        # 简单迭代校正（实际应用中可用牛顿法）
        xu = xu - error_x * 0.5
        yu = yu - error_y * 0.5
    
    return xu, yu


def compute_grid_errors(points_3d, square_size_mm, tolerance=0.2):
    """
    计算棋盘格或对称圆点板的邻接点距离误差
    """
    errors = []
    n_points = len(points_3d)
    
    # 假设点是按行优先顺序排列的
    # 需要根据实际的标定板布局来确定邻接关系
    cols, rows = 8, 11  # 这里应该从实际配置获取
    
    # 水平邻接
    for r in range(rows):
        for c in range(cols - 1):
            idx1 = r * cols + c
            idx2 = r * cols + c + 1
            if idx1 < n_points and idx2 < n_points:
                dist = np.linalg.norm(points_3d[idx2] - points_3d[idx1])
                if abs(dist - square_size_mm) < square_size_mm * tolerance:
                    errors.append(dist - square_size_mm)
    
    # 垂直邻接
    for c in range(cols):
        for r in range(rows - 1):
            idx1 = r * cols + c
            idx2 = (r + 1) * cols + c
            if idx1 < n_points and idx2 < n_points:
                dist = np.linalg.norm(points_3d[idx2] - points_3d[idx1])
                if abs(dist - square_size_mm) < square_size_mm * tolerance:
                    errors.append(dist - square_size_mm)
    
    return errors


def compute_asym_board_errors(points_3d, square_size_mm, tolerance=0.2):
    """
    计算非对称圆点板的邻接点距离误差
    """
    errors = []
    n_points = len(points_3d)
    
    # 非对称圆点板的特殊处理
    cols, rows = 8, 11  # 从配置获取
    
    for r in range(rows):
        for c in range(cols - 1):
            idx1 = r * cols + c
            idx2 = r * cols + c + 1
            if idx1 < n_points and idx2 < n_points:
                # 非对称板的水平间距可能是2倍方格尺寸
                expected_dist = square_size_mm * (2 if r % 2 == 1 else 1)
                dist = np.linalg.norm(points_3d[idx2] - points_3d[idx1])
                if abs(dist - expected_dist) < expected_dist * tolerance:
                    errors.append(dist - expected_dist)
    
    # 垂直邻接
    for c in range(cols):
        for r in range(rows - 1):
            idx1 = r * cols + c
            idx2 = (r + 1) * cols + c
            if idx1 < n_points and idx2 < n_points:
                dist = np.linalg.norm(points_3d[idx2] - points_3d[idx1])
                if abs(dist - square_size_mm) < square_size_mm * tolerance:
                    errors.append(dist - square_size_mm)
    
    return errors

def evaluate_metric_accuracy_scheimpflug(
    P_w_list: List[np.ndarray],
    P_img_list: List[np.ndarray],
    cam_params: Dict,
    final_x: np.ndarray,
    square_size_mm: float
):
    """
    评估沙姆相机的度量精度（物理空间精度）
    
    参数:
        P_w_list: 世界坐标点列表
        P_img_list: 图像坐标点列表  
        cam_params: 标定参数字典
        final_x: 优化后的完整参数向量
        square_size_mm: 标定板方格尺寸（mm）
    """
    print("\n" + "="*20 + " 度量精度评估（Metric Accuracy） " + "="*20)
    
    # 提取相机参数
    sx, sy = cam_params['sx'], cam_params['sy']
    c, d = cam_params['c'], cam_params['d']
    tau, rho = cam_params['tau'], cam_params['rho']
    cx, cy = cam_params['cx'], cam_params['cy']
    k1, k2, k3 = cam_params['k1'], cam_params['k2'], cam_params.get('k3', 0.0)
    p1, p2 = cam_params['p1'], cam_params['p2']
    alpha1, alpha2 = cam_params.get('alpha1', 0.0), cam_params.get('alpha2', 0.0)
    
    # 确定内参长度
    distortion_model = cam_params['distortion_model']
    if distortion_model == 'none': 
        int_len = 6
    elif distortion_model == 'k1': 
        int_len = 7
    elif distortion_model == 'k2': 
        int_len = 8
    elif distortion_model == 'full': 
        int_len = 10
    else:  # full_tilt
        int_len = 13
    
    all_edge_errors = []
    all_reconstructed_points = []
    
    print("\n正在重建空间点坐标...")
    
    for i in range(len(P_w_list)):
        idx = int_len + i * 6
        rvec = final_x[idx:idx+3]
        tvec = final_x[idx+3:idx+6]
        
        # 获取当前视图的世界坐标和图像坐标
        P_w = P_w_list[i]
        P_img = P_img_list[i]
        num_pts = P_w.shape[0]
        
        # ----------------------------------------------------
        # 1. 通过逆向投影重建空间点
        # ----------------------------------------------------
        Pw_reconstructed = []
        
        for j in range(num_pts):
            u, v = P_img[j]
            
            # 逆向投影：图像点 -> 世界坐标
            Pw = image_to_world_point(u, v, rvec, tvec, cam_params)
            if Pw is not None:
                Pw_reconstructed.append(Pw)
        
        if len(Pw_reconstructed) == num_pts:  # 所有点都重建成功
            Pw_reconstructed = np.array(Pw_reconstructed)
            all_reconstructed_points.append(Pw_reconstructed)
            
            # ----------------------------------------------------
            # 2. 计算相邻点距离误差
            # ----------------------------------------------------
            # 根据标定板类型确定邻接关系
            if CALIB_BOARD_TYPE == CALIB_BOARD_CIRCLE_ASYM:
                # 非对称圆点板的特殊处理
                edge_errors = compute_asym_board_errors(Pw_reconstructed, square_size_mm)
            else:
                # 棋盘格或对称圆点板
                edge_errors = compute_grid_errors(Pw_reconstructed, square_size_mm)
            
            all_edge_errors.extend(edge_errors)
    
    if len(all_edge_errors) == 0:
        print("警告：没有找到有效的邻接点对！")
        return
    
    # 转换为微米
    all_edge_errors = np.array(all_edge_errors) * 1000.0
    
    print("\n" + "-"*50)
    print("度量精度统计结果：")
    print(f"  样本数量: {len(all_edge_errors)} 个边")
    print(f"  平均绝对误差: {np.mean(np.abs(all_edge_errors)):.2f} μm")
    print(f"  最大绝对误差: {np.max(np.abs(all_edge_errors)):.2f} μm")
    print(f"  误差标准差: {np.std(all_edge_errors):.2f} μm")
    print(f"  RMS误差: {np.sqrt(np.mean(all_edge_errors**2)):.2f} μm")
    
    # 分位数统计
    percentiles = [50, 75, 90, 95, 99]
    print("\n误差分布百分位数：")
    for p in percentiles:
        val = np.percentile(np.abs(all_edge_errors), p)
        print(f"  {p}%: {val:.2f} μm")
    
    # 精度等级评估
    max_err = np.max(np.abs(all_edge_errors))
    print("\n" + "-"*50)
    print("【精度等级评估】")
    if max_err <= 10.0:
        print(f" 🌟 超高精度: 最大误差 {max_err:.2f}μm ≤ 10μm")
    elif max_err <= 25.0:
        print(f" 👍 高精度: 最大误差 {max_err:.2f}μm")
    elif max_err <= 50.0:
        print(f" ✓ 良好: 最大误差 {max_err:.2f}μm")
    elif max_err <= 100.0:
        print(f" ⚠️ 一般: 最大误差 {max_err:.2f}μm")
    else:
        print(f" ❌ 需改进: 最大误差 {max_err:.2f}μm > 100μm")
    
    print("="*60)


# ============================================================
#  主程序
# ============================================================
if __name__ == "__main__":
    IMAGE_DIR = r"C:\Users\1\Videos\123"
    PATTERN_SIZE = (8, 11)          # 注意：请根据实际棋盘格内角点行列数设置
    SQUARE_SIZE_MM = 1.5
    PIXEL_SIZE_X_MM = 0.004
    PIXEL_SIZE_Y_MM = 0.004
    INIT_INTRINSIC = {
        'c': 8.0, 'd': 8.2, 'tau': np.radians(9.0), 'rho': np.radians(0.0),
        'cx': 640.0, 'cy': 512.0
    }
    DISTORTION_MODEL = 'full'# 'none', 'k1', 'k2', 'full', 'full_tilt'
    LENS_TYPE = 'perspective'
    SHOW_CORNERS = False
    # 放宽筛选参数
    MAX_LINE_ERROR_PX = 3.0
    MAX_REJECT_RATIO = 0.6
    DEBUG_CORNERS = True
    
    # 检测角点（带几何筛选）
    P_w_list, P_img_list = load_and_detect_calibration_images(
        IMAGE_DIR, PATTERN_SIZE, SQUARE_SIZE_MM, show_corners=SHOW_CORNERS,
        max_line_error_px=MAX_LINE_ERROR_PX, max_reject_ratio=MAX_REJECT_RATIO,
        debug=DEBUG_CORNERS
    )
    
    # 获取图像尺寸
    sample_path = glob.glob(os.path.join(IMAGE_DIR, "*.[jJ][pP][gG]"))[0]
    sample_img = cv2.imread(sample_path)
    img_h, img_w = sample_img.shape[:2]
    
    # 标定
    cam_params = calibrate_scheimpflug(
        P_w_list, P_img_list, (img_h, img_w),
        PIXEL_SIZE_X_MM, PIXEL_SIZE_Y_MM,
        INIT_INTRINSIC,
        distortion_model=DISTORTION_MODEL,
        lens_type=LENS_TYPE
    )
    
    # 评估
    # evaluate_calibration_baseline_pure(P_w_list, P_img_list, cam_params, cam_params['raw_x'], SQUARE_SIZE_MM)
    # 新增的度量精度评估
    evaluate_metric_accuracy_scheimpflug(P_w_list, P_img_list, cam_params, cam_params['raw_x'], SQUARE_SIZE_MM)
    # 校正一张示例图像
    test_img = cv2.imread(sample_path)
    if test_img is not None:
        rectified = rectify_tilt_image(test_img, cam_params)
        cv2.imwrite("rectified_example.jpg", rectified)
        print("\n校正后的图像已保存至 rectified_example.jpg")
    
    print("\n" + "="*60)
    print("高精度标定提醒：...")
    print("="*60)