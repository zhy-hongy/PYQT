import numpy as np
import cv2
from scipy.optimize import least_squares
import os
import glob
import sys

# ============================================================
# 图像校正（已移除沙姆倾斜，仅剩畸变校正）
# ============================================================
def rectify_tilt_image(img_original, cam_params, sx, sy, output_size=None):
    """
    校正图像畸变（无倾斜模型，仅径向+切向畸变）
    cam_params 需包含: c, cx, cy, k1, k2, p1, p2
    """
    h, w = img_original.shape[:2]
    if output_size is None:
        output_size = (w, h)
    out_w, out_h = output_size

    k1 = cam_params.get('k1', 0.0)
    k2 = cam_params.get('k2', 0.0)
    p1 = cam_params.get('p1', 0.0)
    p2 = cam_params.get('p2', 0.0)
    cx = cam_params['cx']
    cy = cam_params['cy']

    rectified_img = np.zeros((out_h, out_w, 3), dtype=np.uint8) if img_original.ndim == 3 else np.zeros((out_h, out_w), dtype=np.uint8)

    for v_rect in range(out_h):
        for u_rect in range(out_w):
            # 虚拟无畸变物理坐标
            x_u = (u_rect - cx) * sx
            y_u = (v_rect - cy) * sy

            # 正向畸变 (无畸变 -> 畸变)
            ru2 = x_u*x_u + y_u*y_u
            radial = 1.0 + k1*ru2 + k2*ru2*ru2
            x_d = x_u * radial + (2.0*p1*x_u*y_u + p2*(ru2 + 2.0*x_u*x_u))
            y_d = y_u * radial + (p1*(ru2 + 2.0*y_u*y_u) + 2.0*p2*x_u*y_u)

            # 直接使用畸变坐标（无倾斜）
            x_t = x_d
            y_t = y_d

            # 像素映射
            u_orig = x_t / sx + cx
            v_orig = y_t / sy + cy

            if 0 <= u_orig < w-1 and 0 <= v_orig < h-1:
                u0 = int(np.floor(u_orig)); u1 = u0 + 1
                v0 = int(np.floor(v_orig)); v1 = v0 + 1
                du = u_orig - u0; dv = v_orig - v0
                if img_original.ndim == 3:
                    for c in range(3):
                        val = (1-du)*(1-dv)*img_original[v0,u0,c] + du*(1-dv)*img_original[v0,u1,c] + \
                              (1-du)*dv*img_original[v1,u0,c] + du*dv*img_original[v1,u1,c]
                        rectified_img[v_rect, u_rect, c] = np.clip(val, 0, 255).astype(np.uint8)
                else:
                    val = (1-du)*(1-dv)*img_original[v0,u0] + du*(1-dv)*img_original[v0,u1] + \
                          (1-du)*dv*img_original[v1,u0] + du*dv*img_original[v1,u1]
                    rectified_img[v_rect, u_rect] = np.clip(val, 0, 255).astype(np.uint8)
    return rectified_img


# ============================================================
# 棋盘格角点检测（未改动）
# ============================================================
def load_and_detect_calibration_images(image_dir, pattern_size, square_size_mm):
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    image_paths = sorted(list(set(image_paths)))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"❌ 在路径 '{image_dir}' 中没有找到图片！")

    print(f"🔍 找到 {len(image_paths)} 张图片，开始提取角点...")
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    P_w_list, P_img_list = [], []
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ {filename} 无法读取，跳过。")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            corners_subpix = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), subpix_criteria)
            P_w_list.append(objp.copy())
            P_img_list.append(corners_subpix.reshape(-1, 2))
            print(f"  ✓ {filename} : {len(corners)} 个角点")
        else:
            print(f"  ✗ {filename} : 未检测到完整棋盘格")
    cv2.destroyAllWindows()
    print(f"✅ 有效图片: {len(P_w_list)}/{len(image_paths)}")
    if len(P_w_list) < 3:
        raise ValueError("❌ 有效图片不足 3 张，无法标定！")
    return P_w_list, P_img_list


# ============================================================
# 工具函数（未改动，但针孔模型不再调用 compute_tilt_homography）
# ============================================================
def estimate_initial_extrinisics(P_w_list, P_img_list, c_mm, cx_pix, cy_pix, sx_mm, sy_mm):
    fx_pix = c_mm / sx_mm
    fy_pix = c_mm / sy_mm
    camera_matrix = np.array([[fx_pix, 0, cx_pix], [0, fy_pix, cy_pix], [0,0,1]], dtype=np.float64)
    dist_coeffs = np.zeros(4)
    extrinsic_params_init = []
    print(f"正在用 solvePnP 预估 {len(P_w_list)} 个外参初值...")
    for i in range(len(P_w_list)):
        success, rvec, tvec = cv2.solvePnP(P_w_list[i], P_img_list[i], camera_matrix, dist_coeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            rvec, tvec = np.zeros((3,1)), np.array([[0],[0],[1000.0]])
        extrinsic_params_init.extend(rvec.flatten())
        extrinsic_params_init.extend(tvec.flatten())
        print(f"  视场 {i}: tvec=[{tvec[0,0]:.1f}, {tvec[1,0]:.1f}, {tvec[2,0]:.1f}] mm")
    return extrinsic_params_init

def rodrigues_to_rotation_matrix(r_vec):
    theta = np.linalg.norm(r_vec)
    if theta < 1e-6:
        return np.eye(3)
    r_hat = r_vec / theta
    K = np.array([[0, -r_hat[2], r_hat[1]], [r_hat[2], 0, -r_hat[0]], [-r_hat[1], r_hat[0], 0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*np.dot(K, K)

def compute_tilt_homography(tau, rho, d, lens_type='perspective'):
    """保留但不用于针孔模型"""
    c_tau, s_tau = np.cos(tau), np.sin(tau)
    c_rho, s_rho = np.cos(rho), np.sin(rho)
    Rt = np.array([
        [c_rho**2*(1-c_tau)+c_tau, c_rho*s_rho*(1-c_tau),     s_rho*s_tau],
        [c_rho*s_rho*(1-c_tau),    s_rho**2*(1-c_tau)+c_tau, -c_rho*s_tau],
        [-s_rho*s_tau,             c_rho*s_tau,               c_tau]
    ])
    r11,r12,r13 = Rt[0,0],Rt[0,1],Rt[0,2]
    r21,r22,r23 = Rt[1,0],Rt[1,1],Rt[1,2]
    r31,r32,r33 = Rt[2,0],Rt[2,1],Rt[2,2]
    if lens_type == 'telecentric':
        delta = r11*r22 - r12*r21
        H = np.array([[r22/delta, -r12/delta, 0], [-r21/delta, r11/delta, 0], [0,0,1]])
    else:
        H = np.array([
            [(r22*r33-r23*r32)/r33, (r13*r32-r12*r33)/r33, -d*r31],
            [(r23*r31-r21*r33)/r33, (r11*r33-r13*r31)/r33, -d*r32],
            [(r21*r32-r22*r31)/(d*r33), (r12*r31-r11*r32)/(d*r33), 1.0]
        ])
    return H


# ============================================================
# 针孔模型正向投影（已移除 d, tau, rho）
# ============================================================
def forward_project_steger(P_w, rvec, tvec, c, ks, ps, sx, sy, cx, cy, lens_type='perspective'):
    """
    标准针孔投影 + 畸变，无倾斜
    """
    N = P_w.shape[0]
    R = rodrigues_to_rotation_matrix(rvec)
    P_c = np.dot(R, P_w.T).T + tvec
    Xc, Yc, Zc = P_c[:,0], P_c[:,1], P_c[:,2]

    if lens_type == 'telecentric':
        xu = c * Xc
        yu = c * Yc
    else:
        xu = c * (Xc / Zc)
        yu = c * (Yc / Zc)

    ru2 = xu**2 + yu**2
    k1, k2 = ks[0], ks[1]
    p1, p2 = ps[0], ps[1]
    radial = 1.0 + k1*ru2 + k2*ru2*ru2
    xd = xu * radial + (2.0*p1*xu*yu + p2*(ru2 + 2.0*xu*xu))
    yd = yu * radial + (p1*(ru2 + 2.0*yu*yu) + 2.0*p2*xu*yu)

    # 直接使用畸变坐标，无倾斜
    xt, yt = xd, yd

    u = xt / sx + cx
    v = yt / sy + cy
    return np.vstack((u, v)).T


# ============================================================
# 针孔模型残差与优化函数
# ============================================================
def calibration_residuals_pinhole(params, P_w_list, P_img_list, sx, sy, lens_type, distortion_model):
    """
    参数顺序: [c, cx, cy, (k1), (k2), (p1,p2), 外参...]
    """
    if distortion_model == 'none':
        c, cx, cy = params[0:3]
        k1=k2=p1=p2=0.0
        int_len = 3
    elif distortion_model == 'k1':
        c, cx, cy, k1 = params[0:4]
        k2=p1=p2=0.0
        int_len = 4
    elif distortion_model == 'k2':
        c, cx, cy, k1, k2 = params[0:5]
        p1=p2=0.0
        int_len = 5
    else:  # 'full'
        c, cx, cy, k1, k2, p1, p2 = params[0:7]
        int_len = 7

    ks = [k1, k2]
    ps = [p1, p2]
    residuals = []
    num_views = len(P_w_list)

    for i in range(num_views):
        idx = int_len + i*6
        rvec = params[idx:idx+3]
        tvec = params[idx+3:idx+6]
        P_img_pred = forward_project_steger(
            P_w_list[i], rvec, tvec, c, ks, ps, sx, sy, cx, cy, lens_type
        )
        residuals.extend((P_img_pred - P_img_list[i]).flatten())
    return np.array(residuals)


def run_pinhole_calibration(P_w_list, P_img_list, init_intrinsic, sx, sy, image_shape,
                            lens_type='perspective', distortion_model='k1'):
    """
    纯针孔模型标定，内参无 tau, rho, d
    init_intrinsic 字典只需提供 'c', 'cx', 'cy'
    """
    img_h, img_w = image_shape
    c0 = init_intrinsic['c']
    cx0 = init_intrinsic['cx']
    cy0 = init_intrinsic['cy']

    # 根据畸变模型构建内参初值
    if distortion_model == 'none':
        init_int_vals = [c0, cx0, cy0]
        lower_bounds = [0.5, 0, 0]
        upper_bounds = [100.0, img_w-1, img_h-1]
    elif distortion_model == 'k1':
        init_int_vals = [c0, cx0, cy0, 0.0]
        lower_bounds = [0.5, 0, 0, -1.0]
        upper_bounds = [100.0, img_w-1, img_h-1, 1.0]
    elif distortion_model == 'k2':
        init_int_vals = [c0, cx0, cy0, 0.0, 0.0]
        lower_bounds = [0.5, 0, 0, -1.0, -1.0]
        upper_bounds = [100.0, img_w-1, img_h-1, 1.0, 1.0]
    else:  # 'full'
        init_int_vals = [c0, cx0, cy0, 0.0, 0.0, 0.0, 0.0]
        lower_bounds = [0.5, 0, 0, -1.0, -1.0, -0.5, -0.5]
        upper_bounds = [100.0, img_w-1, img_h-1, 1.0, 1.0, 0.5, 0.5]

    # 外参初值
    extrinsic_inits = estimate_initial_extrinisics(
        P_w_list, P_img_list, c_mm=c0, cx_pix=cx0, cy_pix=cy0, sx_mm=sx, sy_mm=sy
    )
    init_params = np.array(init_int_vals + extrinsic_inits)

    # 外参边界
    for _ in range(len(P_w_list)):
        lower_bounds.extend([-np.pi, -np.pi, -np.pi, -np.inf, -np.inf, 10.0])
        upper_bounds.extend([np.pi, np.pi, np.pi, np.inf, np.inf, np.inf])

    # 边界修正
    for i, (low, high) in enumerate(zip(lower_bounds, upper_bounds)):
        if init_params[i] < low:
            init_params[i] = low
        if init_params[i] > high:
            init_params[i] = high

    result = least_squares(
        calibration_residuals_pinhole,
        init_params,
        bounds=(lower_bounds, upper_bounds),
        args=(P_w_list, P_img_list, sx, sy, lens_type, distortion_model),
        method='trf',
        loss='soft_l1',
        f_scale=2.0,
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        max_nfev=2000,
        verbose=2
    )
    return result.x


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    IMAGE_DIRECTORY = "C:\\Users\\1\\Pictures\\Camera Roll"
    CHESSBOARD_PATTERN = (8, 11)
    SQUARE_SIZE_MM = 5.0

    pixel_size_x = 0.004  # mm/pixel
    pixel_size_y = 0.004

    # 针孔模型只需要 c, cx, cy 初值
    init_intrinsic_guess = {
        'c': 8.0,
        'cx': 640.0,
        'cy': 512.0
    }

    DISTORTION_MODEL = 'full'   # 'none', 'k1', 'k2', 'full'

    try:
        P_w_list, P_img_list = load_and_detect_calibration_images(
            IMAGE_DIRECTORY, CHESSBOARD_PATTERN, SQUARE_SIZE_MM
        )

        sample_img = cv2.imread(IMAGE_DIRECTORY + '/' + os.listdir(IMAGE_DIRECTORY)[0])
        if sample_img is not None:
            img_h, img_w = sample_img.shape[:2]
        else:
            img_w, img_h = 2000, 2000

        # ---------- 针孔模型标定 ----------
        final_optimized_vector = run_pinhole_calibration(
            P_w_list, P_img_list,
            init_intrinsic=init_intrinsic_guess,
            sx=pixel_size_x, sy=pixel_size_y,
            image_shape=(img_h, img_w),
            lens_type='perspective',
            distortion_model=DISTORTION_MODEL
        )

        print("\n🎉 针孔模型标定完成。")

        # 解析结果
        if DISTORTION_MODEL == 'none':
            c, cx, cy = final_optimized_vector[0:3]
            k1 = k2 = p1 = p2 = 0.0
        elif DISTORTION_MODEL == 'k1':
            c, cx, cy, k1 = final_optimized_vector[0:4]
            k2 = p1 = p2 = 0.0
        elif DISTORTION_MODEL == 'k2':
            c, cx, cy, k1, k2 = final_optimized_vector[0:5]
            p1 = p2 = 0.0
        else:  # 'full'
            c, cx, cy, k1, k2, p1, p2 = final_optimized_vector[0:7]

        print(f"  --> 焦距 c: {c:.3f} mm")
        print(f"  --> 主点 X: {cx:.3f} 像素, Y: {cy:.3f} 像素")
        print(f"  --> 畸变 k1: {k1:.6f}")
        if DISTORTION_MODEL in ['k2', 'full']:
            print(f"  --> 畸变 k2: {k2:.6f}")
        if DISTORTION_MODEL == 'full':
            print(f"  --> 畸变 p1: {p1:.6f}, p2: {p2:.6f}")

        # 计算重投影误差
        def compute_reprojection_error(params, P_w_list, P_img_list, sx, sy, distortion_model):
            if distortion_model == 'none':
                c, cx, cy = params[0:3]
                k1=k2=p1=p2=0.0
                int_len = 3
            elif distortion_model == 'k1':
                c, cx, cy, k1 = params[0:4]
                k2=p1=p2=0.0
                int_len = 4
            elif distortion_model == 'k2':
                c, cx, cy, k1, k2 = params[0:5]
                p1=p2=0.0
                int_len = 5
            else:
                c, cx, cy, k1, k2, p1, p2 = params[0:7]
                int_len = 7
            total_err_sq = 0.0
            total_pts = 0
            for i in range(len(P_w_list)):
                idx = int_len + i*6
                rvec = params[idx:idx+3]
                tvec = params[idx+3:idx+6]
                P_pred = forward_project_steger(P_w_list[i], rvec, tvec, c,
                                                [k1,k2], [p1,p2], sx, sy, cx, cy, 'perspective')
                diff = P_pred - P_img_list[i]
                total_err_sq += np.sum(diff**2)
                total_pts += len(P_img_list[i])
            rmse = np.sqrt(total_err_sq / (2*total_pts))
            return rmse, total_err_sq, total_pts

        rmse, err_sq, n_pts = compute_reprojection_error(
            final_optimized_vector, P_w_list, P_img_list, pixel_size_x, pixel_size_y, DISTORTION_MODEL
        )
        print(f"\n📊 重投影误差：RMSE = {rmse:.4f} 像素，总点数 = {n_pts}")

        # 图像校正（无倾斜）
        cam_params = {
            'c': c, 'cx': cx, 'cy': cy,
            'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2
        }
        jpg_paths = sorted(glob.glob(os.path.join(IMAGE_DIRECTORY, "*.jpg")) +
                           glob.glob(os.path.join(IMAGE_DIRECTORY, "*.JPG")))
        if jpg_paths:
            img = cv2.imread(jpg_paths[0])
            if img is not None:
                rectified = rectify_tilt_image(img, cam_params, pixel_size_x, pixel_size_y,
                                               output_size=(img.shape[1], img.shape[0]))
                out_filename = f'rectified_{os.path.basename(jpg_paths[0])}'
                cv2.imwrite(out_filename, rectified)
                print(f"校正图像已保存为 {out_filename}")
            else:
                print("无法读取第一张图片")
        else:
            print("未找到 .jpg 图片")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)