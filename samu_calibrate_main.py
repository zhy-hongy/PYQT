import sys
import os
import numpy as np
from samu_calibrate_copy import run_steger_calibration_v2, load_and_detect_calibration_images, rectify_tilt_image, forward_project_steger, run_steger_calibration_fixed_d
import cv2
import glob

if __name__ == "__main__":
    IMAGE_DIRECTORY = "C:\\Users\\1\\Pictures\\Camera Roll"
    CHESSBOARD_PATTERN = (8, 11)
    SQUARE_SIZE_MM = 5.0

    pixel_size_x = 0.004  # mm/pixel
    pixel_size_y = 0.004
    init_intrinsic_guess = {
        'c': 8.0,
        'd': 8.0,
        'tau': np.radians(15.0),
        'rho': np.radians(0.0),
        'cx': 640.0,
        'cy': 512.0
    }

    DISTORTION_MODEL = 'full'   # 'none', 'k1', 'k2', 'full'

    try:
        P_w_list, P_img_list = load_and_detect_calibration_images(
            image_dir=IMAGE_DIRECTORY,
            pattern_size=CHESSBOARD_PATTERN,
            square_size_mm=SQUARE_SIZE_MM
        )

        sample_img = cv2.imread(IMAGE_DIRECTORY + '/' + os.listdir(IMAGE_DIRECTORY)[0])
        if sample_img is not None:
            img_h, img_w = sample_img.shape[:2]
        else:
            img_w, img_h = 2000, 2000

        
        final_optimized_vector = run_steger_calibration_v2(
            P_w_list=P_w_list,
            P_img_list=P_img_list,
            init_intrinsic=init_intrinsic_guess,
            sx=pixel_size_x,
            sy=pixel_size_y,
            image_shape=(img_h, img_w),
            lens_type='perspective',
            distortion_model=DISTORTION_MODEL
        )

        print("\n🎉 全闭环优化计算完成。")

        # ---------- 根据 DISTORTION_MODEL 动态解包（不含 d） ----------
        if DISTORTION_MODEL == 'none':
            # 参数顺序: [c, tau, rho, cx, cy]
            c, tau, rho, cx, cy = final_optimized_vector[0:5]
            d = c
            k1 = k2 = p1 = p2 = 0.0
            intrinsic_len = 5
        elif DISTORTION_MODEL == 'k1':
            # 参数顺序: [c, tau, rho, cx, cy, k1]
            c, tau, rho, cx, cy, k1 = final_optimized_vector[0:6]
            d = c
            k2 = p1 = p2 = 0.0
            intrinsic_len = 6
        elif DISTORTION_MODEL == 'k2':
            # 参数顺序: [c, tau, rho, cx, cy, k1, k2]
            c, tau, rho, cx, cy, k1, k2 = final_optimized_vector[0:7]
            d = c
            p1 = p2 = 0.0
            intrinsic_len = 7
        else:  # 'full'
            # 参数顺序: [c, tau, rho, cx, cy, k1, k2, p1, p2]
            c, tau, rho, cx, cy, k1, k2, p1, p2 = final_optimized_vector[0:9]
            d = c
            intrinsic_len = 9

        opt_tau_deg = np.degrees(tau)
        opt_rho_deg = np.degrees(rho)

        print("\n[工程应用] 标定内参已成功固化，可用于后续 5μm 高精度测量系统！")
        print(f"  --> 优化后焦距 c: {c:.3f} mm")
        print(f"  --> 像平面距离 d 强制等于 c: {d:.3f} mm")
        print(f"  --> 优化后沙姆角 tau: {opt_tau_deg:.3f} 度")
        print(f"  --> 优化后倾斜轴方向 rho: {opt_rho_deg:.3f} 度")
        print(f"  --> 优化后像中心点 X: {cx:.3f} 像素")
        print(f"  --> 优化后像中心点 Y: {cy:.3f} 像素")
        print(f"  --> 优化后畸变系数 k1: {k1:.6f}")
        if DISTORTION_MODEL in ['k2', 'full']:
            print(f"  --> 优化后畸变系数 k2: {k2:.6f}")
        if DISTORTION_MODEL == 'full':
            print(f"  --> 优化后畸变系数 p1: {p1:.6f}")
            print(f"  --> 优化后畸变系数 p2: {p2:.6f}")

        # ---------- 计算重投影误差（使用固定 d=c 的版本） ----------
        # 我们可以直接利用已有的 calibration_residuals_fixed_d 函数来累计残差
        # 但需要从 final_optimized_vector 计算残差，简便起见，写一个专用函数
        def compute_reprojection_error_fixed_d(params_vector, P_w_list, P_img_list, sx, sy,
                                               lens_type, distortion_model):
            # 解包内参（与残差函数一致）
            if distortion_model == 'none':
                c, tau, rho, cx, cy = params_vector[0:5]
                d = c
                k1 = k2 = p1 = p2 = 0.0
                int_len = 5
            elif distortion_model == 'k1':
                c, tau, rho, cx, cy, k1 = params_vector[0:6]
                d = c
                k2 = p1 = p2 = 0.0
                int_len = 6
            elif distortion_model == 'k2':
                c, tau, rho, cx, cy, k1, k2 = params_vector[0:7]
                d = c
                p1 = p2 = 0.0
                int_len = 7
            else:  # 'full'
                c, tau, rho, cx, cy, k1, k2, p1, p2 = params_vector[0:9]
                d = c
                int_len = 9

            ks = [k1, k2]
            ps = [p1, p2]
            total_err_sq = 0.0
            total_pts = 0
            num_views = len(P_w_list)
            for i in range(num_views):
                idx = int_len + i * 6
                rvec = params_vector[idx:idx+3]
                tvec = params_vector[idx+3:idx+6]
                P_w = P_w_list[i]
                P_img_obs = P_img_list[i]
                P_img_pred = forward_project_steger(
                    P_w, rvec, tvec, c, d, tau, rho, ks, ps, sx, sy, cx, cy, lens_type
                )
                diff = P_img_pred - P_img_obs
                total_err_sq += np.sum(diff**2)
                total_pts += len(P_img_obs)
            rmse = np.sqrt(total_err_sq / (2 * total_pts))
            return rmse, total_err_sq, total_pts

        rmse, err_sq, n_pts = compute_reprojection_error_fixed_d(
            final_optimized_vector, P_w_list, P_img_list,
            sx=pixel_size_x, sy=pixel_size_y, lens_type='perspective',
            distortion_model=DISTORTION_MODEL
        )

        print(f"\n📊 重投影误差统计：")
        print(f"  总角点数: {n_pts}")
        print(f"  重投影 RMSE: {rmse:.4f} 像素")
        print(f"  残差平方和: {err_sq:.2f}")

        # 构造相机参数字典（d = c）
        cam_params = {
            'c': c, 'd': d, 'tau': tau, 'rho': rho,
            'cx': cx, 'cy': cy,
            'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2
        }

        # 读取并校正第一张 jpg 图片
        jpg_paths = glob.glob(os.path.join(IMAGE_DIRECTORY, "*.jpg")) + \
                    glob.glob(os.path.join(IMAGE_DIRECTORY, "*.JPG"))
        jpg_paths = sorted(jpg_paths)

        if len(jpg_paths) == 0:
            print("警告：目录中没有找到 .jpg 图片，跳过校正步骤。")
        else:
            img_path = jpg_paths[0]
            img = cv2.imread(img_path)
            if img is not None:
                rectified = rectify_tilt_image(img, cam_params, pixel_size_x, pixel_size_y,
                                               output_size=(img.shape[1], img.shape[0]))
                out_filename = f'rectified_{os.path.basename(img_path)}'
                cv2.imwrite(out_filename, rectified)
                print(f"校正图像已保存为 {out_filename} (来源: {os.path.basename(img_path)})")
            else:
                print(f"无法读取图片 {img_path}")

    except Exception as e:
        print(f"\n❌ 程序运行中断，错误原因: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)