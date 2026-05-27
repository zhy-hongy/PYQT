import numpy as np
import cv2
import glob
import os
from scipy.optimize import least_squares

# ==================== 配置参数 ====================
CONFIG = {
    "chessboard_size": (8, 11),  # 棋盘格内角点数 (横向, 纵向)
    "square_size": 30.0,         # 单个方格物理边长 (mm)
    "images_path": "saved_images/*.jpg", 
    "show_corners": True,  # 👈 新增：设置为 True 显示角点，False 则静默跳过

    "tilt_angle": 15.0,  # 沙姆倾斜角度 (degrees)
    
}

def scheimpflug_advanced_calibration():
    print("=" * 60)
    print("🚀 开始 [沙姆倾斜 + 高阶非线性畸变] 全局束调整标定...")
    print("=" * 60)
    
    # 1. 准备棋盘格世界坐标 (Zw=0)
    board_sz = CONFIG['chessboard_size']
    sq_sz = CONFIG['square_size']
    objp = np.zeros((board_sz[0] * board_sz[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_sz[0], 0:board_sz[1]].T.reshape(-1, 2) * sq_sz
    world_points_2d = objp[:, :2]
    
    # 2. 提取图像角点
    images = glob.glob(CONFIG['images_path'])
    all_wld_pts = []
    all_img_pts = []
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_sz, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            all_wld_pts.append(world_points_2d)
            all_img_pts.append(corners2.reshape(-1, 2))
            
            # 👇 新增：可视化角点检测结果
            if CONFIG["show_corners"]:
                # 在原图上绘制彩色角点与连接线
                cv2.drawChessboardCorners(img, board_sz, corners2, ret)
                cv2.namedWindow('Calibration Corners', cv2.WINDOW_NORMAL)
                cv2.imshow('Calibration Corners', img)
                
                print(f"正在预览图片: {os.path.basename(fname)}，按任意键继续...")
                # 按任意键继续显示下一张，或等待500ms自动继续
                cv2.waitKey(500) if i < len(images) - 1 else cv2.waitKey(0)
        else:
            print(f"警告：图片 {os.path.basename(fname)} 中未找到棋盘格角点，跳过")   
                
    cv2.destroyAllWindows()
         
    num_images = len(all_img_pts)
    print(f"成功加载并提取了 {num_images} 张图片的角点。")
    
    # 3. 使用普通针孔模型获取初值 (焦距 f, 主点 cx, cy, 以及畸变初值)
    objpoints_3d = [objp for _ in range(num_images)]
    imgpoints_3d = [p.reshape(-1, 1, 2) for p in all_img_pts]
    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints_3d, imgpoints_3d, gray.shape[::-1], None, None)
    
    f_init = (mtx[0, 0] + mtx[1, 1]) / 2.0
    cx_init, cy_init = mtx[0, 2], mtx[1, 2]
    
    # 提取 OpenCV 估算出的畸变初值 [k1, k2, p1, p2, k3]
    k1_init, k2_init, p1_init, p2_init = dist.ravel()[:4]
    
    # 4. 估算每张图的初始外参 (带入 15° 沙姆角先验)
    tau_init = np.radians(CONFIG['tilt_angle'])
    phi_init = np.radians(90.0) 
    
    M_tilt_init = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-np.tan(tau_init)*np.cos(phi_init)/f_init, -np.tan(tau_init)*np.sin(phi_init)/f_init, 1.0]
    ])
    
    init_ext_params = []
    for i in range(num_images):
        H_i, _ = cv2.findHomography(all_wld_pts[i], all_img_pts[i])
        G_i = np.dot(np.linalg.inv(mtx), H_i)
        K_i = np.dot(np.linalg.inv(M_tilt_init), G_i)
        s_i = (np.linalg.norm(K_i[:, 0]) + np.linalg.norm(K_i[:, 1])) / 2.0
        r1, r2 = K_i[:, 0] / s_i, K_i[:, 1] / s_i
        T_i = K_i[:, 2] / s_i
        r3 = np.cross(r1, r2)
        U, _, Vt = np.linalg.svd(np.column_stack((r1, r2, r3)))
        R_i = np.dot(U, Vt)
        rvec_i, _ = cv2.Rodrigues(R_i)
        init_ext_params.extend(np.hstack((rvec_i.flatten(), T_i.flatten())))

    # 5. 组装全局多图优化向量
    # 打包格式: [f, tau, phi, cx, cy, k1, k2, p1, p2,  rvec1, T1,  rvec2, T2, ...]
    initial_all_params = np.hstack(([f_init, tau_init, phi_init, cx_init, cy_init, k1_init, k2_init, p1_init, p2_init], init_ext_params))
    
    # 6. 设置物理边界限制 (沙姆角 tau 严格限制在 14.5° ~ 15.5°)
    tau_min, tau_max = np.radians(14.5), np.radians(15.5)
    low_b = [-np.inf, tau_min, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf] + [-np.inf] * (6 * num_images)
    up_b  = [ np.inf, tau_max,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf] + [ np.inf] * (6 * num_images)

    # 7. 包含高阶畸变模型的残差函数
    def loss_func(params):
        f, tau, phi, cx, cy, k1, k2, p1, p2 = params[0:9]
        
        # 倾斜投影矩阵
        M_t = np.array([
            [1.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0], 
            [-np.tan(tau)*np.cos(phi)/f, -np.tan(tau)*np.sin(phi)/f, 1.0]
        ])
        
        residuals = []
        for i in range(num_images):
            idx = 9 + i * 6
            rvec = params[idx : idx+3].reshape(3, 1)
            T_vec = params[idx+3 : idx+6].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            
            # 外参矩阵
            M_e = np.column_stack((R[:, 0], R[:, 1], T_vec))
            
            # 基础投影变换矩阵 (不含内参缩放)
            H_proj = np.dot(M_t, M_e)
            
            img_pts = all_img_pts[i]
            wld_pts = all_wld_pts[i]
            
            for j, pt in enumerate(wld_pts):
                # 阶段 1: 投影到理想的倾斜归一化传感器平面
                p_homo = np.array([pt[0], pt[1], 1.0])
                p_norm = np.dot(H_proj, p_homo)
                x = p_norm[0] / p_norm[2]
                y = p_norm[1] / p_norm[2]
                
                # 阶段 2: 施加径向与切向畸变公式 🌀
                r2 = x**2 + y**2
                radial = 1.0 + k1 * r2 + k2 * r2**2
                dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x**2)
                dy = p1 * (r2 + 2.0 * y**2) + 2.0 * p2 * x * y
                
                x_distorted = x * radial + dx
                y_distorted = y * radial + dy
                
                # 阶段 3: 转换至最终像素坐标
                u_proj = f * x_distorted + cx
                v_proj = f * y_distorted + cy
                
                # 计算像素残差
                residuals.append(u_proj - img_pts[j, 0])
                residuals.append(v_proj - img_pts[j, 1])
                
        return np.array(residuals)

    # 8. 执行非线性最小二乘优化 (开启 verbose=2) 📈
    print("开始迭代计算，请观察下方日志中的残差变化...")
    res = least_squares(loss_func, initial_all_params, bounds=(low_b, up_b), method='trf', verbose=2)
    
    # 9. 输出最优参数
    f_opt, tau_opt, phi_opt, cx_opt, cy_opt, k1_opt, k2_opt, p1_opt, p2_opt = res.x[0:9]
    print("\n" + "="*50)
    print("🎉 标定完成！最终核心参数：")
    print("="*50)
    print(f"📷 优化后的沙姆倾斜角 (Tau): {np.degrees(tau_opt):.4f} 度")
    print(f"📐 优化后的倾斜方向角 (Phi): {np.degrees(phi_opt):.4f} 度")
    print(f"🔍 优化后的相机焦距 (f): {f_opt:.2f} 像素")
    print(f"🌀 优化后的畸变系数: k1={k1_opt:.5f}, k2={k2_opt:.5f}, p1={p1_opt:.5f}, p2={p2_opt:.5f}")

    # === 新增：计算并打印每张图片各自的平均像素误差 ===
    print("\n" + "="*50)
    print("📸 每张图片的重投影误差 (RMS Error) 分析：")
    print("="*50)
    
    # 提取最终优化后的共享参数
    f_opt, tau_opt, phi_opt, cx_opt, cy_opt, k1_opt, k2_opt, p1_opt, p2_opt = res.x[0:9]
    A_opt = np.array([[f_opt, 0, cx_opt], [0, f_opt, cy_opt], [0, 0, 1]])
    M_t_opt = np.array([
        [1.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0], 
        [-np.tan(tau_opt)*np.cos(phi_opt)/f_opt, -np.tan(tau_opt)*np.sin(phi_opt)/f_opt, 1.0]
    ])
    H_proj_opt = M_t_opt # 基础投影变换
    
    # 遍历每张图计算误差
    for i in range(num_images):
        idx = 9 + i * 6
        rvec = res.x[idx : idx+3].reshape(3, 1)
        T_vec = res.x[idx+3 : idx+6].reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec)
        M_e = np.column_stack((R[:, 0], R[:, 1], T_vec))
        
        # 组合当前图的投影
        H_curr = np.dot(H_proj_opt, M_e)
        
        img_pts = all_img_pts[i]
        wld_pts = all_wld_pts[i]
        
        image_squared_errors = []
        
        for j, pt in enumerate(wld_pts):
            p_homo = np.array([pt[0], pt[1], 1.0])
            p_norm = np.dot(H_curr, p_homo)
            x = p_norm[0] / p_norm[2]
            y = p_norm[1] / p_norm[2]
            
            # 应用畸变
            r2 = x**2 + y**2
            radial = 1.0 + k1_opt * r2 + k2_opt * r2**2
            dx = 2.0 * p1_opt * x * y + p2_opt * (r2 + 2.0 * x**2)
            dy = p1_opt * (r2 + 2.0 * y**2) + 2.0 * p2_opt * x * y
            
            x_distorted = x * radial + dx
            y_distorted = y * radial + dy
            
            # 像素坐标
            u_proj = f_opt * x_distorted + cx_opt
            v_proj = f_opt * y_distorted + cy_opt
            
            # 计算欧氏距离平方
            error_sq = (u_proj - img_pts[j, 0])**2 + (v_proj - img_pts[j, 1])**2
            image_squared_errors.append(error_sq)
            
        # 计算当前图的均方根误差 (RMS)
        rms_i = np.sqrt(np.mean(image_squared_errors))
        img_name = os.path.basename(images[i])
        print(f"图片 [{i:02d}] -> {img_name:<25} 平均误差: {rms_i:.4f} 像素")
if __name__ == "__main__":
    scheimpflug_advanced_calibration()