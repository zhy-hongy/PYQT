import cv2
import numpy as np
import glob
from scipy.optimize import least_squares
import os

# =========================================================
# 1. 核心算法区 (沙姆投影模型与优化目标函数)
# =========================================================
def get_tilt_matrix(tau_x, tau_y):
    """根据倾斜角构造像面的三维旋转矩阵"""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(tau_x), -np.sin(tau_x)],
                   [0, np.sin(tau_x), np.cos(tau_x)]])
    Ry = np.array([[np.cos(tau_y), 0, np.sin(tau_y)],
                   [0, 1, 0],
                   [-np.sin(tau_y), 0, np.cos(tau_y)]])
    return Ry @ Rx

def project_scheimpflug(obj_pts, rvec, tvec, K, dist, tau_x, tau_y):
    """沙姆相机的自定义投影过程"""
    K_eye = np.eye(3)
    dist_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_eye, dist)
    dist_pts = dist_pts.reshape(-1, 2)
    
    pts_homo = np.column_stack((dist_pts, np.ones(len(dist_pts))))
    R_tilt = get_tilt_matrix(tau_x, tau_y)
    pts_tilt = pts_homo @ R_tilt.T
    
    x_tilt = pts_tilt[:, 0] / pts_tilt[:, 2]
    y_tilt = pts_tilt[:, 1] / pts_tilt[:, 2]
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u = fx * x_tilt + cx
    v = fy * y_tilt + cy
    return np.column_stack((u, v))

def residuals(params, num_images, obj_pts_list, img_pts_list):
    """计算所有特征点的重投影误差"""
    fx, fy, cx, cy = params[0:4]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = params[4:9] 
    tau_x, tau_y = params[9:11]
    extrinsic_params = params[11:]
    
    error = []
    for i in range(num_images):
        rvec = extrinsic_params[i*6 : i*6+3]
        tvec = extrinsic_params[i*6+3 : i*6+6]
        obj_pts = obj_pts_list[i]
        img_pts = img_pts_list[i].reshape(-1, 2)
        proj_pts = project_scheimpflug(obj_pts, rvec, tvec, K, dist, tau_x, tau_y)
        error.append((proj_pts - img_pts).ravel())
    return np.concatenate(error)

def calibrate_scheimpflug(obj_pts_list, img_pts_list, image_size):
    """沙姆标定主流程"""
    num_images = len(obj_pts_list)
    print("\n[1/2] 正在进行 OpenCV 标准针孔模型初始化...")
    ret, K_init, dist_init, rvecs_init, tvecs_init = cv2.calibrateCamera(
        obj_pts_list, img_pts_list, image_size, None, None
    )
    
    fx, fy, cx, cy = K_init[0,0], K_init[1,1], K_init[0,2], K_init[1,2]
    k1, k2, p1, p2, k3 = dist_init.ravel()
    tau_x_init, tau_y_init = np.radians(-9.3), np.radians(0.0)
    
    params_init = [fx, fy, cx, cy, k1, k2, p1, p2, k3, tau_x_init, tau_y_init]
    for i in range(num_images):
        params_init.extend(rvecs_init[i].ravel())
        params_init.extend(tvecs_init[i].ravel())
    
    params_init = np.array(params_init)
    
    print(f"初始化 RMSE: {ret:.4f} pixels")
    print("[2/2] 正在进行沙姆模型全局非线性优化 (带有参数边界)...")
    
    # 1. 设置初值为您的设计值 (假设 X轴向下，我们先测正的 9.3 度)
    params_init[9] = np.radians(9.3)  # tau_x
    params_init[10] = 0.0             # tau_y
    
    # 2. 构建边界数组
    num_params = len(params_init)
    lower_bounds = np.full(num_params, -np.inf)
    upper_bounds = np.full(num_params, np.inf)
    
    # 3. 强制 tau_x 在设计值附近浮动 (例如 7度 到 11度)
    # 如果您的设计是负角度，请改为 [-11.0, -7.0]
    lower_bounds[9] = np.radians(7.0)
    upper_bounds[9] = np.radians(11.0)
    
    # 4. 执行带边界的 TRF 优化
    res = least_squares(
        residuals, params_init, 
        method='trf',               # 从 'lm' 切换到 'trf' 以支持 bounds
        bounds=(lower_bounds, upper_bounds), 
        args=(num_images, obj_pts_list, img_pts_list),
        ftol=1e-8, xtol=1e-8,
        verbose=1
    )
    
    opt_params = res.x
    K_opt = np.array([[opt_params[0], 0, opt_params[2]], 
                      [0, opt_params[1], opt_params[3]], 
                      [0, 0, 1]])
    dist_opt = opt_params[4:9]
    tau_x_opt, tau_y_opt = opt_params[9], opt_params[10]
    
    final_residuals = residuals(opt_params, num_images, obj_pts_list, img_pts_list)
    rmse_opt = np.sqrt(np.mean(final_residuals**2))
    
    print("\n" + "="*40)
    print("🎯 标定完成 🎯")
    print("="*40)
    print(f"优化后均方根误差 (RMSE): {rmse_opt:.4f} pixels")
    print(f"沙姆倾斜角 tau_x (俯仰): {np.degrees(tau_x_opt):.4f} 度")
    print(f"沙姆倾斜角 tau_y (偏航): {np.degrees(tau_y_opt):.4f} 度")
    print(f"优化后内参矩阵 K:\n{K_opt}")
    print(f"畸变系数 [k1, k2, p1, p2, k3]:\n{dist_opt}")
    
    return K_opt, dist_opt, tau_x_opt, tau_y_opt

# =========================================================
# 2. 数据加载与主程序区
# =========================================================
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # 🛑 TODO 1: 填写标定板参数
    # ---------------------------------------------------------
    # 棋盘格内角点的数量 (列数, 行数) -> 注意是内角点，不是方格数！
    # 例如：一张 12x9 个方格的棋盘格，内角点数量是 (11, 8)
    CHECKERBOARD_SIZE = (11, 8) 
    
    # 棋盘格单个方格的物理真实边长，单位必须统一 (通常用毫米 mm)
    SQUARE_SIZE = 5.0 
    
    # ---------------------------------------------------------
    # 🛑 TODO 2: 填写包含标定图像的文件夹路径
    # ---------------------------------------------------------
    # 支持绝对路径或相对路径，确保路径下全是 .jpg、.png 等格式的标定图
    IMAGE_DIR = r"C:\Users\1\Pictures\Camera Roll" 
    IMAGE_FORMAT = "*.jpg" # 如果您的图是 png，请改为 "*.png"
    
    # ---------------------------------------------------------
    
    # 构建世界坐标系下的 3D 点 (Z=0)
    # 格式如: (0,0,0), (1*SQUARE_SIZE,0,0), (2*SQUARE_SIZE,0,0) ...
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    obj_pts_list = [] # 存储所有图像的 3D 世界坐标
    img_pts_list = [] # 存储所有图像的 2D 像素坐标
    
    # 亚像素角点提取的停止条件：迭代最大次数 30，精度 0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 读取图像路径
    search_path = os.path.join(IMAGE_DIR, IMAGE_FORMAT)
    images = glob.glob(search_path)
    
    if not images:
        print(f"❌ 错误: 在路径 '{search_path}' 下没有找到图片，请检查路径。")
        exit()

    print(f"找到 {len(images)} 张图片，开始提取角点...")
    image_size = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1] # 获取图像宽高 (Width, Height)

        # 提取角点 (粗提取)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

        if ret:
            obj_pts_list.append(objp)
            # 提取亚像素角点 (精提取)，这对高精度三维重建极其重要
            corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_pts_list.append(corners_subpix)
            print(f"✔️ 成功提取: {os.path.basename(fname)}")
        else:
            print(f"⚠️ 提取失败: {os.path.basename(fname)} (未能识别出所有角点)")

    # 检查是否有足够的有效图像进行标定
    if len(obj_pts_list) < 10:
        print("❌ 错误: 成功提取角点的图片数量少于 10 张，标定结果可能不可靠。请检查图片质量或重拍。")
        exit()

    # 开始执行标定流程
    calibrate_scheimpflug(obj_pts_list, img_pts_list, image_size)