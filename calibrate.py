"""
相机标定程序 (优化版)
支持两种标定板类型：棋盘格 (chessboard) 和对称圆点 (circlegrid)
"""

import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# ==================== ★ 宏：切换标定板类型 ★ ====================
PATTERN_TYPE = "circlegrid"  # "chessboard" 或 "circlegrid"
# ====================================================================

CONFIG = {
    "pattern_type": PATTERN_TYPE,
    "pattern_size": (7, 7),  # (列数, 行数) 
    "square_size": 5.0,       # 实际测量间距 (单位: mm)
    "images_path": "./saved_images/*.jpg",
    "show_detection": True,   
    "save_result": True,
    "result_path": "calibration_data/calibration_result.npz",
}


def _create_blob_detector():
    """ 为圆形标定板创建 SimpleBlobDetector 参数 """
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 50000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.filterByConvexity = True
    params.minConvexity = 0.85
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    return cv2.SimpleBlobDetector_create(params)


def calibrate_camera(config):
    pattern_type = config["pattern_type"]
    pattern_name = "棋盘格" if pattern_type == "chessboard" else "对称圆点阵列"

    print("=" * 50)
    print("开始相机标定...")
    print(f"标定板类型: {pattern_name}")
    print(f"图案尺寸 (cols, rows): {config['pattern_size']}")
    print(f"间距: {config['square_size']} mm")
    print(f"图片路径: {config['images_path']}")
    print("=" * 50)

    pattern_size = config["pattern_size"]
    square_size = config["square_size"]

    # 1. 准备世界坐标系中的点
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  
    imgpoints = []  

    images = glob.glob(config["images_path"])
    if len(images) == 0:
        print(f"错误：在 {config['images_path']} 没有找到图片文件！")
        return None

    print(f"找到 {len(images)} 张图片")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    blob_detector = _create_blob_detector() if pattern_type == "circlegrid" else None

    success_count = 0
    image_shape = None  # 用于记录标准的图像尺寸 (w, h)
    test_img = None
    test_img_name = ""

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"警告：无法读取图片 {fname}，跳过")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # 尺寸一致性校验
        if image_shape is None:
            image_shape = (w, h)
        elif image_shape != (w, h):
            print(f"警告：图片 {os.path.basename(fname)} 尺寸 {w}x{h} 与首张图片 {image_shape} 不符，跳过")
            continue

        # ---- 根据标定板类型选择检测方法 ----
        if pattern_type == "chessboard":
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            if ret:
                # 棋盘格允许且需要进行角点亚像素细化
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        else:
            # 对称圆点：寻找圆心（findCirclesGrid 内部已包含高精度质心定位）
            ret, corners = cv2.findCirclesGrid(
                gray, pattern_size,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                blobDetector=blob_detector
            )

        if ret:
            success_count += 1
            objpoints.append(objp)
            imgpoints.append(corners)
            
            if test_img is None:
                test_img = img.copy()
                test_img_name = fname

            # 可视化检测结果
            if config["show_detection"]:
                cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                win_name = "Detection Result"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.imshow(win_name, img)
                cv2.waitKey(200)  # 适当缩短预览停留时间
        else:
            print(f"警告：图片 {os.path.basename(fname)} 中未检测到{pattern_name}，跳过")

    cv2.destroyAllWindows()

    print(f"成功检测到{pattern_name}的图片数量: {success_count}/{len(images)}")

    if success_count < 5:
        print("错误：有效图片数量不足5张，无法进行标定！建议15-20张。")
        return None

    # 4. 执行相机标定
    print("正在进行相机标定计算...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    # 5. 输出标定结果
    print("\n" + "=" * 50)
    print("标定结果：")
    print("=" * 50)
    print(f"重投影误差 (RMS): {ret:.6f} 像素")
    print("\n相机内参矩阵 (Camera Intrinsic Matrix):")
    print(mtx)
    print("\n畸变系数 (Distortion Coefficients):")
    print(dist.ravel())

    # 6. 验证标定效果并合并显示
    if test_img is not None:
        print("\n" + "=" * 50)
        print(f"验证标定效果 (样本: {os.path.basename(test_img_name)})...")
        print("=" * 50)
        
        w, h = image_shape
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)

        # 裁剪黑边
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            dst = dst[y:y+h_roi, x:x+w_roi]

        # 优化显示：合并到一个 Matplotlib 窗口中，防止多次阻塞弹窗
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Undistorted Image (Cropped)")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

    # 7. 保存标定结果
    if config["save_result"]:
        save_dir = os.path.dirname(config["result_path"])
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.savez(config["result_path"],
                 camera_matrix=mtx,
                 distortion_coeffs=dist,
                 reprojection_error=ret,
                 pattern_type=pattern_type)
        print(f"\n标定结果已保存到: {config['result_path']}")

        txt_path = config["result_path"].replace(".npz", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"标定板类型: {pattern_name}\n")
            f.write(f"重投影误差: {ret:.6f} 像素\n\n")
            f.write(f"相机内参矩阵:\n{str(mtx)}\n\n")
            f.write(f"畸变系数:\n{str(dist.ravel())}\n")
        print(f"文本结果已保存到: {txt_path}")

    return {
        "camera_matrix": mtx,
        "distortion_coeffs": dist,
        "reprojection_error": ret,
        "pattern_type": pattern_type,
    }


if __name__ == "__main__":
    result = calibrate_camera(CONFIG)