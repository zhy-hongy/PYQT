"""
相机标定程序
使用方法：
1. 修改下方的 CONFIG 部分，设置棋盘格参数和图片路径
2. 运行程序，等待标定完成
"""

import cv2
import numpy as np
import glob
import os

# ==================== 配置参数（请根据实际情况修改）====================
CONFIG = {
    # 棋盘格内角点数量（横向格子数-1, 纵向格子数-1）
    # 例如：打印了10x7个方格，内角点就是9x6
    "chessboard_size": (8, 11),  # (width, height) 即 (横向内角点数, 纵向内角点数)
    
    # 单个方格的边长（单位：毫米）
    # 请用尺子实际测量打印出来的棋盘格
    "square_size": 5.0,  # 单位：mm
    
    # 标定图片存放路径（支持通配符）
    # 把这个路径改成你存放照片的文件夹
    "images_path": "C:\\Users\\1\\Pictures\\Camera Roll/*.jpg",
    
    # 是否显示角点检测结果
    "show_corners": True,
    
    # 是否保存标定结果到文件
    "save_result": True,
    
    
    # 标定结果保存路径
    "result_path": "calibration_data/calibration_result.npz",
}
# ====================================================================

def calibrate_camera(config):
    """
    执行相机标定
    """
    print("=" * 50)
    print("开始相机标定...")
    print(f"棋盘格内角点数量: {config['chessboard_size']}")
    print(f"方格边长: {config['square_size']} mm")
    print(f"图片路径: {config['images_path']}")
    print("=" * 50)
    
    # 1. 准备世界坐标系中的点
    # 棋盘格上的每个内角点，对应一个三维坐标 (x, y, 0)
    # 假设棋盘格位于 z=0 平面上
    chessboard_size = config['chessboard_size']
    square_size = config['square_size']
    
    # 创建棋盘格上所有内角点的世界坐标
    # shape: (内角点总数, 3)，每个点格式 (x, y, z=0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    # mgrid 生成网格坐标，然后乘以方格边长得到实际毫米坐标
    # 例如：第0行第0列的点坐标是 (0,0,0)，第0行第1列是 (25,0,0)...
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 
                            0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    
    # 存储所有图片的世界坐标和像素坐标
    objpoints = []  # 世界坐标系中的点（3D）
    imgpoints = []  # 图像平面中的对应点（2D）
    
    # 2. 获取所有标定图片
    images = glob.glob(config['images_path'])
    if len(images) == 0:
        print(f"错误：在 {config['images_path']} 没有找到图片文件！")
        print("请检查图片路径是否正确，图片格式是否为 .jpg")
        return None
    
    print(f"找到 {len(images)} 张图片")
    
    # 3. 逐张图片寻找棋盘格角点
    # 亚像素角点优化的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    success_count = 0
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"警告：无法读取图片 {fname}，跳过")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            success_count += 1
            objpoints.append(objp)
            
            # 亚像素细化：将角点精度提升到亚像素级别
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # 可视化角点检测结果
            if config['show_corners']:
                # 在图像上绘制角点
                cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
                
                # 显示图像（调整窗口大小）
                cv2.namedWindow('Chessboard Corners', cv2.WINDOW_NORMAL)
                cv2.imshow('Chessboard Corners', img)
                # 按任意键继续显示下一张，或等待500ms自动继续
                cv2.waitKey(500) if i < len(images) - 1 else cv2.waitKey(0)
        else:
            print(f"警告：图片 {os.path.basename(fname)} 中未找到棋盘格角点，跳过")
    
    cv2.destroyAllWindows()
    
    print(f"成功检测到棋盘格的图片数量: {success_count}/{len(images)}")
    
    if success_count < 5:
        print("错误：有效图片数量不足5张，无法进行标定！")
        print("请拍摄更多不同角度的棋盘格照片（建议15-20张）")
        return None
    
    # 4. 执行相机标定
    print("正在进行相机标定计算...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # 5. 输出标定结果
    print("\n" + "=" * 50)
    print("标定结果：")
    print("=" * 50)
    print(f"重投影误差: {ret:.6f} 像素")
    print("\n相机内参矩阵 (Camera Intrinsic Matrix):")
    print("[[fx,  0, cx],")
    print(" [ 0, fy, cy],")
    print(" [ 0,  0,  1]]")
    print(mtx)
    print("\n畸变系数 (Distortion Coefficients):")
    print("[k1, k2, p1, p2, k3]")
    print(dist.ravel())
    
    # 6. 验证标定效果：矫正第一张成功检测的图片
    print("\n" + "=" * 50)
    print("验证标定效果...")
    print("=" * 50)
    
    # 找到第一张成功的图片进行验证
    test_img = None
    for fname in images:
        img = cv2.imread(fname)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret_test, _ = cv2.findChessboardCorners(gray, chessboard_size, None)
            if ret_test:
                test_img = img
                test_img_name = fname
                break
    
    if test_img is not None:
        h, w = test_img.shape[:2]
        
        # 获取优化后的相机内参矩阵（用于去除畸变后裁剪边缘黑边）
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # 方法1：使用 undistort 函数
        dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
        
        # 根据 ROI 裁剪边缘黑边
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
        
        # 显示对比结果
        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Image', test_img)
        cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Undistorted Image', dst)
        print(f"验证图片: {os.path.basename(test_img_name)}")
        print("请查看显示窗口，按任意键关闭...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("无法找到有效的验证图片")
    
    # 7. 保存标定结果
    if config['save_result']:
        # 确保保存目录存在
        save_dir = os.path.dirname(config['result_path'])
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存为 npz 文件
        np.savez(config['result_path'], 
                 camera_matrix=mtx, 
                 distortion_coeffs=dist,
                 reprojection_error=ret)
        print(f"\n标定结果已保存到: {config['result_path']}")
        
        # 同时保存为文本文件，方便查看
        txt_path = config['result_path'].replace('.npz', '.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("相机标定结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"重投影误差: {ret:.6f} 像素\n\n")
            f.write("相机内参矩阵:\n")
            f.write(str(mtx) + "\n\n")
            f.write("畸变系数:\n")
            f.write(str(dist.ravel()) + "\n")
        print(f"文本结果已保存到: {txt_path}")
    
    return {
        "camera_matrix": mtx,
        "distortion_coeffs": dist,
        "reprojection_error": ret
    }


def load_calibration_result(filepath):
    """
    加载之前保存的标定结果
    """
    data = np.load(filepath)
    return data['camera_matrix'], data['distortion_coeffs']


if __name__ == "__main__":
    # 执行标定
    result = calibrate_camera(CONFIG)
    
    if result is not None:
        print("\n" + "=" * 50)
        print("标定完成！✓")
        print("=" * 50)
        print("\n提示：")
        print("1. 重投影误差越小越好，通常小于0.5像素是理想结果")
        print("2. 相机内参矩阵中的 fx, fy 应该比较接近，cx, cy 应该接近图像中心")
        print("3. 畸变系数通常很小（接近0），如果数值很大可能需要重新标定")
    else:
        print("\n标定失败！请检查：")
        print("1. 图片路径是否正确")
        print("2. 棋盘格内角点数量设置是否正确")
        print("3. 图片中能否清晰看到完整的棋盘格")
        print("4. 有效图片数量是否足够（建议15-20张）")