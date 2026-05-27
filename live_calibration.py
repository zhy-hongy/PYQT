"""
实时摄像头画面畸变矫正程序
使用之前标定得到的相机内参和畸变系数，实时矫正摄像头画面
"""

import cv2
import numpy as np
import os

# ==================== 配置参数 ====================
# 标定结果文件路径（使用你之前标定得到的文件）
CALIBRATION_FILE = "calibration_data/calibration_result.npz"

# 摄像头ID（通常0是默认摄像头，如果你有多个摄像头可以改成1,2...）
CAMERA_ID = 0

# 是否保存矫正后的图片（按 's' 键保存）
SAVE_PATH = "undistorted_screenshots"

# 显示窗口名称
WINDOW_NAME = "Camera Calibration - Real Time"
# =================================================

def load_calibration_result(filepath):
    """
    加载标定结果
    """
    if not os.path.exists(filepath):
        print(f"错误：找不到标定文件 {filepath}")
        print("请先运行 calibrate.py 完成标定")
        return None, None
    
    data = np.load(filepath)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['distortion_coeffs']
    
    print("成功加载标定结果")
    print(f"相机内参矩阵:\n{camera_matrix}")
    print(f"畸变系数: {dist_coeffs.ravel()}")
    
    return camera_matrix, dist_coeffs

def create_trackbar_callback(value):
    """
    轨迹条回调函数（空函数，只需要存在即可）
    """
    pass

def main():
    # 1. 加载标定结果
    camera_matrix, dist_coeffs = load_calibration_result(CALIBRATION_FILE)
    if camera_matrix is None:
        return
    
    # 2. 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {CAMERA_ID}")
        print("请检查摄像头是否连接正确")
        return
    
    # 获取摄像头分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {width} x {height}")
    
    # 先读取一帧，确保摄像头正常工作
    ret, frame = cap.read()
    if not ret:
        print("错误：无法获取摄像头画面")
        cap.release()
        return
    
    # 3. 创建窗口和轨迹条（在读取第一帧之后）
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    # 选项: 0=原始画面, 1=矫正后画面(保留黑边), 2=矫正后画面(裁剪黑边)
    mode = 0
    cv2.createTrackbar("Mode", WINDOW_NAME, mode, 2, create_trackbar_callback)
    
    # 可选：调整alpha参数 (0=裁剪边缘, 1=保留全部黑边)
    alpha = 1
    cv2.createTrackbar("Alpha", WINDOW_NAME, alpha, 1, create_trackbar_callback)
    
    # 预计算优化后的相机内参矩阵（用于矫正）
    # alpha=1 表示保留所有原始图像像素，会有黑边
    # alpha=0 表示裁剪掉黑边
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), alpha, (width, height)
    )
    
    # 获取ROI裁剪区域
    x, y, w, h = roi
    
    # 创建保存目录
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    frame_count = 0
    
    print("\n" + "=" * 50)
    print("实时矫正程序已启动")
    print("=" * 50)
    print("操作说明:")
    print("  [Mode轨迹条] 0=原始画面  1=矫正(保留黑边)  2=矫正(裁剪黑边)")
    print("  [Alpha轨迹条] 0=裁剪边缘  1=保留黑边")
    print("  [s] 键 - 保存当前矫正后的图片")
    print("  [q] 或 [ESC] 键 - 退出程序")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面")
            break
        
        # 获取当前轨迹条的值
        mode = cv2.getTrackbarPos("Mode", WINDOW_NAME)
        alpha = cv2.getTrackbarPos("Alpha", WINDOW_NAME)
        
        # 根据alpha重新计算新相机矩阵
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (width, height), alpha, (width, height)
        )
        x, y, w, h = roi
        
        if mode == 0:
            # 原始画面
            display_frame = frame.copy()
            status_text = "RAW (Original)"
        else:
            # 矫正画面
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            
            if mode == 2:
                # 裁剪黑边
                if w > 0 and h > 0 and w <= undistorted.shape[1] and h <= undistorted.shape[0]:
                    display_frame = undistorted[y:y+h, x:x+w]
                else:
                    display_frame = undistorted
                status_text = "UNDISTORTED (Cropped)"
            else:
                # 保留黑边
                display_frame = undistorted
                status_text = "UNDISTORTED (With Border)"
        
        # 在画面上添加信息文字
        cv2.putText(display_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示重投影误差（从标定结果中获取）
        cv2.putText(display_frame, f"Reprojection Error: 0.3509 px", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 显示分辨率信息
        h_display, w_display = display_frame.shape[:2]
        cv2.putText(display_frame, f"Size: {w_display} x {h_display}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 显示操作提示
        cv2.putText(display_frame, "S:Save | Q/ESC:Quit", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示Mode和Alpha的当前值
        cv2.putText(display_frame, f"Mode: {mode} (0=Raw,1=Border,2=Crop)", (10, height - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display_frame, f"Alpha: {alpha} (0=Crop,1=Border)", (10, height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 显示画面
        cv2.imshow(WINDOW_NAME, display_frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q 或 ESC
            print("退出程序")
            break
        
        elif key == ord('s'):  # 保存图片
            frame_count += 1
            # 保存矫正后的图片（而不是原始画面）
            if mode != 0:
                # 获取纯矫正图（不带文字）
                undistorted_only = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
                if mode == 2 and w > 0 and h > 0:
                    save_img = undistorted_only[y:y+h, x:x+w]
                else:
                    save_img = undistorted_only
                
                filename = os.path.join(SAVE_PATH, f"undistorted_{frame_count}.jpg")
                cv2.imwrite(filename, save_img)
                print(f"已保存: {filename}")
            else:
                print("当前为原始画面模式，请先切换到矫正模式(Mode=1或2)再保存")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")


def quick_demo():
    """
    简化版演示：只显示原始画面和矫正画面对比
    """
    # 加载标定结果
    camera_matrix, dist_coeffs = load_calibration_result(CALIBRATION_FILE)
    if camera_matrix is None:
        return
    
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 读取一帧获取分辨率
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头画面")
        cap.release()
        return
    
    height, width = frame.shape[:2]
    print(f"摄像头分辨率: {width} x {height}")
    
    # 预计算新相机矩阵
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    x, y, w, h = roi
    
    print("简化版演示 - 按 'q' 退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 矫正
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # 裁剪黑边
        if w > 0 and h > 0 and w <= undistorted.shape[1] and h <= undistorted.shape[0]:
            cropped = undistorted[y:y+h, x:x+w]
        else:
            cropped = undistorted
        
        # 调整大小使并排显示更协调（如果两图高度不同，进行缩放）
        h_frame, w_frame = frame.shape[:2]
        h_crop, w_crop = cropped.shape[:2]
        
        # 如果裁剪后的图像尺寸异常，使用矫正图代替
        if h_crop <= 0 or w_crop <= 0:
            cropped = undistorted
            h_crop, w_crop = cropped.shape[:2]
        
        # 将裁剪后的图片缩放到与原始图片相同的高度
        if h_crop != h_frame:
            scale = h_frame / h_crop
            new_w = int(w_crop * scale)
            cropped = cv2.resize(cropped, (new_w, h_frame))
        
        # 并排显示：左边原始，右边矫正
        combined = np.hstack((frame, cropped))
        
        # 添加标签
        cv2.putText(combined, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(combined, "UNDISTORTED", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Original vs Undistorted", combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("请选择运行模式:")
    print("1. 完整版（带控制选项）")
    print("2. 简化版（并排对比）")
    choice = input("请输入 1 或 2 (默认1): ").strip()
    
    if choice == "2":
        quick_demo()
    else:
        main()