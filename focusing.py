import cv2
import numpy as np

# 模拟硬件控制函数（请替换为你实际的GPIO或串口控制代码）
def set_light_status(point_id, is_focus):
    """
    point_id: 1, 2, 3, 4 代表四个点
    is_focus: True 亮绿灯, False 熄灭
    """
    status = "亮绿灯" if is_focus else "熄灭"
    print(f"[硬件控制] 点 {point_id} 状态: {status}")
    # TODO: 在这里加入你的硬件控制代码，例如：
    # if point_id == 1: GPIO.output(LED1_PIN, GPIO.HIGH if is_focus else GPIO.LOW)

def get_focus_score(roi):
    """计算一个图像区域的清晰度分数 (Laplacian 方差法)"""
    if roi.size == 0:
        return 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 计算拉普拉斯方差
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def main():
    cap = cv2.VideoCapture(0) # 打开默认摄像头，如果是工业相机可能需要修改索引或使用SDK
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 获取画面宽高
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 设定对焦判断的阈值 (需要根据实际环境光线和相机分辨率调试)
    FOCUS_THRESHOLD = 150.0 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面")
            break

        h, w, _ = frame.shape
        
        # 定义4个检测区域的坐标 (这里按画面四等分举例，你可以改成标定板4个点的固定像素坐标)
        # 格式: (x, y, width, height)
        regions = [
            (0, 0, w//2, h//2),          # 左上
            (w//2, 0, w//2, h//2),       # 右上
            (0, h//2, w//2, h//2),       # 左下
            (w//2, h//2, w//2, h//2)     # 右下
        ]

        # 遍历4个区域进行对焦检测
        for i, (x, y, rw, rh) in enumerate(regions):
            # 截取感兴趣区域 (ROI)
            roi = frame[y:y+rh, x:x+rw]
            
            # 计算清晰度分数
            score = get_focus_score(roi)
            is_focus = score > FOCUS_THRESHOLD
            
            # 调用硬件控制函数
            set_light_status(i + 1, is_focus)
            
            # 在画面上绘制区域框和状态提示（方便调试）
            color = (0, 255, 0) if is_focus else (0, 0, 255) # 绿色代表对上焦，红色代表没对上
            cv2.rectangle(frame, (x, y), (x+rw, y+rh), color, 2)
            cv2.putText(frame, f"Pt{i+1}: {int(score)}", (x+10, y+30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Focus Detection", frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
