import sys
import socket
import struct
import time
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QTextEdit, QVBoxLayout, QHBoxLayout, QWidget, 
                             QLineEdit, QComboBox, QGroupBox, QFormLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt

# ==========================================
# 1. 网络接收核心线程 (针对 UDP 协议定制)
# ==========================================
class UdpImageReceiver(QThread):
    # 定义信号：(sensor_id, bgr_image)
    frame_ready = pyqtSignal(int, np.ndarray)
    log_signal = pyqtSignal(str)

    def __init__(self, port=60000, img_width=1280, img_height=1024):
        super().__init__()
        self.port = port
        self.running = True
        
        # 图像参数配置（需与硬件 Sensor 严格一致）
        self.img_width = img_width
        self.img_height = img_height
        self.frame_size = img_width * img_height  # 假设为 Raw 8 格式 (1字节/像素)
        self.payload_per_packet = 1032
        
        # 计算一帧完整图像理論上需要的 UDP 包数
        self.expected_pkts = (self.frame_size + self.payload_per_packet - 1) // self.payload_per_packet
        
        # 初始化 Sensor 0 和 Sensor 1 的数据缓冲区与包序号计数器
        self.frame_buffers = {
            0: bytearray(self.expected_pkts * self.payload_per_packet),
            1: bytearray(self.expected_pkts * self.payload_per_packet)
        }
        self.received_packet_ids = {0: set(), 1: set()}

    def verify_checksum(self, packet_bytes):
        """ 
        根据协议要求校验：
        chip_id、cmd功能码、pkt_id、datalength、数据字节相加之和，低16位有效
        """
        # 对应 packet_bytes[2:1041]：从第3字节(sensor ID)累加到第1041字节(data1031)
        calc_sum = sum(packet_bytes[2:1041]) & 0xFFFF
        # 协议规定：校验和高字节在前，低字节在后 (大端序)
        recv_sum = struct.unpack(">H", packet_bytes[1041:1043])[0]
        return calc_sum == recv_sum

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 允许端口复用，方便调试
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", self.port))
        except Exception as e:
            self.log_signal.emit(f"绑定端口 {self.port} 失败: {e}")
            return
            
        sock.settimeout(0.5)
        self.log_signal.emit(f"UDP 接收线程已启动，监听端口: {self.port}")

        while self.running:
            try:
                # 协议包长固定 1044 字节
                data, addr = sock.recvfrom(2048)
                if len(data) != 1044:
                    continue
                
                # 1. 验证固定帧头与帧尾 (大端序)
                header = struct.unpack(">H", data[0:2])[0]
                tail = struct.unpack(">H", data[1043:1045])[0]
                if header != 0x55AA or tail != 0xfafa:
                    continue
                
                # 2. 解析协议基础控制字段
                sensor_id = data[2]
                cmd_id = data[3]
                pkt_id = struct.unpack(">H", data[4:6])[0]
                data_len = struct.unpack(">H", data[6:8])[0]
                
                # 处理下位机应答的握手帧 (功能码 0x01)
                if cmd_id == 0x01:
                    self.log_signal.emit(f"成功收到来自 Sensor {sensor_id} ({addr[0]}) 的握手应答帧！")
                    continue
                
                # 处理图像数据上行帧 (功能码 0x00)
                if cmd_id == 0x00 and data_len == 1032:
                    # 3. 协议校验和验证
                    if not self.verify_checksum(data):
                        # 如果在高速传输中频繁打印会阻塞主线程，可根据需要关闭此日志
                        continue
                    
                    # 4. 防止数据包越界导致的崩溃
                    if pkt_id >= self.expected_pkts:
                        continue
                    
                    # 5. 拼包装入缓冲区
                    start_idx = pkt_id * self.payload_per_packet
                    end_idx = start_idx + self.payload_per_packet
                    self.frame_buffers[sensor_id][start_idx:end_idx] = data[8:1040]
                    self.received_packet_ids[sensor_id].add(pkt_id)
                    
                    # 6. 当收齐了一帧所有的包时，进行图像重构
                    if len(self.received_packet_ids[sensor_id]) == self.expected_pkts:
                        # 提取有效的一帧原始单通道数据
                        raw_data = np.frombuffer(self.frame_buffers[sensor_id][:self.frame_size], dtype=np.uint8)
                        # 变换为二维图像矩阵
                        frame_gray = raw_data.reshape((self.img_height, self.img_width))
                        
                        # 【重要提示】如果你的 FPGA 发出的是 Bayer 格式的彩色 Raw 数据，
                        # 请根据实际排布选择下行转换函数，例如:
                        # frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_BAYER_GB2BGR)
                        # 这里暂按标准灰度图转 BGR 处理，以确保代码直接可用：
                        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                        
                        # 异步投递给主线程 UI
                        self.frame_ready.emit(sensor_id, frame_bgr)
                        
                        # 清空当前计数器，迎接下一帧
                        self.received_packet_ids[sensor_id].clear()
                        
            except socket.timeout:
                continue
            except Exception as e:
                self.log_signal.emit(f"接收异常: {str(e)}")
                
        sock.close()

    def send_handshake(self, target_ip):
        """ 根据1.1节规范，向下位机异步发送 UDP 握手包 """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # 协议构成: 帧头0xaa55(2B), chip_id 0x00(1B), cmd 0x01(1B), pkt_id 0(2B), 
            #          data_len 16(2B), data 16B(0x00~0x0f), checksum 0x0000(2B), tail 0xfafa(2B)
            handshake_data = bytes(range(16))
            packet = struct.pack(">H B B H H 16s H H", 
                                 0xAA55, 0x00, 0x01, 0, 16, 
                                 handshake_data, 0x0000, 0xfafa)
            sock.sendto(packet, (target_ip, self.port))
        except Exception as e:
            self.log_signal.emit(f"发送握手指令失败: {e}")
        finally:
            sock.close()

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# 2. 相机标定算法业务类
# ==========================================
class CameraCalibrator:
    def __init__(self):
        # 默认棋盘格参数（内角点行列数：如 11x8）
        self.grid_rows = 11
        self.grid_cols = 8
        self.square_size = 20.0  # 单个方格物理尺寸 (mm)
        
        self.obj_points = []     # 世界坐标系中的三维点
        self.img_points = []     # 图像像素坐标系中的二维点
        self.img_shape = None
        self.reset_calib_data()

    def reset_calib_data(self):
        self.obj_points.clear()
        self.img_points.clear()
        # 预先构建通用的物方坐标
        self.objp = np.zeros((self.grid_rows * self.grid_cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.grid_rows, 0:self.grid_cols].T.reshape(-1, 2) * self.square_size

    def update_settings(self, rows, cols, size):
        self.grid_rows = rows
        self.grid_cols = cols
        self.square_size = size
        self.reset_calib_data()

    def detect_chessboard(self, bgr_frame):
        """ 实时寻找当前帧的标定板角点 """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        self.img_shape = gray.shape[::-1]
        
        # 棋盘格角点检测
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, (self.grid_rows, self.grid_cols), flags)
        
        if ret:
            # 亚像素级边缘精细化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 在新画布上绘制标定轨迹，用于界面反馈展示
            annotated_frame = bgr_frame.copy()
            cv2.drawChessboardCorners(annotated_frame, (self.grid_rows, self.grid_cols), corners_sub, ret)
            return True, corners_sub, annotated_frame
        return False, None, bgr_frame

    def save_sample(self, corners_sub):
        self.obj_points.append(self.objp)
        self.img_points.append(corners_sub)
        return len(self.obj_points)

    def compute_intrinsic(self):
        """ 计算相机内外参数 """
        if len(self.obj_points) < 5:
            return False, "图像样本过少。推荐至少成功捕捉 10 组以上不同角度的图像"
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, self.img_shape, None, None
        )
        
        result = {
            "rms": ret,            # 重投影均方根误差
            "mtx": mtx,            # 内参矩阵 (fx, fy, cx, cy)
            "dist": dist           # 畸变参数 (k1, k2, p1, p2, k3)
        }
        return True, result


# ==========================================
# 3. PyQt5 主界面业务逻辑
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPGA相机网络数据通道高级标定系统")
        self.resize(1100, 700)
        
        # 算法与底层初始化
        self.calibrator = CameraCalibrator()
        self.latest_frame = None
        self.active_sensor_id = 0  # 当前界面选择观察的 Sensor ID
        
        self.init_ui()
        
        # 实例化接收网络线程 (根据具体相机参数修改，例如 1280x1024)
        self.net_worker = UdpImageReceiver(port=60000, img_width=1280, img_height=1024)
        self.net_worker.frame_ready.connect(self.on_image_arrived)
        self.net_worker.log_signal.connect(self.append_log)
        self.net_worker.start()

    def init_ui(self):
        # 创建中央主布局
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # ----------------- 左侧：图像监控渲染区 -----------------
        left_layout = QVBoxLayout()
        self.image_viewer = QLabel("等待下位机 UDP 图像数据流...")
        self.image_viewer.setAlignment(Qt.AlignCenter)
        self.image_viewer.setStyleSheet("background-color: #121212; color: #666666; font-size: 16px; border: 2px solid #333;")
        self.image_viewer.setMinimumSize(640, 512)
        # 允许控件缩放图像
        self.image_viewer.setScaledContents(True)
        left_layout.addWidget(self.image_viewer)
        main_layout.addLayout(left_layout, stretch=3)
        
        # ----------------- 右侧：设备控制与算法操作台 -----------------
        right_panel = QVBoxLayout()
        
        # 模块 1: 下位机通信控制
        comm_group = QGroupBox("下位机控制")
        comm_layout = QFormLayout(comm_group)
        self.ip_field = QLineEdit("192.168.1.10")  # 默认目标 FPGA IP
        self.sensor_selector = QComboBox()
        self.sensor_selector.addItems(["Sensor 0 (ID: 0x00)", "Sensor 1 (ID: 0x01)"])
        self.sensor_selector.currentIndexChanged.connect(self.switch_sensor_channel)
        
        self.btn_shake = QPushButton("发送下行握手包 (CMD: 0x01)")
        self.btn_shake.clicked.connect(self.trigger_handshake)
        
        comm_layout.addRow("目标 FPGA IP:", self.ip_field)
        comm_layout.addRow("显示传感器:", self.sensor_selector)
        comm_layout.addRow(self.btn_shake)
        right_panel.addWidget(comm_group)
        
        # 模块 2: 标定参数配置
        calib_cfg_group = QGroupBox("标定板规格设置")
        cfg_layout = QFormLayout(calib_cfg_group)
        self.input_rows = QLineEdit("11")
        self.input_cols = QLineEdit("8")
        self.input_size = QLineEdit("20.0")
        self.btn_update_cfg = QPushButton("应用当前棋盘格规格")
        self.btn_update_cfg.clicked.connect(self.apply_grid_config)
        
        cfg_layout.addRow("内部行角点数:", self.input_rows)
        cfg_layout.addRow("内部列角点数:", self.input_cols)
        cfg_layout.addRow("单格物理尺寸(mm):", self.input_size)
        cfg_layout.addRow(self.btn_update_cfg)
        right_panel.addWidget(calib_cfg_group)

        # 模块 3: 核心动作触发
        action_group = QGroupBox("标定控制台")
        action_layout = QVBoxLayout(action_group)
        self.btn_capture = QPushButton("捕获并保存当前帧角点")
        self.btn_run = QPushButton("★ 执行相机标定解算 ★")
        self.btn_reset = QPushButton("清空已存样本")
        
        self.btn_capture.setStyleSheet("background-color: #2a52be; color: white; font-weight: bold; padding: 6px;")
        self.btn_run.setStyleSheet("background-color: #4f7942; color: white; font-weight: bold; padding: 6px;")
        
        self.btn_capture.clicked.connect(self.capture_and_add_sample)
        self.btn_run.clicked.connect(self.execute_calibration_algorithm)
        self.btn_reset.clicked.connect(self.clear_calib_cache)
        
        action_layout.addWidget(self.btn_capture)
        action_layout.addWidget(self.btn_run)
        action_layout.addWidget(self.btn_reset)
        right_panel.addWidget(action_group)
        
        # 模块 4: 系统状态与参数输出日志
        right_panel.addWidget(QLabel("系统日志与数据矩阵输出:"))
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setStyleSheet("background-color: #1e1e1e; color: #a9dfbf; font-family: Consolas;")
        right_panel.addWidget(self.log_viewer)
        
        main_layout.addLayout(right_panel, stretch=2)
        self.setCentralWidget(central_widget)

    # ----------------- 业务逻辑插槽函数 -----------------
    @pyqtSlot(str)
    def append_log(self, text):
        """ 线程安全的日志追加 """
        current_time = time.strftime("%H:%M:%S", time.localtime())
        self.log_viewer.append(f"[{current_time}] {text}")

    def trigger_handshake(self):
        target_ip = self.ip_field.text().strip()
        self.net_worker.send_handshake(target_ip)
        self.append_log(f"已向 {target_ip}:60000 递交握手包，请求开流...")

    def switch_sensor_channel(self, index):
        self.active_sensor_id = index
        self.append_log(f"视图已切换，当前聚焦监视: Sensor {index}")

    @pyqtSlot(int, np.ndarray)
    def on_image_arrived(self, sensor_id, frame_bgr):
        """ 底层拼包完毕后激活的实时渲染插槽 """
        # 只渲染当前用户所选通道的图像
        if sensor_id == self.active_sensor_id:
            self.latest_frame = frame_bgr
            self.display_opencv_img(frame_bgr)

    def display_opencv_img(self, img):
        """ 将 OpenCV 的 Mat 阵列渲染到 QLabel 视图 """
        h, w, ch = img.shape
        bytes_per_line = ch * w
        # 由于 OpenCV 默认为 BGR，需转换为 RGB 进行显示
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_viewer.setPixmap(QPixmap.fromImage(q_image))

    def apply_grid_config(self):
        try:
            r = int(self.input_rows.text())
            c = int(self.input_cols.text())
            s = float(self.input_size.text())
            self.calibrator.update_settings(r, c, s)
            self.append_log(f"成功更新标定规格：内角点 [{r} x {c}], 网格尺寸: {s} mm.")
        except ValueError:
            self.append_log("配置应用失败：请输入合法的数字输入格式！")

    def capture_and_add_sample(self):
        if self.latest_frame is None:
            self.append_log("捕获失败：当前未接收到有效的视频流帧！")
            return
            
        # 调用 OpenCV 进行实时寻找
        success, corners_sub, annotated_img = self.calibrator.detect_chessboard(self.latest_frame)
        if success:
            # 高亮突出显示检测到的连线结果
            self.display_opencv_img(annotated_img)
            # 保存到算法数据队列
            total_samples = self.calibrator.save_sample(corners_sub)
            self.append_log(f"样本添加成功 (第 {total_samples} 组).")
        else:
            self.append_log("捕捉失败：当前视场内未检测到完整的棋盘格角点。")

    def execute_calibration_algorithm(self):
        self.append_log("正基于当前样本池启动解算，请稍候...")
        # 强制更新下界面渲染防止卡死观感
        QApplication.processEvents()
        
        success, response = self.calibrator.compute_intrinsic()
        if success:
            self.append_log("\n" + "="*15 + " 标定成功计算报告 " + "="*15)
            self.append_log(f"重投影均方根误差 (RMS Error): {response['rms']:.5f} pixels")
            
            # 格式化输出内参矩阵
            mtx = response['mtx']
            self.append_log(f"相机内参矩阵 (Camera Matrix):\n"
                            f"  [ fx={mtx[0,0]:.3f},  0,          cx={mtx[0,2]:.3f} ]\n"
                            f"  [ 0,          fy={mtx[1,1]:.3f},  cy={mtx[1,2]:.3f} ]\n"
                            f"  [ 0,          0,          1          ]")
            
            # 格式化输出畸变系数
            dist = response['dist'].flatten()
            dist_str = ", ".join([f"{val:.5f}" for val in dist])
            self.append_log(f"畸变参数 (k1, k2, p1, p2, k3):\n  [{dist_str}]")
            self.append_log("="*46 + "\n")
        else:
            self.append_log(f"解算终止: {response}")

    def clear_calib_cache(self):
        self.calibrator.reset_calib_data()
        self.append_log("已成功重置并清空所有底层标定样本点。")

    def closeEvent(self, event):
        # 拦截关闭，安全释放 socket 资源，规避线程死锁
        self.net_worker.stop()
        event.accept()

# ==========================================
# 4. 程序主入口
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置一下全系统的主题美化样式
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())