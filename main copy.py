import sys
import socket
import struct
import time
import random
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QComboBox,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QTimer

# ==========================================
# CRC-16/MODBUS 工具
# ==========================================
class CRC16_MODBUS:
    """计算 CRC-16/MODBUS：多项式 0x8005，初始 0xFFFF，输出异或 0x0000"""
    @staticmethod
    def calc(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

# ==========================================
# 1. 数据通道接收线程（图像 + 握手应答）
# ==========================================
class UdpImageReceiver(QThread):
    frame_ready = pyqtSignal(int, np.ndarray)   # sensor_id, bgr_image
    log_signal = pyqtSignal(str)
    handshake_ack = pyqtSignal(int)             # sensor_id 收到握手应答

    def __init__(self, data_port=60000, img_width=1280, img_height=1024):
        super().__init__()
        self.data_port = data_port
        self.img_width = img_width
        self.img_height = img_height
        self.frame_size = img_width * img_height
        self.payload_per_packet = 1032
        self.expected_pkts = (self.frame_size + self.payload_per_packet - 1) // self.payload_per_packet

        self.frame_buffers = {0: bytearray(self.expected_pkts * self.payload_per_packet),
                              1: bytearray(self.expected_pkts * self.payload_per_packet)}
        self.received_packet_ids = {0: set(), 1: set()}
        self.running = True
        self.sock = None

    def verify_checksum(self, packet_bytes):
        """校验和：chip_id(1B) + cmd(1B) + pkt_id(2B) + dataLen(2B) + data(1032B) 累加，低16位"""
        calc = sum(packet_bytes[2:1041]) & 0xFFFF   # 下标2到1040共1039字节？实际包结构：2~1040 包含了上述字段
        recv = struct.unpack(">H", packet_bytes[1041:1043])[0]
        return calc == recv

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(("0.0.0.0", self.data_port))
        except Exception as e:
            self.log_signal.emit(f"绑定数据端口 {self.data_port} 失败: {e}")
            return
        self.sock.settimeout(0.5)
        self.log_signal.emit(f"数据通道 UDP 监听端口 {self.data_port}")

        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
                if len(data) < 1044:
                    continue

                # 解析帧头 (大端)
                header = struct.unpack(">H", data[0:2])[0]
                if header != 0x55AA:   # FPGA 上行帧头为 0x55AA
                    continue
                tail = struct.unpack(">H", data[1043:1045])[0]
                if tail != 0xFAFA:
                    continue

                sensor_id = data[2]
                cmd = data[3]
                pkt_id = struct.unpack(">H", data[4:6])[0]
                data_len = struct.unpack(">H", data[6:8])[0]

                # 握手应答帧 (cmd=0x01)
                if cmd == 0x01 and data_len == 16:
                    self.log_signal.emit(f"收到 Sensor{sensor_id} 握手应答")
                    self.handshake_ack.emit(sensor_id)
                    continue

                # 图像数据帧 (cmd=0x00)
                if cmd == 0x00 and data_len == 1032:
                    if not self.verify_checksum(data):
                        continue
                    if pkt_id >= self.expected_pkts:
                        continue
                    start = pkt_id * self.payload_per_packet
                    end = start + self.payload_per_packet
                    self.frame_buffers[sensor_id][start:end] = data[8:1040]
                    self.received_packet_ids[sensor_id].add(pkt_id)

                    if len(self.received_packet_ids[sensor_id]) == self.expected_pkts:
                        raw = np.frombuffer(self.frame_buffers[sensor_id][:self.frame_size], dtype=np.uint8)
                        gray = raw.reshape((self.img_height, self.img_width))
                        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        self.frame_ready.emit(sensor_id, bgr)
                        self.received_packet_ids[sensor_id].clear()

            except socket.timeout:
                continue
            except Exception as e:
                self.log_signal.emit(f"数据接收异常: {e}")
        if self.sock:
            self.sock.close()

    def send_handshake(self, target_ip, sensor_id=0):
        """发送数据通道握手包（上位机下行）"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # 构造 data 字段 0x00~0x0f
            data_field = bytes(range(16))
            # 包序号从0开始
            pkt_id = 0
            # 计算校验和：chip_id + cmd + pkt_id + dataLen + data
            checksum_data = struct.pack(">B B H H 16s", sensor_id, 0x01, pkt_id, 16, data_field)
            checksum = sum(checksum_data) & 0xFFFF
            packet = struct.pack(">H B B H H 16s H H",
                                 0xAA55, sensor_id, 0x01, pkt_id, 16,
                                 data_field, checksum, 0xFAFA)
            sock.sendto(packet, (target_ip, self.data_port))
        except Exception as e:
            self.log_signal.emit(f"握手发送失败: {e}")
        finally:
            sock.close()

    def stop(self):
        self.running = False
        self.wait()


# ==========================================
# 2. 控制通道客户端（同步请求-应答）
# ==========================================
class UdpControlClient:
    def __init__(self, ctrl_port=60001):
        self.ctrl_port = ctrl_port
        self.sock = None
        self.sequence = 0   # 随机消息序号，实际用 random
        self.resp_event = None
        self.resp_data = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("0.0.0.0", self.ctrl_port))
        self.sock.settimeout(1.0)

    def close(self):
        if self.sock:
            self.sock.close()

    def _crc16(self, data: bytes) -> int:
        return CRC16_MODBUS.calc(data)

    def _build_request(self, device_type, func_code, rw_flag, data_bytes, msg_id=None):
        """构建控制请求帧"""
        if msg_id is None:
            msg_id = random.randint(0, 65535)
        msg_type = 0x00          # 请求消息
        data_len = len(data_bytes)
        # 待校验字段：设备类型(1) + 功能码(1) + 消息序号(2) + 消息类型(1) + 读/写控制(1) + 数据长度(2) + 数据(n)
        check_data = struct.pack(">B B H B B H", device_type, func_code, msg_id, msg_type, rw_flag, data_len) + data_bytes
        crc = self._crc16(check_data)
        packet = struct.pack(">H B B H B B H {}s H H".format(data_len),
                             0xAA55, device_type, func_code, msg_id, msg_type, rw_flag, data_len,
                             data_bytes, crc, 0xFAFA)
        return packet, msg_id

    def send_command(self, target_ip, device_type, func_code, rw_flag, data_bytes, timeout=2.0):
        """发送命令并等待应答，返回 (success, response_data)"""
        if not self.sock:
            self.start()
        packet, msg_id = self._build_request(device_type, func_code, rw_flag, data_bytes)
        self.sock.sendto(packet, (target_ip, self.ctrl_port))

        # 等待应答（匹配同设备类型、功能码、消息序号）
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp, addr = self.sock.recvfrom(2048)
                if len(resp) < 10:
                    continue
                # 解析应答
                header = struct.unpack(">H", resp[0:2])[0]
                if header != 0xAA55:
                    continue
                tail = struct.unpack(">H", resp[-2:])[0]
                if tail != 0xFAFA:
                    continue
                dev_type_recv = resp[2]
                func_recv = resp[3]
                msg_id_recv = struct.unpack(">H", resp[4:6])[0]
                msg_type = resp[6]
                rw_recv = resp[7]
                data_len = struct.unpack(">H", resp[8:10])[0]
                # 校验 CRC
                calc_crc = self._crc16(resp[2:-4])   # 从设备类型到数据末尾
                recv_crc = struct.unpack(">H", resp[-4:-2])[0]
                if calc_crc != recv_crc:
                    continue
                if dev_type_recv == device_type and func_recv == func_code and msg_id_recv == msg_id:
                    if data_len >= 2:
                        status = struct.unpack(">H", resp[10:12])[0]  # 应答状态码
                        if status != 0:
                            return False, f"错误码 {status}"
                        # 读操作时，后面还有数据
                        if rw_flag == 0x80 and data_len > 2:
                            return True, resp[12:12+data_len-2]
                        else:
                            return True, b''
                    else:
                        return False, "应答长度错误"
            except socket.timeout:
                continue
        return False, "超时未收到应答"

    # 以下为封装好的便捷函数
    def set_exposure(self, target_ip, sensor_id, exposure_us):
        """设置曝光（微秒）"""
        data = struct.pack(">I", exposure_us)
        return self.send_command(target_ip, sensor_id, 0x01, 0x40, data)

    def get_exposure(self, target_ip, sensor_id):
        success, data = self.send_command(target_ip, sensor_id, 0x01, 0x80, b'\x00\x00')
        if success and len(data) >= 4:
            return struct.unpack(">I", data[:4])[0]
        return None

    def set_gain(self, target_ip, sensor_id, gain):
        data = struct.pack(">I", gain)
        return self.send_command(target_ip, sensor_id, 0x02, 0x40, data)

    def get_gain(self, target_ip, sensor_id):
        success, data = self.send_command(target_ip, sensor_id, 0x02, 0x80, b'\x00\x00')
        if success and len(data) >= 4:
            return struct.unpack(">I", data[:4])[0]
        return None

    def set_trigger_interval(self, target_ip, sensor_id, interval_us):
        data = struct.pack(">I", interval_us)
        return self.send_command(target_ip, sensor_id, 0x03, 0x40, data)

    def set_encoder_trigger_mode(self, target_ip, sensor_id, mode):
        data = struct.pack(">I", mode)
        return self.send_command(target_ip, sensor_id, 0x04, 0x40, data)

    def set_sample_rate(self, target_ip, sensor_id, rate):
        data = struct.pack(">I", rate)
        return self.send_command(target_ip, sensor_id, 0x07, 0x40, data)

    def get_sample_rate(self, target_ip, sensor_id):
        success, data = self.send_command(target_ip, sensor_id, 0x07, 0x80, b'\x00\x00')
        if success and len(data) >= 4:
            return struct.unpack(">I", data[:4])[0]
        return None

    def set_roi_x(self, target_ip, sensor_id, roi_start, roi_end):
        data = struct.pack(">II", roi_start, roi_end)
        return self.send_command(target_ip, sensor_id, 0x13, 0x40, data)

    def set_x_interval(self, target_ip, sensor_id, interval):
        data = struct.pack(">I", interval)
        return self.send_command(target_ip, sensor_id, 0x14, 0x40, data)

    def set_z_range(self, target_ip, sensor_id, min_z, max_z):
        data = struct.pack(">II", min_z, max_z)
        return self.send_command(target_ip, sensor_id, 0x15, 0x40, data)

    def set_horizontal_flip(self, target_ip, sensor_id, enable):
        data = struct.pack(">I", 1 if enable else 0)
        return self.send_command(target_ip, sensor_id, 0x16, 0x40, data)

    def set_vertical_flip(self, target_ip, sensor_id, enable):
        data = struct.pack(">I", 1 if enable else 0)
        return self.send_command(target_ip, sensor_id, 0x17, 0x40, data)

    def set_batch_length(self, target_ip, sensor_id, length):
        data = struct.pack(">I", length)
        return self.send_command(target_ip, sensor_id, 0x09, 0x40, data)

    def set_batch_output_type(self, target_ip, sensor_id, out_type):
        data = struct.pack(">I", out_type)
        return self.send_command(target_ip, sensor_id, 0x10, 0x40, data)

    def start_batch(self, target_ip, sensor_id):
        return self.send_command(target_ip, sensor_id, 0x11, 0x40, struct.pack(">H", 1))

    def stop_batch(self, target_ip, sensor_id):
        return self.send_command(target_ip, sensor_id, 0x11, 0x40, struct.pack(">H", 0))

    def get_sensor_serial(self, target_ip, sensor_id):
        return self.send_command(target_ip, sensor_id, 0x05, 0x80, b'\x00\x00')

    def get_sensor_fw_version(self, target_ip, sensor_id):
        return self.send_command(target_ip, sensor_id, 0x06, 0x80, b'\x00\x00')

    # FPGA 相关
    def set_fpga_ip(self, target_ip, ip_int):
        data = struct.pack(">I", ip_int)
        return self.send_command(target_ip, 0x00, 0x01, 0x40, data)

    def get_fpga_ip(self, target_ip):
        success, data = self.send_command(target_ip, 0x00, 0x01, 0x80, b'\x00\x00')
        if success and len(data) >= 4:
            return struct.unpack(">I", data[:4])[0]
        return None

    def set_host_ip(self, target_ip, ip_int):
        data = struct.pack(">I", ip_int)
        return self.send_command(target_ip, 0x00, 0x02, 0x40, data)

    def get_host_ip(self, target_ip):
        success, data = self.send_command(target_ip, 0x00, 0x02, 0x80, b'\x00\x00')
        if success and len(data) >= 4:
            return struct.unpack(">I", data[:4])[0]
        return None

    def get_fpga_version(self, target_ip):
        success, data = self.send_command(target_ip, 0x00, 0x03, 0x80, b'\x00\x00')
        if success and len(data) >= 8:
            return data[:8].decode('ascii', errors='ignore')
        return None

    # EEPROM 读写
    def eeprom_write(self, target_ip, eeprom_id, address, data_bytes):
        """
        eeprom_id: 0x20 或 0x21
        address: 24位地址
        data_bytes: 要写入的字节串，长度 ≤256
        """
        if len(data_bytes) > 256:
            return False, "数据过长"
        # 构造数据: 命令(0x40) + 长度(16bit) + 地址(24bit) + 数据
        length = len(data_bytes)
        # 如果长度为奇数，填充0x00
        if length % 2 != 0:
            data_bytes += b'\x00'
        payload = struct.pack(">B H I", 0x40, length, address & 0xFFFFFF) + data_bytes
        return self.send_command(target_ip, eeprom_id, 0x00, 0x40, payload)

    def eeprom_read(self, target_ip, eeprom_id, address, length):
        payload = struct.pack(">B H I", 0x80, length, address & 0xFFFFFF)
        success, data = self.send_command(target_ip, eeprom_id, 0x00, 0x40, payload)
        if success and len(data) >= 6:
            # 数据格式: 命令 + 长度 + 地址 + 实际数据
            return data[6:6+length]
        return None


# ==========================================
# 3. 相机标定算法（保持不变）
# ==========================================
class CameraCalibrator:
    def __init__(self):
        self.grid_rows = 11
        self.grid_cols = 8
        self.square_size = 20.0
        self.obj_points = []
        self.img_points = []
        self.img_shape = None
        self.reset_calib_data()

    def reset_calib_data(self):
        self.obj_points.clear()
        self.img_points.clear()
        self.objp = np.zeros((self.grid_rows * self.grid_cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.grid_rows, 0:self.grid_cols].T.reshape(-1, 2) * self.square_size

    def update_settings(self, rows, cols, size):
        self.grid_rows = rows
        self.grid_cols = cols
        self.square_size = size
        self.reset_calib_data()

    def detect_chessboard(self, bgr_frame):
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        self.img_shape = gray.shape[::-1]
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, (self.grid_rows, self.grid_cols), flags)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            annotated = bgr_frame.copy()
            cv2.drawChessboardCorners(annotated, (self.grid_rows, self.grid_cols), corners_sub, ret)
            return True, corners_sub, annotated
        return False, None, bgr_frame

    def save_sample(self, corners_sub):
        self.obj_points.append(self.objp)
        self.img_points.append(corners_sub)
        return len(self.obj_points)

    def compute_intrinsic(self):
        if len(self.obj_points) < 5:
            return False, "样本数不足5"
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, self.img_shape, None, None)
        return True, {"rms": ret, "mtx": mtx, "dist": dist}


# ==========================================
# 4. 主窗口
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPGA 多传感器控制与标定系统")
        self.resize(1280, 800)

        # 网络组件
        self.data_receiver = None
        self.ctrl_client = UdpControlClient()
        self.fpga_ip = "192.168.1.10"    # 默认 FPGA IP
        self.current_sensor = 1           # 1~4

        # 标定器
        self.calibrator = CameraCalibrator()
        self.latest_frame = None

        self.init_ui()
        self.init_network()

    def init_network(self):
        self.ctrl_client.start()
        self.data_receiver = UdpImageReceiver(data_port=60000, img_width=1280, img_height=1024)
        self.data_receiver.frame_ready.connect(self.on_image_received)
        self.data_receiver.log_signal.connect(self.append_log)
        self.data_receiver.handshake_ack.connect(self.on_handshake_ack)
        self.data_receiver.start()

    def init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout(central)

        # 左侧：图像显示
        left = QVBoxLayout()
        self.image_label = QLabel("等待图像...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 512)
        self.image_label.setScaledContents(True)
        left.addWidget(self.image_label)
        main_layout.addLayout(left, stretch=3)

        # 右侧：标签页控件
        right = QTabWidget()

        # ---- 控制标签页 ----
        control_tab = QWidget()
        c_layout = QVBoxLayout(control_tab)

        # 基本设置区域
        base_group = QGroupBox("基本参数设置")
        base_form = QFormLayout(base_group)
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(["Sensor1", "Sensor2", "Sensor3", "Sensor4"])
        self.sensor_combo.currentIndexChanged.connect(self.change_sensor)
        self.fpga_ip_edit = QLineEdit(self.fpga_ip)
        self.btn_send_handshake = QPushButton("发送数据通道握手")
        self.btn_send_handshake.clicked.connect(self.send_handshake)

        base_form.addRow("当前传感器:", self.sensor_combo)
        base_form.addRow("FPGA IP:", self.fpga_ip_edit)
        base_form.addRow(self.btn_send_handshake)
        c_layout.addWidget(base_group)

        # 曝光/增益/触发组内的修正
        param_group = QGroupBox("曝光 / 增益 / 触发")
        param_form = QFormLayout(param_group)

        self.exposure_spin = QSpinBox()
        self.exposure_spin.setRange(0, 1000000)
        self.exposure_spin.setSuffix(" us")
        self.btn_set_exp = QPushButton("设置")
        self.btn_get_exp = QPushButton("读取")
        exp_hlayout = QHBoxLayout()
        exp_hlayout.addWidget(self.btn_set_exp)
        exp_hlayout.addWidget(self.btn_get_exp)

        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 48)
        self.btn_set_gain = QPushButton("设置")
        self.btn_get_gain = QPushButton("读取")
        gain_hlayout = QHBoxLayout()
        gain_hlayout.addWidget(self.btn_set_gain)
        gain_hlayout.addWidget(self.btn_get_gain)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(0, 1000000)
        self.interval_spin.setSuffix(" us")
        self.btn_set_interval = QPushButton("设置触发间隔")

        param_form.addRow("曝光时间:", self.exposure_spin)
        param_form.addRow("", exp_hlayout)          # 一行放置两个按钮
        param_form.addRow("增益:", self.gain_spin)
        param_form.addRow("", gain_hlayout)         # 一行放置两个按钮
        param_form.addRow("触发间隔:", self.interval_spin)
        param_form.addRow("", self.btn_set_interval)

        # ROI / 翻转 / 批处理等
        roi_group = QGroupBox("ROI / 翻转 / 批处理")
        roi_form = QFormLayout(roi_group)
        self.roi_start = QSpinBox(); self.roi_start.setRange(0, 1280)
        self.roi_end = QSpinBox(); self.roi_end.setRange(0, 1280)
        self.btn_set_roi = QPushButton("设置X轴ROI")
        self.hflip_cb = QCheckBox("水平翻转")
        self.vflip_cb = QCheckBox("垂直翻转")
        self.btn_set_flip = QPushButton("应用翻转")
        self.batch_len = QSpinBox(); self.batch_len.setRange(0, 65535)
        self.btn_set_batch_len = QPushButton("设置批处理长度")
        self.btn_start_batch = QPushButton("启动批处理")
        self.btn_stop_batch = QPushButton("停止批处理")

        roi_form.addRow("ROI X起始:", self.roi_start)
        roi_form.addRow("ROI X结束:", self.roi_end)
        roi_form.addRow(self.btn_set_roi)
        roi_form.addRow(self.hflip_cb, self.vflip_cb)
        roi_form.addRow(self.btn_set_flip)
        roi_form.addRow("批处理长度:", self.batch_len)
        roi_form.addRow(self.btn_set_batch_len)
        roi_form.addRow(self.btn_start_batch, self.btn_stop_batch)
        c_layout.addWidget(roi_group)

        # FPGA 配置
        fpga_group = QGroupBox("FPGA 配置与版本")
        fpga_form = QFormLayout(fpga_group)
        self.fpga_ip_set = QLineEdit("192.168.10.10")
        self.btn_set_fpga_ip = QPushButton("设置FPGA IP")
        self.btn_get_fpga_ip = QPushButton("获取FPGA IP")
        self.host_ip_set = QLineEdit("192.168.10.100")
        self.btn_set_host_ip = QPushButton("设置上位机IP")
        self.btn_get_host_ip = QPushButton("获取上位机IP")
        self.btn_get_fpga_ver = QPushButton("获取FPGA固件版本")
        self.btn_get_sensor_serial = QPushButton("获取传感器序列号")
        self.btn_get_sensor_fw = QPushButton("获取传感器固件版本")

        fpga_form.addRow("FPGA IP:", self.fpga_ip_set)
        fpga_form.addRow(self.btn_set_fpga_ip, self.btn_get_fpga_ip)
        fpga_form.addRow("上位机 IP:", self.host_ip_set)
        fpga_form.addRow(self.btn_set_host_ip, self.btn_get_host_ip)
        fpga_form.addRow(self.btn_get_fpga_ver)
        fpga_form.addRow(self.btn_get_sensor_serial)
        fpga_form.addRow(self.btn_get_sensor_fw)
        c_layout.addWidget(fpga_group)

        # EEPROM 读写
        eeprom_group = QGroupBox("EEPROM 读写 (0x20/0x21)")
        eeprom_form = QFormLayout(eeprom_group)
        self.eeprom_id = QComboBox()
        self.eeprom_id.addItems(["EEPROM0 (0x20)", "EEPROM1 (0x21)"])
        self.eeprom_addr = QLineEdit("0x000000")
        self.eeprom_len = QSpinBox(); self.eeprom_len.setRange(1, 256)
        self.eeprom_data = QTextEdit()
        self.eeprom_data.setMaximumHeight(80)
        self.btn_eeprom_read = QPushButton("读取")
        self.btn_eeprom_write = QPushButton("写入")
        eeprom_form.addRow("EEPROM选择:", self.eeprom_id)
        eeprom_form.addRow("地址(hex):", self.eeprom_addr)
        eeprom_form.addRow("长度(字节):", self.eeprom_len)
        eeprom_form.addRow("数据(hex):", self.eeprom_data)
        eeprom_form.addRow(self.btn_eeprom_read, self.btn_eeprom_write)
        c_layout.addWidget(eeprom_group)

        c_layout.addStretch()
        right.addTab(control_tab, "设备控制")

        # ---- 标定标签页 ----
        calib_tab = QWidget()
        calib_layout = QVBoxLayout(calib_tab)
        calib_cfg = QGroupBox("标定板参数")
        cfg_form = QFormLayout(calib_cfg)
        self.calib_rows = QSpinBox(); self.calib_rows.setRange(4, 20); self.calib_rows.setValue(11)
        self.calib_cols = QSpinBox(); self.calib_cols.setRange(4, 20); self.calib_cols.setValue(8)
        self.calib_size = QDoubleSpinBox(); self.calib_size.setRange(1, 100); self.calib_size.setValue(20.0)
        self.btn_apply_calib = QPushButton("应用棋盘格配置")
        cfg_form.addRow("内角点行数:", self.calib_rows)
        cfg_form.addRow("内角点列数:", self.calib_cols)
        cfg_form.addRow("方格尺寸(mm):", self.calib_size)
        cfg_form.addRow(self.btn_apply_calib)
        calib_layout.addWidget(calib_cfg)

        calib_btn_group = QGroupBox("标定操作")
        btn_layout = QHBoxLayout()
        self.btn_capture = QPushButton("捕获当前帧角点")
        self.btn_calibrate = QPushButton("执行标定")
        self.btn_reset_calib = QPushButton("清空样本")
        btn_layout.addWidget(self.btn_capture)
        btn_layout.addWidget(self.btn_calibrate)
        btn_layout.addWidget(self.btn_reset_calib)
        calib_btn_group.setLayout(btn_layout)
        calib_layout.addWidget(calib_btn_group)

        calib_log_group = QGroupBox("标定结果输出")
        self.calib_log = QTextEdit()
        self.calib_log.setReadOnly(True)
        calib_log_layout = QVBoxLayout(calib_log_group)
        calib_log_layout.addWidget(self.calib_log)
        calib_layout.addWidget(calib_log_group)
        right.addTab(calib_tab, "相机标定")

        # ---- 日志标签页 ----
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        log_layout.addWidget(self.log_viewer)
        right.addTab(log_tab, "系统日志")

        main_layout.addWidget(right, stretch=2)
        self.setCentralWidget(central)

        # 连接信号槽
        self.btn_set_exp.clicked.connect(self.set_exposure)
        self.btn_get_exp.clicked.connect(self.get_exposure)
        self.btn_set_gain.clicked.connect(self.set_gain)
        self.btn_get_gain.clicked.connect(self.get_gain)
        self.btn_set_interval.clicked.connect(self.set_trigger_interval)
        self.btn_set_roi.clicked.connect(self.set_roi)
        self.btn_set_flip.clicked.connect(self.set_flip)
        self.btn_set_batch_len.clicked.connect(self.set_batch_length)
        self.btn_start_batch.clicked.connect(self.start_batch)
        self.btn_stop_batch.clicked.connect(self.stop_batch)
        self.btn_set_fpga_ip.clicked.connect(self.set_fpga_ip)
        self.btn_get_fpga_ip.clicked.connect(self.get_fpga_ip)
        self.btn_set_host_ip.clicked.connect(self.set_host_ip)
        self.btn_get_host_ip.clicked.connect(self.get_host_ip)
        self.btn_get_fpga_ver.clicked.connect(self.get_fpga_version)
        self.btn_get_sensor_serial.clicked.connect(self.get_sensor_serial)
        self.btn_get_sensor_fw.clicked.connect(self.get_sensor_fw)
        self.btn_eeprom_read.clicked.connect(self.eeprom_read)
        self.btn_eeprom_write.clicked.connect(self.eeprom_write)
        self.btn_apply_calib.clicked.connect(self.apply_calib_config)
        self.btn_capture.clicked.connect(self.capture_corners)
        self.btn_calibrate.clicked.connect(self.run_calibration)
        self.btn_reset_calib.clicked.connect(self.reset_calibration)

    # ---------- 辅助函数 ----------
    def change_sensor(self, idx):
        self.current_sensor = idx + 1

    def get_target_ip(self):
        return self.fpga_ip_edit.text().strip()

    def append_log(self, msg):
        self.log_viewer.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def on_handshake_ack(self, sensor_id):
        self.append_log(f"数据通道握手成功，Sensor{sensor_id} 已就绪")

    def send_handshake(self):
        ip = self.get_target_ip()
        sensor = self.current_sensor - 1   # 内部使用0/1，这里发送所有传感器？暂时只发当前选中的
        self.data_receiver.send_handshake(ip, sensor)
        self.append_log(f"向 {ip} 发送数据通道握手 (Sensor{sensor})")

    @pyqtSlot(int, np.ndarray)
    def on_image_received(self, sensor_id, bgr):
        if sensor_id == self.current_sensor - 1:
            self.latest_frame = bgr
            h, w, ch = bgr.shape
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimg))

    # ---------- 控制命令封装 ----------
    def set_exposure(self):
        ip = self.get_target_ip()
        val = self.exposure_spin.value()
        suc, msg = self.ctrl_client.set_exposure(ip, self.current_sensor, val)
        self.append_log(f"设置曝光 {val}us: {'成功' if suc else '失败 '+msg}")

    def get_exposure(self):
        ip = self.get_target_ip()
        val = self.ctrl_client.get_exposure(ip, self.current_sensor)
        if val is not None:
            self.exposure_spin.setValue(val)
            self.append_log(f"当前曝光: {val}us")
        else:
            self.append_log("读取曝光失败")

    def set_gain(self):
        ip = self.get_target_ip()
        val = self.gain_spin.value()
        suc, msg = self.ctrl_client.set_gain(ip, self.current_sensor, val)
        self.append_log(f"设置增益 {val}: {'成功' if suc else '失败 '+msg}")

    def get_gain(self):
        ip = self.get_target_ip()
        val = self.ctrl_client.get_gain(ip, self.current_sensor)
        if val is not None:
            self.gain_spin.setValue(val)
            self.append_log(f"当前增益: {val}")
        else:
            self.append_log("读取增益失败")

    def set_trigger_interval(self):
        ip = self.get_target_ip()
        val = self.interval_spin.value()
        suc, msg = self.ctrl_client.set_trigger_interval(ip, self.current_sensor, val)
        self.append_log(f"设置触发间隔 {val}us: {'成功' if suc else '失败 '+msg}")

    def set_roi(self):
        ip = self.get_target_ip()
        start = self.roi_start.value()
        end = self.roi_end.value()
        suc, msg = self.ctrl_client.set_roi_x(ip, self.current_sensor, start, end)
        self.append_log(f"设置ROI [{start},{end}]: {'成功' if suc else '失败 '+msg}")

    def set_flip(self):
        ip = self.get_target_ip()
        h = self.hflip_cb.isChecked()
        v = self.vflip_cb.isChecked()
        suc1, _ = self.ctrl_client.set_horizontal_flip(ip, self.current_sensor, h)
        suc2, _ = self.ctrl_client.set_vertical_flip(ip, self.current_sensor, v)
        self.append_log(f"设置翻转 (H:{h} V:{v}): {suc1 and suc2}")

    def set_batch_length(self):
        ip = self.get_target_ip()
        length = self.batch_len.value()
        suc, msg = self.ctrl_client.set_batch_length(ip, self.current_sensor, length)
        self.append_log(f"设置批处理长度 {length}: {'成功' if suc else '失败 '+msg}")

    def start_batch(self):
        ip = self.get_target_ip()
        suc, msg = self.ctrl_client.start_batch(ip, self.current_sensor)
        self.append_log(f"启动批处理: {'成功' if suc else '失败 '+msg}")

    def stop_batch(self):
        ip = self.get_target_ip()
        suc, msg = self.ctrl_client.stop_batch(ip, self.current_sensor)
        self.append_log(f"停止批处理: {'成功' if suc else '失败 '+msg}")

    def set_fpga_ip(self):
        ip = self.get_target_ip()
        ip_str = self.fpga_ip_set.text()
        ip_int = struct.unpack(">I", socket.inet_aton(ip_str))[0]
        suc, msg = self.ctrl_client.set_fpga_ip(ip, ip_int)
        self.append_log(f"设置FPGA IP {ip_str}: {'成功' if suc else '失败 '+msg}")

    def get_fpga_ip(self):
        ip = self.get_target_ip()
        val = self.ctrl_client.get_fpga_ip(ip)
        if val:
            ip_str = socket.inet_ntoa(struct.pack(">I", val))
            self.fpga_ip_set.setText(ip_str)
            self.append_log(f"FPGA IP: {ip_str}")
        else:
            self.append_log("获取FPGA IP失败")

    def set_host_ip(self):
        ip = self.get_target_ip()
        ip_str = self.host_ip_set.text()
        ip_int = struct.unpack(">I", socket.inet_aton(ip_str))[0]
        suc, msg = self.ctrl_client.set_host_ip(ip, ip_int)
        self.append_log(f"设置上位机IP {ip_str}: {'成功' if suc else '失败 '+msg}")

    def get_host_ip(self):
        ip = self.get_target_ip()
        val = self.ctrl_client.get_host_ip(ip)
        if val:
            ip_str = socket.inet_ntoa(struct.pack(">I", val))
            self.host_ip_set.setText(ip_str)
            self.append_log(f"上位机IP: {ip_str}")
        else:
            self.append_log("获取上位机IP失败")

    def get_fpga_version(self):
        ip = self.get_target_ip()
        ver = self.ctrl_client.get_fpga_version(ip)
        if ver:
            self.append_log(f"FPGA固件版本: {ver}")
        else:
            self.append_log("获取固件版本失败")

    def get_sensor_serial(self):
        ip = self.get_target_ip()
        suc, data = self.ctrl_client.get_sensor_serial(ip, self.current_sensor)
        if suc:
            self.append_log(f"传感器序列号: {data.hex()}")
        else:
            self.append_log("获取序列号失败")

    def get_sensor_fw(self):
        ip = self.get_target_ip()
        suc, data = self.ctrl_client.get_sensor_fw_version(ip, self.current_sensor)
        if suc:
            self.append_log(f"传感器固件版本: {data.hex()}")
        else:
            self.append_log("获取固件版本失败")

    def eeprom_read(self):
        ip = self.get_target_ip()
        eeprom_id = 0x20 if self.eeprom_id.currentIndex() == 0 else 0x21
        addr = int(self.eeprom_addr.text(), 16)
        length = self.eeprom_len.value()
        data = self.ctrl_client.eeprom_read(ip, eeprom_id, addr, length)
        if data:
            hex_str = ' '.join(f'{b:02x}' for b in data)
            self.eeprom_data.setText(hex_str)
            self.append_log(f"EEPROM读取成功, 长度{len(data)}")
        else:
            self.append_log("EEPROM读取失败")

    def eeprom_write(self):
        ip = self.get_target_ip()
        eeprom_id = 0x20 if self.eeprom_id.currentIndex() == 0 else 0x21
        addr = int(self.eeprom_addr.text(), 16)
        hex_str = self.eeprom_data.toPlainText().strip().replace(' ', '')
        try:
            data = bytes.fromhex(hex_str)
        except:
            self.append_log("数据格式错误，请输入十六进制字节，如 0012AB")
            return
        suc, msg = self.ctrl_client.eeprom_write(ip, eeprom_id, addr, data)
        self.append_log(f"EEPROM写入: {'成功' if suc else '失败 '+msg}")

    # ---------- 标定相关 ----------
    def apply_calib_config(self):
        rows = self.calib_rows.value()
        cols = self.calib_cols.value()
        size = self.calib_size.value()
        self.calibrator.update_settings(rows, cols, size)
        self.calib_log.append(f"棋盘格配置已更新: {rows}x{cols}, 方格{size}mm")

    def capture_corners(self):
        if self.latest_frame is None:
            self.calib_log.append("无图像，请确保数据流已启动")
            return
        ret, corners, annotated = self.calibrator.detect_chessboard(self.latest_frame)
        if ret:
            self.calibrator.save_sample(corners)
            cnt = len(self.calibrator.obj_points)
            self.calib_log.append(f"成功捕获第 {cnt} 组角点")
            # 显示带角点的图像
            h,w,ch = annotated.shape
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qimg))
        else:
            self.calib_log.append("未检测到完整棋盘格")

    def run_calibration(self):
        ret, res = self.calibrator.compute_intrinsic()
        if ret:
            self.calib_log.append("===== 标定结果 =====")
            self.calib_log.append(f"重投影误差 RMS: {res['rms']:.5f}")
            mtx = res['mtx']
            self.calib_log.append(f"内参矩阵:\n{mtx}")
            dist = res['dist'].flatten()
            self.calib_log.append(f"畸变系数: {dist}")
        else:
            self.calib_log.append(f"标定失败: {res}")

    def reset_calibration(self):
        self.calibrator.reset_calib_data()
        self.calib_log.append("已清空所有标定样本")

    def closeEvent(self, event):
        if self.data_receiver:
            self.data_receiver.stop()
        if self.ctrl_client:
            self.ctrl_client.close()
        event.accept()


# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())