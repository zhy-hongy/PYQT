# -*- coding: utf-8 -*-
# 依赖: pip install PyQt5 pyserial

import sys
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


def _make_link_icon(size=20, connected=True):
    """简单绿色/灰色链环状图标（无资源文件时占位）。"""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    color = QColor(46, 204, 113) if connected else QColor(160, 160, 160)
    p.setPen(color)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(2, 6, 10, 10)
    p.drawEllipse(8, 4, 10, 10)
    p.end()
    return QIcon(pm)


def _make_refresh_icon(size=18):
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QColor(80, 80, 80))
    p.drawArc(3, 3, 12, 12, 45 * 16, 270 * 16)
    p.drawLine(13, 4, 15, 6)
    p.drawLine(14, 3, 15, 6)
    p.end()
    return QIcon(pm)


class SerialSettingsDialog(QDialog):
    """详细串口参数对话框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dialog")
        gb = QGroupBox("设置")
        form = QFormLayout(gb)

        self.port_combo = QComboBox()
        self.baud_combo = QComboBox()
        for b in (9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600):
            self.baud_combo.addItem(str(b), b)

        self.data_combo = QComboBox()
        for d in (5, 6, 7, 8):
            self.data_combo.addItem(str(d), d)

        self.stop_combo = QComboBox()
        if serial:
            self.stop_combo.addItem("1", serial.STOPBITS_ONE)
            self.stop_combo.addItem("1.5", serial.STOPBITS_ONE_POINT_FIVE)
            self.stop_combo.addItem("2", serial.STOPBITS_TWO)
        else:
            self.stop_combo.addItem("1", 1)
            self.stop_combo.addItem("1.5", 1.5)
            self.stop_combo.addItem("2", 2)

        self.parity_combo = QComboBox()
        if serial:
            self.parity_combo.addItem("None", serial.PARITY_NONE)
            self.parity_combo.addItem("Even", serial.PARITY_EVEN)
            self.parity_combo.addItem("Odd", serial.PARITY_ODD)
            self.parity_combo.addItem("Mark", serial.PARITY_MARK)
            self.parity_combo.addItem("Space", serial.PARITY_SPACE)
        else:
            for name, ch in (
                ("None", "N"),
                ("Even", "E"),
                ("Odd", "O"),
                ("Mark", "M"),
                ("Space", "S"),
            ):
                self.parity_combo.addItem(name, ch)

        self.flow_combo = QComboBox()
        self.flow_combo.addItem("None", "none")
        self.flow_combo.addItem("RTS/CTS", "rtscts")
        self.flow_combo.addItem("XON/XOFF", "xonxoff")

        form.addRow("端口号:", self.port_combo)
        form.addRow("波特率:", self.baud_combo)
        form.addRow("数据位:", self.data_combo)
        form.addRow("停止位:", self.stop_combo)
        form.addRow("校验位:", self.parity_combo)
        form.addRow("流控制:", self.flow_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout(self)
        root.addWidget(gb)
        root.addWidget(buttons)

    def refresh_ports(self, select_device=None):
        self.port_combo.clear()
        if list_ports is None:
            return
        for p in list_ports.comports():
            label = f"{p.description} ({p.device})"
            self.port_combo.addItem(label, p.device)
        if select_device:
            for i in range(self.port_combo.count()):
                if self.port_combo.itemData(i) == select_device:
                    self.port_combo.setCurrentIndex(i)
                    break

    def set_values(self, device, baud, data_bits, stop_bits, parity, flow):
        self.refresh_ports(select_device=device)
        idx = self.baud_combo.findData(baud)
        if idx >= 0:
            self.baud_combo.setCurrentIndex(idx)
        else:
            self.baud_combo.setCurrentText(str(baud))
        di = self.data_combo.findData(data_bits)
        if di >= 0:
            self.data_combo.setCurrentIndex(di)
        si = self.stop_combo.findData(stop_bits)
        if si >= 0:
            self.stop_combo.setCurrentIndex(si)
        pi = self.parity_combo.findData(parity)
        if pi >= 0:
            self.parity_combo.setCurrentIndex(pi)
        if flow == "rtscts":
            self.flow_combo.setCurrentIndex(1)
        elif flow == "xonxoff":
            self.flow_combo.setCurrentIndex(2)
        else:
            self.flow_combo.setCurrentIndex(0)

    def get_values(self):
        device = self.port_combo.currentData()
        baud_text = self.baud_combo.currentText()
        try:
            baud = int(baud_text)
        except ValueError:
            baud = 115200
        data_bits = self.data_combo.currentData()
        stop_bits = self.stop_combo.currentData()
        parity = self.parity_combo.currentData()
        flow = self.flow_combo.currentData()
        return device, baud, data_bits, stop_bits, parity, flow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MainWindow")
        self.resize(960, 640)

        self._ser = None
        self._device = None
        self._baud = 115200
        self._data_bits = 8
        self._stop_bits = serial.STOPBITS_ONE if serial else 1
        self._parity = serial.PARITY_NONE if serial else "N"
        self._flow = "none"

        self._read_timer = QTimer(self)
        self._read_timer.timeout.connect(self._poll_read)

        tabs = QTabWidget()
        tabs.addTab(self._build_serial_tab(), "串口助手")
        tabs.addTab(self._build_bluetooth_tab(), "蓝牙助手")
        self.setCentralWidget(tabs)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("就绪")

    def _build_bluetooth_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        hint = QLabel("蓝牙助手：后续可在此对接蓝牙串口或 BLE。")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(hint)
        return w

    def _build_serial_tab(self):
        root = QWidget()
        main_lay = QHBoxLayout(root)

        left = QGroupBox("串口设置")
        left_grid = QGridLayout(left)

        dev_lbl = QLabel("设备:")
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(220)
        refresh_btn = QPushButton()
        refresh_btn.setIcon(_make_refresh_icon())
        refresh_btn.setFixedWidth(32)
        refresh_btn.setToolTip("刷新端口")
        refresh_btn.clicked.connect(self._refresh_ports)

        baud_lbl = QLabel("波特率:")
        self.baud_combo = QComboBox()
        for b in (9600, 19200, 38400, 57600, 115200, 230400, 460800):
            self.baud_combo.addItem(str(b), b)
        self.baud_combo.setCurrentIndex(self.baud_combo.findData(115200))
        baud_more = QPushButton("...")
        baud_more.setFixedWidth(28)
        baud_more.setToolTip("详细设置")
        baud_more.clicked.connect(self._open_settings_dialog)

        self.link_label = QLabel()
        self.link_label.setPixmap(_make_link_icon(22, False).pixmap(22, 22))
        self.toggle_btn = QPushButton("打开设备")
        self.toggle_btn.setMinimumHeight(36)
        self.toggle_btn.clicked.connect(self._toggle_serial)

        row_dev = QHBoxLayout()
        row_dev.addWidget(self.device_combo, 1)
        row_dev.addWidget(refresh_btn)

        row_baud = QHBoxLayout()
        row_baud.addWidget(self.baud_combo, 1)
        row_baud.addWidget(baud_more)

        left_grid.addWidget(dev_lbl, 0, 0)
        left_grid.addLayout(row_dev, 0, 1)
        left_grid.addWidget(baud_lbl, 1, 0)
        left_grid.addLayout(row_baud, 1, 1)

        conn_row = QHBoxLayout()
        conn_row.addWidget(self.link_label)
        conn_row.addWidget(self.toggle_btn, 1)
        left_grid.addLayout(conn_row, 2, 0, 1, 2)

        self.chk_rts = QCheckBox("RTS")
        self.chk_dtr = QCheckBox("DTR")
        self.chk_rts.toggled.connect(self._on_rts)
        self.chk_dtr.toggled.connect(self._on_dtr)
        left_grid.addWidget(self.chk_rts, 3, 0, 1, 2)
        left_grid.addWidget(self.chk_dtr, 4, 0, 1, 2)
        left_grid.setRowStretch(5, 1)

        right_split = QSplitter(Qt.Vertical)

        recv_box = QGroupBox("接收区")
        recv_lay = QHBoxLayout(recv_box)
        self.recv_edit = QTextEdit()
        self.recv_edit.setReadOnly(True)
        self.recv_edit.setPlaceholderText("接收数据将显示在此处…")
        recv_side = QVBoxLayout()
        self.chk_hex_show = QCheckBox("HEX显示")
        self.chk_timestamp = QCheckBox("加时间戳")
        btn_clear_recv = QPushButton("清空接收")
        btn_clear_recv.clicked.connect(self.recv_edit.clear)
        recv_side.addWidget(self.chk_hex_show)
        recv_side.addWidget(self.chk_timestamp)
        recv_side.addWidget(btn_clear_recv)
        recv_side.addStretch()
        recv_lay.addWidget(self.recv_edit, 1)
        recv_lay.addLayout(recv_side)

        send_box = QGroupBox("发送区")
        send_lay = QHBoxLayout(send_box)
        self.send_edit = QTextEdit()
        self.send_edit.setPlaceholderText("输入要发送的内容…")
        send_side = QVBoxLayout()
        self.chk_hex_send = QCheckBox("HEX发送")
        self.chk_crlf = QCheckBox("加回车换行")
        btn_clear_send = QPushButton("清空发送")
        btn_clear_send.clicked.connect(self.send_edit.clear)
        btn_send = QPushButton("发送")
        btn_send.setMinimumHeight(40)
        btn_send.clicked.connect(self._send_data)
        send_side.addWidget(self.chk_hex_send)
        send_side.addWidget(self.chk_crlf)
        send_side.addWidget(btn_clear_send)
        send_side.addStretch()
        send_side.addWidget(btn_send)
        send_lay.addWidget(self.send_edit, 1)
        send_lay.addLayout(send_side)

        right_split.addWidget(recv_box)
        right_split.addWidget(send_box)
        right_split.setSizes([380, 220])

        main_lay.addWidget(left, 0)
        main_lay.addWidget(right_split, 1)

        self._refresh_ports()
        return root

    def _refresh_ports(self):
        self.device_combo.clear()
        if list_ports is None:
            self.device_combo.addItem("(未安装 pyserial)", None)
            return
        for p in list_ports.comports():
            label = f"{p.description} ({p.device})"
            self.device_combo.addItem(label, p.device)
        if self.device_combo.count() == 0:
            self.device_combo.addItem("(无可用串口)", None)

    def _current_device(self):
        return self.device_combo.currentData()

    def _open_settings_dialog(self):
        dlg = SerialSettingsDialog(self)
        dev = self._current_device()
        dlg.set_values(
            dev, self._baud, self._data_bits, self._stop_bits, self._parity, self._flow
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        device, baud, db, sb, par, flow = dlg.get_values()
        if device:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == device:
                    self.device_combo.setCurrentIndex(i)
                    break
            else:
                self._refresh_ports()
                for i in range(self.device_combo.count()):
                    if self.device_combo.itemData(i) == device:
                        self.device_combo.setCurrentIndex(i)
                        break
        self._baud = baud
        self._data_bits = db
        self._stop_bits = sb
        self._parity = par
        self._flow = flow
        bi = self.baud_combo.findData(baud)
        if bi >= 0:
            self.baud_combo.setCurrentIndex(bi)
        else:
            self.baud_combo.addItem(str(baud), baud)
            self.baud_combo.setCurrentIndex(self.baud_combo.count() - 1)

    def _serial_kwargs(self):
        kwargs = {
            "baudrate": self._baud,
            "bytesize": self._data_bits,
            "parity": self._parity,
            "stopbits": self._stop_bits,
            "timeout": 0,
        }
        if self._flow == "rtscts":
            kwargs["rtscts"] = True
        elif self._flow == "xonxoff":
            kwargs["xonxoff"] = True
        return kwargs

    def _toggle_serial(self):
        if self._ser is not None and self._ser.is_open:
            self._close_serial()
        else:
            self._open_serial()

    def _open_serial(self):
        if serial is None:
            QMessageBox.warning(self, "错误", "请先安装 pyserial：pip install pyserial")
            return
        dev = self._current_device()
        if not dev:
            QMessageBox.warning(self, "提示", "请选择有效串口设备。")
            return
        baud = self.baud_combo.currentData()
        if baud is None:
            try:
                baud = int(self.baud_combo.currentText())
            except ValueError:
                baud = 115200
        self._baud = baud
        try:
            self._ser = serial.Serial(dev, **self._serial_kwargs())
        except Exception as e:
            QMessageBox.critical(self, "打开失败", str(e))
            self._ser = None
            return
        self._device = dev
        self._apply_rts_dtr_from_ui()
        self.toggle_btn.setText("关闭设备")
        self.link_label.setPixmap(_make_link_icon(22, True).pixmap(22, 22))
        self._status.showMessage(f"串口 {dev} 已成功打开。")
        self._read_timer.start(30)

    def _close_serial(self):
        self._read_timer.stop()
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.close()
            except Exception:
                pass
            self._ser = None
        self.toggle_btn.setText("打开设备")
        self.link_label.setPixmap(_make_link_icon(22, False).pixmap(22, 22))
        self._status.showMessage("串口已关闭。")

    def _apply_rts_dtr_from_ui(self):
        if not self._ser or not self._ser.is_open:
            return
        try:
            self._ser.rts = self.chk_rts.isChecked()
            self._ser.dtr = self.chk_dtr.isChecked()
        except Exception:
            pass

    def _on_rts(self, _):
        self._apply_rts_dtr_from_ui()

    def _on_dtr(self, _):
        self._apply_rts_dtr_from_ui()

    def _poll_read(self):
        if not self._ser or not self._ser.is_open:
            return
        try:
            n = self._ser.in_waiting
            if n <= 0:
                return
            data = self._ser.read(n)
        except Exception as e:
            self._status.showMessage(f"读取错误: {e}")
            self._close_serial()
            return
        self._append_recv(data)

    def _append_recv(self, data: bytes):
        if self.chk_hex_show.isChecked():
            text = " ".join(f"{b:02X}" for b in data)
        else:
            text = data.decode("utf-8", errors="replace")
        if self.chk_timestamp.isChecked():
            ts = datetime.now().strftime("[%H:%M:%S.%f")[:-3] + "] "
            text = ts + text
        self.recv_edit.moveCursor(self.recv_edit.textCursor().End)
        self.recv_edit.insertPlainText(text)
        self.recv_edit.moveCursor(self.recv_edit.textCursor().End)

    def _parse_hex_send(self, s: str) -> bytes:
        parts = s.replace(",", " ").split()
        out = bytearray()
        for p in parts:
            p = p.strip()
            if not p:
                continue
            out.append(int(p, 16))
        return bytes(out)

    def _send_data(self):
        if not self._ser or not self._ser.is_open:
            QMessageBox.information(self, "提示", "请先打开串口。")
            return
        raw = self.send_edit.toPlainText()
        if self.chk_hex_send.isChecked():
            try:
                payload = self._parse_hex_send(raw)
            except ValueError:
                QMessageBox.warning(self, "格式错误", "HEX 发送格式应为空格分隔，如: 31 32 33")
                return
        else:
            payload = raw.encode("utf-8")
        if self.chk_crlf.isChecked():
            payload = payload + b"\r\n"
        try:
            self._ser.write(payload)
        except Exception as e:
            QMessageBox.critical(self, "发送失败", str(e))

    def closeEvent(self, event):
        self._close_serial()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
