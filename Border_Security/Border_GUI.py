"""
UPDATED — FULLY WORKING DASHBOARD WITH AUTO-RECONNECT SERIAL & YOLOv8 (2025)
✔ PyQt5 dark-dashboard GUI
✔ YOLOv8 detection with bounding boxes
✔ Alarm on detection (loops until stopped)
✔ Arduino commands sent reliably
✔ Serial connection auto-reconnect
✔ Structured alerts with timestamp, source, and type
✔ Non-blocking alerts & single command trigger
"""

import sys
import time
import cv2
import numpy as np
import serial
import serial.tools.list_ports
import pygame
from ultralytics import YOLO
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout,
    QHBoxLayout, QTextEdit, QMessageBox, QComboBox, QFormLayout, QSlider, QFrame
)

# ---------------- Styles -----------------
APP_STYLE = """
QWidget { background: #111218; color: #e6eef6; font-family: 'Segoe UI', Roboto; }
QLabel#title { font-size: 18px; font-weight: bold; color: white; }
QFrame.card { background: #151826; border-radius: 10px; padding: 10px; }
QLineEdit, QComboBox, QTextEdit { background: #0f1220; border: 1px solid #232634; border-radius: 6px; padding: 6px; }
QPushButton { background: #0ea5a4; color: #031217; border-radius: 8px; padding: 8px; font-weight: 600; }
QPushButton#danger { background: #ef4444; color: white; }
QTextEdit { min-height: 200px; }
"""

# ---------------- YOLO Detection Thread -----------------
class VideoDetectThread(QThread):
    frame_updated = pyqtSignal(np.ndarray)
    detection_alert = pyqtSignal(str)
    no_detection = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, stream_url, target_label, model='yolov8n.pt', conf=0.35):
        super().__init__()
        self.stream_url = stream_url
        self.target_label = target_label
        self.model_path = model
        self.conf = conf
        self.model = None
        self._running = True

    def run(self):
        try:
            self.status.emit("Loading YOLO model...")
            self.model = YOLO(self.model_path)
            self.status.emit("Model loaded successfully.")
        except Exception as e:
            self.status.emit(f"ERROR loading model: {e}")
            return

        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            self.status.emit("ERROR: Failed to open camera stream.")
            return

        last_time = time.time()
        no_target_count = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.status.emit("Stream lost — reconnecting...")
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(self.stream_url)
                continue

            scaled = cv2.resize(frame, (640, 480))
            alerted = False

            try:
                results = self.model(scaled, conf=self.conf)
                if results and len(results) > 0:
                    r = results[0]
                    names = r.names
                    for box in r.boxes:
                        cls = int(box.cls)
                        label = names.get(cls, str(cls))
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        sx = frame.shape[1] / 640
                        sy = frame.shape[0] / 480
                        X1, Y1 = int(x1 * sx), int(y1 * sy)
                        X2, Y2 = int(x2 * sx), int(y2 * sy)

                        cv2.rectangle(frame, (X1, Y1), (X2, Y2), (14,165,164), 2)
                        cv2.putText(frame, label, (X1, Y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240,240,255), 2)

                        if label.lower() == self.target_label.lower():
                            alerted = True
                            no_target_count = 0
                            break
            except Exception as e:
                self.status.emit(f"Detection error: {e}")

            now = time.time()
            fps = 1 / (now - last_time)
            last_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

            if alerted:
                self.detection_alert.emit(self.target_label)
            else:
                no_target_count += 1
                if no_target_count >= 10:
                    self.no_detection.emit(self.target_label)
                    no_target_count = 0

            self.frame_updated.emit(frame)

        cap.release()

    def stop(self):
        self._running = False
        self.wait()

# ---------------- Serial Connection with Auto-Reconnect -----------------
class SerialReaderThread(QThread):
    ultrasonic_alert = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, port, baud=115200, retry_delay=2):
        super().__init__()
        self.port = port
        self.baud = baud
        self.retry_delay = retry_delay
        self._running = True
        self.ser = None

    def run(self):
        while self._running:
            if self.ser is None or not self.ser.is_open:
                try:
                    self.ser = serial.Serial(self.port, self.baud, timeout=1)
                    time.sleep(2)
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                    self.status_update.emit(f"Serial connected: {self.port}")
                except Exception as e:
                    self.status_update.emit(f"Serial connection failed: {e}")
                    time.sleep(self.retry_delay)
                    continue

            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        self.status_update.emit(line)
                        if any(word in line.upper() for word in ["ULTRA", "ALERT", "DETECT"]):
                            self.ultrasonic_alert.emit(line)
            except Exception as e:
                self.status_update.emit(f"Serial error: {e}")
                if self.ser and self.ser.is_open:
                    self.ser.close()
                    self.ser = None
                    self.status_update.emit("Serial disconnected. Retrying...")

            time.sleep(0.05)

    def stop(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.wait()

    def send(self, cmd):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((cmd.strip() + "\r\n").encode())
                self.ser.flush()
                self.status_update.emit(f"Sent: {cmd}")
            except Exception as e:
                self.status_update.emit(f"Failed to send command: {e}")
        else:
            self.status_update.emit("Serial not connected. Will auto-reconnect.")

# ---------------- GUI Dashboard -----------------
class Dashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Border Surveillance System")
        self.setGeometry(80, 40, 1350, 780)
        self.setStyleSheet(APP_STYLE)

        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound("alarm.mp3")
        self.alarm_active = False
        self.target_alerted = False

        self.init_ui()

        self.vthread = None
        self.sthread = None

    def init_ui(self):
        # Left panel
        left = QFrame()
        left.setProperty("class", "card")
        left.setMinimumWidth(380)
        left_layout = QVBoxLayout(left)

        title = QLabel("Detection Panel")
        title.setObjectName("title")
        left_layout.addWidget(title)

        cam_box = QFrame()
        cam_box.setProperty("class", "card")
        cam_form = QFormLayout(cam_box)
        self.stream_input = QLineEdit("http://192.168.209.139:8080/video")
        self.model_input = QLineEdit("yolov8n.pt")
        self.target_label = QLineEdit("person")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setValue(35)
        cam_form.addRow("Camera URL:", self.stream_input)
        cam_form.addRow("Model:", self.model_input)
        cam_form.addRow("Target:", self.target_label)
        cam_form.addRow("Confidence:", self.conf_slider)
        left_layout.addWidget(cam_box)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Detection")
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.setObjectName("danger")
        self.stop_alarm_btn = QPushButton("Stop Alarm")
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.stop_alarm_btn)
        left_layout.addLayout(btn_row)

        # Serial
        ser_box = QFrame()
        ser_box.setProperty("class", "card")
        ser_form = QFormLayout(ser_box)
        self.port_combo = QComboBox()
        self.refresh_ports()
        self.baud_input = QLineEdit("9600")
        ser_form.addRow("Port:", self.port_combo)
        ser_form.addRow("Baud:", self.baud_input)
        left_layout.addWidget(ser_box)

        ser_btns = QHBoxLayout()
        self.ser_connect = QPushButton("Connect Serial")
        self.ser_disconnect = QPushButton("Disconnect")
        ser_btns.addWidget(self.ser_connect)
        ser_btns.addWidget(self.ser_disconnect)
        left_layout.addLayout(ser_btns)

        cmd_row = QHBoxLayout()
        self.cmd_input = QLineEdit()
        self.cmd_send = QPushButton("Send")
        cmd_row.addWidget(self.cmd_input)
        cmd_row.addWidget(self.cmd_send)
        left_layout.addLayout(cmd_row)

        log_box = QFrame()
        log_box.setProperty("class", "card")
        log_layout = QVBoxLayout(log_box)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        log_layout.addWidget(QLabel("Event Log:"))
        log_layout.addWidget(self.log)
        left_layout.addWidget(log_box)
        left_layout.addStretch()

        # Right panel
        right = QFrame()
        right.setProperty("class", "card")
        right_layout = QVBoxLayout(right)
        self.video_label = QLabel("No Video")
        self.video_label.setMinimumSize(800, 500)
        self.video_label.setStyleSheet("background:#000;border-radius:8px;")
        right_layout.addWidget(self.video_label)

        main = QHBoxLayout()
        main.addWidget(left)
        main.addWidget(right, 1)
        self.setLayout(main)

        # ---------------- Connections ----------------
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_alarm_btn.clicked.connect(self.stop_alarm)
        self.ser_connect.clicked.connect(self.connect_serial)
        self.ser_disconnect.clicked.connect(self.disconnect_serial)
        self.cmd_send.clicked.connect(self.send_command)

    # ---------------- Helpers -----------------------
    def refresh_ports(self):
        self.port_combo.clear()
        for p in serial.tools.list_ports.comports():
            self.port_combo.addItem(p.device)

    def append_log(self, msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    def structured_alert(self, source, alert_type, message=""):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{ts}] [{source}] [{alert_type}] {message}"
        self.append_log(formatted)
        icon = QMessageBox.Warning if alert_type.upper() in ["DETECTED", "ALERT"] else QMessageBox.Information
        alert_box = QMessageBox(self)
        alert_box.setWindowTitle(f"{alert_type} from {source}")
        alert_box.setText(formatted)
        alert_box.setIcon(icon)
        alert_box.setStandardButtons(QMessageBox.Ok)
        alert_box.setModal(False)
        alert_box.show()
        QApplication.beep()

    # ---------------- Detection ----------------------
    def start_detection(self):
        if self.vthread:
            self.append_log("Detection already running.")
            return

        url = self.stream_input.text().strip()
        model = self.model_input.text()
        target = self.target_label.text().strip()
        conf = max(0.05, min(0.99, self.conf_slider.value() / 100))

        self.vthread = VideoDetectThread(url, target, model, conf)
        self.vthread.frame_updated.connect(self.on_frame)
        self.vthread.status.connect(self.append_log)
        self.vthread.detection_alert.connect(self.on_detection_alert)
        self.vthread.no_detection.connect(self.on_no_detection)
        self.vthread.start()
        self.append_log("Detection started.")

    def stop_detection(self):
        if self.vthread:
            self.vthread.stop()
            self.vthread = None
            self.append_log("Detection stopped.")
            self.target_alerted = False

    @pyqtSlot(np.ndarray)
    def on_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pix)

    @pyqtSlot(str)
    def on_detection_alert(self, lbl):
        if not self.target_alerted:
            self.target_alerted = True
            self.alarm_active = True
            self.alarm_sound.play(loops=-1)
            self.structured_alert("DETECTED", lbl)

            if self.sthread:
                self.sthread.send("g")

    @pyqtSlot(str)
    def on_no_detection(self, lbl):
        self.append_log(f"{lbl} NOT detected.")
        self.target_alerted = False
        if self.sthread:
            self.sthread.send("l")

    # ---------------- Alarm control ----------------
    def stop_alarm(self):
        if self.alarm_active:
            self.alarm_sound.stop()
            self.alarm_active = False
            self.append_log("Alarm manually stopped.")

    # ---------------- Serial connection ---------------- 
    def connect_serial(self):
        if self.sthread:
            self.append_log("Serial already running.")
            return
        port = self.port_combo.currentText()
        baud = int(self.baud_input.text())
        self.sthread = SerialReaderThread(port, baud, retry_delay=2)
        self.sthread.status_update.connect(self.append_log)
        self.sthread.ultrasonic_alert.connect(self.on_ultra_alert)
        self.sthread.start()
        self.append_log("Serial thread started with auto-reconnect.")

    def disconnect_serial(self):
        if self.sthread:
            self.sthread.stop()
            self.sthread = None
            self.append_log("Serial disconnected.")

    @pyqtSlot(str)
    def on_ultra_alert(self, msg):
        self.structured_alert("ULTRASONIC", "ALERT", msg)

    def send_command(self):
        cmd = self.cmd_input.text().strip()
        if self.sthread:
            self.sthread.send(cmd)
        else:
            self.append_log("Serial not connected. Will auto-reconnect if started.")

# ---------------- Main -----------------
def main():
    app = QApplication(sys.argv)
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
