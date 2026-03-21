import sys
import cv2
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel,
    QHBoxLayout, QVBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import QScrollArea
from datetime import datetime

from utils.detection_model import detection
from utils.identification_model import identification


# ---- CAMERA WORKER THREAD ---- #
class CameraWorker(QThread):
    frame_signal = pyqtSignal(object)       # send frame to left side
    detection_signal = pyqtSignal(object)   # send detect history to right side

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # ---- DETECTION ---- #
            result = detection(frame)

            # send frame to show
            if result is not None:
                self.detection_signal.emit(result)
                self.frame_signal.emit(result["annotated_frame"])
            else:
                self.frame_signal.emit(frame)

            # delay detect (10 FPS)
            time.sleep(0.1)

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

# ---- CARD WIDGET LOGS ---- #
class DetectionCard(QWidget):
    def __init__(self, image, label, timestamp):
        super().__init__()

        self.setFixedSize(140, 150)
        self.setStyleSheet("""
            background-color: white;
            border-radius: 10px;
            border: 1px solid #ddd;
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # ---- Image ---- #
        img_label = QLabel()
        img_label.setFixedSize(124, 90)
        img_label.setAlignment(Qt.AlignCenter)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        qt_img = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            124, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        img_label.setPixmap(pixmap)

        # ---- Class Label ---- #
        class_label = QLabel(f"Class : {label.upper()}")
        class_label.setAlignment(Qt.AlignCenter)
        class_label.setStyleSheet("""
            font-size: 12px;
            font-weight: bold;
        """)

        # ---- Timestamp ---- #
        time_label = QLabel(f"Time : {timestamp}")
        time_label.setAlignment(Qt.AlignCenter)
        time_label.setStyleSheet("""
            font-size: 10px;
            font-weight: bold;
            background-color: #4CAF50;
            color: white;
            padding: 2px 6px;
            border-radius: 6px;
        """)

        layout.addWidget(img_label)
        layout.addWidget(class_label)
        layout.addWidget(time_label)

        self.setLayout(layout)


# ---- MAIN GUI ---- #
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animal Detection System")
        self.setGeometry(100, 100, 1000, 500)
        self.setFixedSize(1200, 850)

        self.last_detection = None

        # ---- LEFT: Camera View ---- #
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(640, 480)

        # ---- RIGHT: Cropped Face ---- #
        self.face_label = QLabel()
        self.face_label.setAlignment(Qt.AlignCenter)
        self.face_label.setFixedSize(300, 300)
        self.face_label.setStyleSheet("""
            border: 2px solid #cccccc;
            border-radius: 10px;
            background-color: white;
        """)

        self.info_label = QLabel("Waiting for detection...")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
        """)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setFixedHeight(200)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #f8f9fb;
                border-top: 1px solid #ddd;
            }
        """)

        # ---- CONTRAINER ---- #
        title_label = QLabel("Result")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        right_container = QVBoxLayout()
        right_container.addStretch()
        right_container.addWidget(title_label)
        right_container.addSpacing(20)
        right_container.addWidget(self.face_label, alignment=Qt.AlignCenter)
        right_container.addSpacing(15)
        right_container.addWidget(self.info_label, alignment=Qt.AlignCenter)
        right_container.addStretch()

        right_widget = QWidget()
        right_widget.setLayout(right_container)
        right_widget.setFixedWidth(350)

        # ---- TOP SECTION (Camera + Info) ---- #
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(10, 10, 10, 5)

        top_layout.addWidget(self.camera_label)
        top_layout.addWidget(right_widget)

        # ---- LOG SECTION ---- #
        self.log_title = QLabel("Detection History")
        self.log_title.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            color: #333; 
            margin-left: 5px;
            margin-bottom: 5px;
        """)

        self.log_scroll = QScrollArea()
        self.log_scroll.setFixedHeight(180)
        self.log_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.log_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.log_scroll.setWidgetResizable(True)

        self.log_container = QWidget()

        self.log_layout = QHBoxLayout()
        self.log_layout.setContentsMargins(10, 10, 10, 10)
        self.log_layout.setSpacing(15)
        self.log_layout.setAlignment(Qt.AlignLeft)
        self.log_layout.addStretch()

        self.log_container.setLayout(self.log_layout)
        self.log_scroll.setWidget(self.log_container)

        # ---- MAIN LAYOUT ---- #
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.log_title)
        main_layout.addWidget(self.log_scroll)

        self.setLayout(main_layout)

        # style
        self.setStyleSheet("""
            QWidget {
                background-color: #f4f6f9;
            }
        """)

        # ---- Start Camera Thread ---- #
        self.worker = CameraWorker()
        self.worker.frame_signal.connect(self.update_camera)
        self.worker.detection_signal.connect(self.update_detection)
        self.worker.start()

    # Update Left Panel (Live Camera)
    def update_camera(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.camera_label.setPixmap(pixmap)

    # Update Right Panel (Only When Detect)
    def update_detection(self, data):
        self.last_detection = data

        face_img = data["image"]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        h, w, ch = face_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(face_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.face_label.setPixmap(pixmap)

        animal_class = data["class"]
        animal_conf = data["confidence"]

        nose_conf = data["nose_data"]["confidence"]

        result = identification(data)
        
        if type(result) is tuple and len(result) == 3:
            status, face_dist, nose_dist = result
            if nose_dist == -1.0:
                dist_info = "Distances: First Animal"
            else:
                dist_info = f"Face Dist: {face_dist:.2f} | Nose Dist: {nose_dist:.2f}"
        else:
            status = result
            dist_info = "Distances: N/A"

        status_text = "KNOWN" if status else "NEW"
        if not status:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.add_log_card(face_img, animal_class, timestamp)

        info_text = (
            f"Class: {animal_class.upper()}\n"
            f"Animal Conf: {animal_conf:.2f}\n"
            f"Nose Conf: {nose_conf:.2f}\n"
            f"{dist_info}\n"
            f"Status: {status_text}"
        )

        self.info_label.setText(info_text)

    # Add Card Logs (Only When NEW Information)
    def add_log_card(self, img, label, timestamp):
        card = QWidget()
        card.setFixedSize(140, 140)
        card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #ddd;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        img = cv2.resize(img, (100, 80))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        img_label = QLabel()
        img_label.setPixmap(QPixmap.fromImage(qt_img))
        img_label.setAlignment(Qt.AlignCenter)

        text_label = QLabel(f"{label.upper()}\n{timestamp}")
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("""
            font-size: 10px;
            color: #333;
        """)

        layout.addWidget(img_label)
        layout.addWidget(text_label)

        card.setLayout(layout)

        self.log_layout.insertWidget(0, card)
        self.log_scroll.horizontalScrollBar().setValue(0)

    # Close Event
    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


# ---- RUN APP ---- #
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())