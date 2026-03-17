import sys
import cv2
import time
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel,
    QHBoxLayout, QVBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

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


# ---- MAIN GUI ---- #
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animal Detection System")
        self.setGeometry(100, 100, 1000, 500)

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

        # Container
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

        # main layout
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.camera_label)
        main_layout.addWidget(right_widget)

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

        info_text = (
            f"Class: {animal_class.upper()}\n"
            f"Animal Conf: {animal_conf:.2f}\n"
            f"Nose Conf: {nose_conf:.2f}\n"
            f"{dist_info}\n"
            f"Status: {status_text}"
        )

        self.info_label.setText(info_text)

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