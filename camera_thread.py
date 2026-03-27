import cv2
import time
from PyQt5.QtCore import QThread, pyqtSignal
from utils.detection_model import detection

# ---- CAMERA WORKER THREAD ---- #
class CameraWorker(QThread):
    frame_signal = pyqtSignal(object) # send frame to left side
    detection_signal = pyqtSignal(object) # send detect history to right side

    def __init__(self, video_soruce=0):
        super().__init__()
        self.running = True
        self.video_source = video_soruce


    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.video_source, str):
                    break
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