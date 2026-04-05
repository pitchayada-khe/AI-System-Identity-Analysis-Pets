import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QFrame, QScrollArea, QTabWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QPen, QColor, QBrush, QPainterPath
from PyQt5.QtCore import Qt, QRectF, QTimer
from datetime import datetime

from widgets import AntialiasedLabel, DetectionCard
from camera_thread import CameraWorker
from styles import MAIN_STYLE, get_result_table_html
from utils.identification_model import identification, clear_known_database
from utils.detection_model import detection

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Animal Detection System")
        self.setWindowIcon(QIcon("paw.png"))
        self.setFixedSize(1045, 840)

        self.last_detection = None

        # ---- LEFT: Mode Tabs ---- #
        self.tabs = QTabWidget()
        self.tabs.setFixedSize(660, 560)
        self.tabs.setAttribute(Qt.WA_TranslucentBackground)
        
        self.tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 2px solid #E5E0FF;
                border-radius: 15px;
                border-top-left-radius: 0px;
                background-color: #FAFAFF;
                top: -2px;
            }
                                
            QTabWidget > QWidget {
                background: transparent;
            }
            
            QTabBar::tab { 
                background: #E5E0FF; 
                border: 2px solid #E5E0FF; 
                border-bottom: 2px solid #E5E0FF; 
                border-top-left-radius: 10px; 
                border-top-right-radius: 10px; 
                min-width: 100px; 
                padding: 10px; 
                margin-right: 2px; 
                font-weight: bold; 
                color: #4B3FBC;
            }
                                
            QTabBar::tab:hover {
                background: #FCE7F3;
            }
            
            QTabBar::tab:selected { 
                background: #6C5CE7; 
                border: 2px solid #6C5CE7; 
                border-bottom: 2px solid #FAFAFF; 
                color: white;
                margin-bottom: -2px;       
            }
        """)

        # Camera Tab
        self.tab_camera = QWidget()
        self.tab_camera.setStyleSheet("background: transparent; border: none;")

        self.camera_label = QLabel() 
        self.camera_label.setAlignment(Qt.AlignCenter)

        cam_layout = QVBoxLayout()
        cam_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        self.tab_camera.setLayout(cam_layout)

        # Image Tab
        self.tab_image = QWidget()
        self.tab_image.setStyleSheet("background: transparent; border: none;")

        self.image_label = QLabel() 
        self.image_label.setAlignment(Qt.AlignCenter)

        self.btn_upload_img = QPushButton("CHOOSE YOUR IMAGE")
        self.btn_upload_img.setFixedSize(160, 40)
        self.btn_upload_img.setCursor(Qt.PointingHandCursor)
        self.btn_upload_img.setStyleSheet("""
            QPushButton { 
                background-color: #FF6B9A; 
                color: white; 
                border-radius: 20px; 
                font-weight: bold; 
                font-size: 12px;
                border: none;
            }
            QPushButton:hover { 
                background-color: #E05584; 
            }
        """)
        self.btn_upload_img.clicked.connect(self.upload_image)

        img_layout = QVBoxLayout()
        img_layout.addWidget(self.image_label, stretch=1, alignment=Qt.AlignCenter)
        img_layout.addWidget(self.btn_upload_img, alignment=Qt.AlignCenter)
        self.tab_image.setLayout(img_layout)

        # Video Tab
        self.tab_video = QWidget()
        self.tab_video.setStyleSheet("background: transparent; border: none;")

        self.video_label = QLabel() 
        self.video_label.setAlignment(Qt.AlignCenter)

        self.btn_upload_vid = QPushButton("CHOOSE YOUR VIDEO")
        self.btn_upload_vid.setFixedSize(160, 40)
        self.btn_upload_vid.setCursor(Qt.PointingHandCursor)
        self.btn_upload_vid.setStyleSheet("""
            QPushButton { 
                background-color: #FF6B9A; 
                color: white; 
                border-radius: 20px; 
                font-weight: bold; 
                font-size: 12px;
                border: none;
            }
            QPushButton:hover { 
                background-color: #E05584; 
            }
        """)
        self.btn_upload_vid.clicked.connect(self.upload_video)

        vid_layout = QVBoxLayout()
        vid_layout.addWidget(self.video_label, stretch=1, alignment=Qt.AlignCenter)
        vid_layout.addWidget(self.btn_upload_vid, alignment=Qt.AlignCenter)
        self.tab_video.setLayout(vid_layout)

        # 3 Tabs
        self.tabs.addTab(self.tab_camera, "LIVE CAMERA")
        self.tabs.addTab(self.tab_image, "IMAGE")
        self.tabs.addTab(self.tab_video, "VIDEO")

        self.tabs.currentChanged.connect(self.on_tab_changed)


        # ---- RIGHT: Result Panel ---- #
        self.result_card = QFrame()
        self.result_card.setFixedSize(361, 524)
        self.result_card.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 15px;
                border: 2px solid #E5E0FF;
            }
        """)
        
        title_label = QLabel("RESULT")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 22px; 
            font-weight: bold; 
            color: #4B3FBC; 
            margin-top: 3px;
            background-color: transparent;
            border: none;
        """)

        self.face_label = AntialiasedLabel()
        self.face_label.setAlignment(Qt.AlignCenter)
        self.face_label.setFixedWidth(270)
        self.face_label.setFixedHeight(270)
        self.face_label.setStyleSheet("border: none; background: transparent;")
        self.face_label.setBorder("#E5E0FF", 2, 15)
        self.face_label.hide()

        self.info_label = AntialiasedLabel()
        self.info_label.setFixedWidth(270)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setBorder("transparent", 0, 15)
        self.info_label.setText("")
        self.info_label.hide()

        self.status_label = QLabel("...WAITING...") 
        self.status_label.setFixedSize(140, 26)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 13px; 
                font-weight: bold; 
                border-radius: 13px; 
                border: none;
                color: #6B6B6B;
                background-color: #fefdfa; 
            }
        """)

        right_container = QVBoxLayout()
        right_container.setContentsMargins(15, 0, 15, 0)
        right_container.setSpacing(0)

        right_container.addStretch(1) 
        right_container.addWidget(title_label, alignment=Qt.AlignCenter)
        right_container.addSpacing(10) 
        right_container.addWidget(self.face_label, alignment=Qt.AlignCenter)
        right_container.addSpacing(10) 
        right_container.addWidget(self.info_label, alignment=Qt.AlignCenter)
        right_container.addSpacing(10) 
        right_container.addWidget(self.status_label, alignment=Qt.AlignCenter)
        right_container.addStretch(1)
        
        self.result_card.setLayout(right_container)

        right_wrapper = QVBoxLayout()
        right_wrapper.setContentsMargins(0, 35, 0, 0) 
        right_wrapper.addWidget(self.result_card)

        # ---- TOP SECTION (Camera + Result Panel) ---- #
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(15)

        top_layout.addWidget(self.tabs, stretch=2)
        top_layout.addLayout(right_wrapper, stretch=1)

        # ---- LOG SECTION ---- #
        self.history_frame = QFrame()
        self.history_frame.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 15px;
                border: 2px solid #E5E0FF;
            }
        """)
        
        history_layout = QVBoxLayout()
        history_layout.setContentsMargins(15, 15, 15, 15)
        history_layout.setSpacing(10)

        self.log_title = QLabel("DETECTION HISTORY")
        self.log_title.setAlignment(Qt.AlignCenter)
        self.log_title.setStyleSheet("""
            font-size: 16px; 
            font-weight: bold; 
            color: #4B3FBC; 
            background-color: transparent;
            border: none;
        """)

        self.log_scroll = QScrollArea()
        self.log_scroll.setFixedHeight(190)
        self.log_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.log_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.log_scroll.setWidgetResizable(True)
        self.log_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        self.log_container = QWidget()
        self.log_container.setStyleSheet("background-color: transparent;")

        self.log_layout = QHBoxLayout()
        self.log_layout.setContentsMargins(5, 5, 5, 10) 
        self.log_layout.setSpacing(15)
        self.log_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.log_layout.addStretch()

        self.log_container.setLayout(self.log_layout)
        self.log_scroll.setWidget(self.log_container)

        history_layout.addWidget(self.log_title)
        history_layout.addWidget(self.log_scroll)
        self.history_frame.setLayout(history_layout)

        # ---- MAIN LAYOUT ---- #
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.history_frame)

        self.setLayout(main_layout)

        self.setLayout(main_layout)
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #fefdfa;
            }
        """)

        # ---- Start Camera Thread ---- #
        self.on_tab_changed(0)

        # ---- Auto Clear History Timer ---- #
        self.clear_timer = QTimer(self)
        self.clear_timer.timeout.connect(self.clear_history_data)
        # Check every 10 minutes (600,000 ms)
        self.clear_timer.start(600000) 

    def clear_history_data(self):
        """Clear both the UI logs and the memory database for items older than threshold."""
        import time
        max_age_seconds = 10800 # 3 hours
        
        # 1. Clear UI history
        # Iterate backwards to safely remove items without messing up indices
        layout = self.log_layout
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget and hasattr(widget, 'created_at'):
                if time.time() - widget.created_at > max_age_seconds:
                    # Remove it from layout and delete
                    layout.takeAt(i)
                    widget.deleteLater()
                
        # 2. Clear known database in backend
        clear_known_database(max_age_seconds)

    # Update Left Panel
    def on_tab_changed(self, index):
        # Turn off the old input before changing the tab
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
        
        # If return to the camera tab (reopen the camera)
        if index == 0:
            self.start_camera_worker(0)

    def start_camera_worker(self, source):
        self.worker = CameraWorker(source)
        if source == 0:
            self.worker.frame_signal.connect(lambda frame: self.display_frame(frame, self.camera_label))
        else:
            self.worker.frame_signal.connect(lambda frame: self.display_frame(frame, self.video_label))
        self.worker.detection_signal.connect(self.update_detection)
        self.worker.start()

    def display_frame(self, frame, target_label):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        target_label.setPixmap(pixmap)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            frame = cv2.imread(file_path)
            if frame is not None:
                result = detection(frame)
                if result is not None:
                    self.update_detection(result)
                    self.display_frame(result["annotated_frame"], self.image_label)
                else:
                    self.display_frame(frame, self.image_label)

    def upload_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            # Close the old video (if any) and open the new video
            if hasattr(self, 'worker') and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait()
            
            self.video_label.clear() 
            self.start_camera_worker(file_path)

    # Update Right Panel (Only When Detect)
    def update_detection(self, data):
        self.last_detection = data
        self.face_label.show()
        self.info_label.show()

        # Image part
        face_img = data["image"]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        h, w, ch = face_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(face_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        display_size = 266
        scaled_pixmap = pixmap.scaled(display_size, display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        final_pixmap = QPixmap(270, 270)
        final_pixmap.fill(Qt.transparent)

        painter = QPainter(final_pixmap)
        painter.setRenderHint(QPainter.Antialiasing) 
        rect = QRectF(0, 0, 270, 270)
        border_radius = 15

        x_off = (270 - scaled_pixmap.width()) / 2
        y_off = (270 - scaled_pixmap.height()) / 2

        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        path.addRoundedRect(rect, border_radius, border_radius)
        painter.setClipPath(path)
        painter.drawPixmap(int(x_off), int(y_off), scaled_pixmap)
        
        painter.setClipping(False)
        pen = QPen(QColor("#E5E0FF")) 
        pen.setWidth(2) 
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect.adjusted(1,1,-1,-1), border_radius, border_radius) 
        painter.end()

        self.face_label.setPixmap(final_pixmap)

        # Info part
        self.info_label.setBorder("#E5E0FF", 2, 15)
        
        animal_class = data["class"].upper()
        animal_conf = f"{data['confidence']:.2f}"
        nose_conf = f"{data['nose_data']['confidence']:.2f}"

        result = identification(data)

        if type(result) is tuple and len(result) == 3:
            status, face_dist, nose_dist = result
            if nose_dist == -1.0:
                face_d = "FIRST DETECT"
                nose_d = "FIRST DETECT"
            else:
                face_d = f"{face_dist:.2f}"
                nose_d = f"{nose_dist:.2f}"
        else:
            status = result
            face_d = "N/A"
            nose_d = "N/A"

        info_html = f"""
        <table width='100%' cellspacing='0' cellpadding='6' 
               style='font-family: Segoe UI; border: none; text-align: center;'>
            <tr style='background-color: #FFFFFF;'>
                <td width='50%' style='color: #4B3FBC; text-align: center; font-weight: bold; border-bottom: 1px solid #E5E0FF; border-right: 1px solid #E5E0FF;'>CLASS</td>
                <td width='50%' style='color: #2D2D2D; text-align: center; border-bottom: 1px solid #E5E0FF;'>{animal_class}</td>
            </tr>
            <tr style='background-color: #F7F5FF;'>
                <td style='color: #4B3FBC; font-weight: bold; text-align: center; border-bottom: 1px solid #E5E0FF; border-right: 1px solid #E5E0FF;'>ANIMAL CONFIDENT</td>
                <td width='50%' style='color: #2D2D2D; text-align: center; border-bottom: 1px solid #E5E0FF;'>{animal_conf}</td>
            </tr>
            <tr style='background-color: #FFFFFF;'>
                <td style='color: #4B3FBC; font-weight: bold; text-align: center; border-bottom: 1px solid #E5E0FF; border-right: 1px solid #E5E0FF;'>NOSE CONFIDENT</td>
                <td width='50%' style='color: #2D2D2D; text-align: center; border-bottom: 1px solid #E5E0FF;'>{nose_conf}</td>
            </tr>
            <tr style='background-color: #F7F5FF;'>
                <td style='color: #4B3FBC; font-weight: bold; text-align: center; border-bottom: 1px solid #E5E0FF; border-right: 1px solid #E5E0FF;'>FACE DISTANCE</td>
                <td width='50%' style='color: #2D2D2D; text-align: center; border-bottom: 1px solid #E5E0FF;'>{face_d}</td>
            </tr>
            <tr style='background-color: #FFFFFF;'>
                <td style='color: #4B3FBC; font-weight: bold; text-align: center; border-right: 1px solid #E5E0FF;'>NOSE DISTANCE</td>
                <td width='50%' style='color: #2D2D2D; text-align: center;'>{nose_d}</td>
            </tr>
        </table>
        """
        self.info_label.setText(info_html)

        status_text = "KNOWN" if status else "UNKNOWN"
        status_bg_color = "#D1FAE5" if status else "#FFE4E6"
        status_text_color = "#065F46" if status else "#9F1239"

        self.status_label.setText(f"STATUS : {status_text}")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 13px; 
                font-weight: bold; 
                border-radius: 13px; 
                border: none;
                color: {status_text_color};
                background-color: {status_bg_color}; 
            }}
        """)

        if not status:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.add_log_card(face_img, animal_class, timestamp)

    # Add Card Logs
    def add_log_card(self, img, label, timestamp):
        card = DetectionCard(img, label, timestamp, is_new=True)
        self.log_layout.insertWidget(0, card)
        self.log_scroll.horizontalScrollBar().setValue(0)

    # Close Event
    def closeEvent(self, event):
        self.worker.stop()
        event.accept()


# ---- RUN APP ---- #
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QScrollBar:horizontal {
            border: none;
            background: #F7F5FF;
            height: 8px;
            margin: 0px;
        }
                      
        QScrollBar::handle:horizontal {
            background: #D6CCFF;
            min-width: 20px;
            border-radius: 4px;
        }
                      
        QScrollBar::handle:horizontal:hover {
            background: #F9A8D4; 
        }
                      
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())