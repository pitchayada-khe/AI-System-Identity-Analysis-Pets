import cv2
from PyQt5.QtWidgets import QLabel, QFrame, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import Qt, QRectF

# ---- DECORATE ---- #
class AntialiasedLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._border_color = QColor("transparent") 
        self._border_width = 0
        self._corner_radius = 15
        self._bg_color = Qt.transparent

    def setBorder(self, color_str, width, radius):
        self._border_color = QColor(color_str)
        self._border_width = width
        self._corner_radius = radius
        self.update() 

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._border_color == QColor("transparent") or self._border_width == 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self._border_color)
        pen.setWidth(self._border_width)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QBrush(self._bg_color))
        rect = QRectF(self.rect())
        offset = self._border_width / 2.0
        rect.adjust(offset, offset, -offset, -offset)
        painter.drawRoundedRect(rect, self._corner_radius, self._corner_radius)

# ---- CARD WIDGET LOGS ---- #
class DetectionCard(QFrame):
    def __init__(self, image, label, timestamp):
        super().__init__()
        self.setFixedSize(140, 160)
        self.setObjectName("LogCard") 
        self.setStyleSheet("""
            #LogCard { background-color: white; border-radius: 12px; border: 2px solid #ffccd5; }
            #LogCard:hover { border: 2px solid #b3a1d9; }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        img_label = QLabel()
        scaled_pixmap = pixmap.scaled(120, 95, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label.setPixmap(scaled_pixmap)
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("border: none; background: transparent;")

        class_lbl = QLabel(f"Class : {label.upper()}")
        class_lbl.setAlignment(Qt.AlignCenter)
        class_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #333; border: none;")

        time_lbl = QLabel(f"Time : {timestamp}")
        time_lbl.setAlignment(Qt.AlignCenter)
        time_lbl.setStyleSheet("font-size: 10px; color: #555; border: none;")

        layout.addWidget(img_label, alignment=Qt.AlignCenter)
        layout.addWidget(class_lbl)
        layout.addWidget(time_lbl)
        self.setLayout(layout)