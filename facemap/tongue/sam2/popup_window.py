from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent, QPainter, QPainterPath, QColor
import numpy as np
from facemap import utils

class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.click_positions = []  # List to store positions where clicks occurred
        self.click_positions_mask = []  # List to store label type for each click
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()
            self.click_positions.append((x, y))
            print(f"Clicked position: x={x}, y={y}")
            if self.parent().mask_type_selection == self.parent().add_button:
                self.click_positions_mask.append(1)
            else:
                self.click_positions_mask.append(0)
            self.update()  # Request a repaint to show the star

    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.pixmap():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            for pos, mask in zip(self.click_positions, self.click_positions_mask):
                x, y = pos
                size = 10  # Size of the star
                
                if mask == 1:
                    painter.setPen(QColor(0, 255, 0))  # Green color for the star
                    painter.setBrush(QColor(0, 255, 0))  # Fill color for the star
                else:
                    painter.setPen(QColor(255, 0, 0))  # Red color for the star
                    painter.setBrush(QColor(255, 0, 0))  # Fill color for the star
                
                self.draw_star(painter, x, y, size)

    def draw_star(self, painter, x, y, size):
        # Draw a simple star shape
        points = []
        for i in range(5):
            angle = i * 72
            x_offset = size * np.cos(np.radians(angle))
            y_offset = size * np.sin(np.radians(angle))
            points.append(QPoint(x + x_offset, y - y_offset))

        path = QPainterPath()
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        path.closeSubpath()

        painter.drawPath(path)

class Sam2Popup(QDialog):
    def __init__(self, parent=None, cumframes=[], Ly=[], Lx=[], containers=None):
        super().__init__(parent)

        self.cumframes = cumframes
        self.Ly = Ly
        self.Lx = Lx
        self.containers = containers
        self.current_frame_index = 0  # Index of the currently displayed frame

        # Set window title and size
        self.setWindowTitle("SAM2 Segmentation Options")
        self.window_max_size = QDesktopWidget().screenGeometry(-1)

        # Main layout with two sections (left and right)
        main_layout = QHBoxLayout(self)

        # Left section: Object list and controls
        left_layout = QVBoxLayout()
        
        self.object_label = QLabel("Object 1")
        self.object_label.setStyleSheet("color: lightgrey;")
        left_layout.addWidget(self.object_label)

        # Object Thumbnail (Placeholder for now)
        self.object_thumbnail = QLabel()
        self.object_thumbnail.setFixedSize(100, 100)
        self.object_thumbnail.setStyleSheet("background-color: #333333; border: 1px solid grey;")
        left_layout.addWidget(self.object_thumbnail)

        # Add/Remove Buttons
        self.button_group = QButtonGroup()
        self.add_button = QPushButton("Add")
        self.remove_button = QPushButton("Remove")
        self.add_button.setStyleSheet("background-color: #007BFF; color: white; border: 2px solid #0056b3;")
        self.remove_button.setStyleSheet("background-color: #FF3333; color: white; border: 2px solid #cc0000;")
        self.button_group.addButton(self.add_button)
        self.button_group.addButton(self.remove_button)
        self.button_group.setExclusive(True)
        left_layout.addWidget(self.add_button)
        left_layout.addWidget(self.remove_button)

        # Connect button group signal
        self.button_group.buttonClicked.connect(self.update_button_styles)

        # Set initial selection
        self.mask_type_selection = self.add_button
        self.update_button_styles(self.mask_type_selection)

        # Button to add more objects
        self.add_object_button = QPushButton("Add another object")
        left_layout.addWidget(self.add_object_button)

        # Spacer to fill up the space
        left_layout.addStretch(1)

        # Start over and Track buttons
        self.start_over_button = QPushButton("Start over")
        self.track_objects_button = QPushButton("Track objects")
        self.track_objects_button.setStyleSheet("background-color: #6A00FF; color: white;")

        left_layout.addWidget(self.start_over_button)
        left_layout.addWidget(self.track_objects_button)

        # Right section: Visual area (Placeholder for image/video frame)
        right_layout = QVBoxLayout()
        self.setup_visual_area(right_layout)

        # Add both layouts to the main layout
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # Connect buttons to their respective methods
        self.add_button.clicked.connect(self.add_area)
        self.remove_button.clicked.connect(self.remove_area)
        self.track_objects_button.clicked.connect(self.track_objects)

        self.update_window_size()

    def update_button_styles(self, button):
        # Reset styles for both buttons
        self.add_button.setStyleSheet("background-color: #007BFF; color: white; border: 2px solid #0056b3;")
        self.remove_button.setStyleSheet("background-color: #FF3333; color: white; border: 2px solid #cc0000;")

        # Highlight the selected button
        if button == self.add_button:
            self.add_button.setStyleSheet("background-color: #0056b3; color: white; border: 2px solid #003d7a;")
        else:
            self.remove_button.setStyleSheet("background-color: #cc0000; color: white; border: 2px solid #990000;")

        # Update current selection
        self.mask_type_selection = button

    def add_area(self):
        # Trigger button click to handle selection
        self.button_group.buttonClicked.emit(self.add_button)

    def remove_area(self):
        # Trigger button click to handle selection
        self.button_group.buttonClicked.emit(self.remove_button)
        
    def track_objects(self):
        # Logic for tracking objects (to be implemented)
        print("Tracking objects...")

    def update_window_size(self, frac=0.5, aspect_ratio=1.0):
        # Set the size of the window to be a fraction of the screen size using the aspect ratio
        self.resize(
            int(np.floor(self.window_max_size.width() * frac)),
            int(np.floor(self.window_max_size.height() * frac * aspect_ratio)),
        )
    
    def setup_visual_area(self, layout):
        # Visual area for displaying the frame
        self.visual_area = ClickableLabel()
        self.visual_area.setStyleSheet("background-color: black; border: 2px solid #00AEEF;")
        self.visual_area.setAlignment(Qt.AlignCenter)
        # get size of right layout
        visual_area_size = self.visual_area.size()
        width = visual_area_size.width()
        height = visual_area_size.height()
        print(f"Visual Area Size: Width={width}, Height={height}")
        self.visual_area.setFixedSize(width*3, height*2)
        layout.addWidget(self.visual_area)

        # Frame navigation slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.cumframes[-1])
        self.frame_slider.valueChanged.connect(self.slider_changed)
        # change style of slider to be visible in dark mode
        self.frame_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                border: 1px solid #3A3A3A;
                height: 10px;
                background: #333333;
                margin: 0px;
            }
            QSlider::handle:horizontal {
                background: #00AEEF;
                border: 1px solid #00AEEF;
                width: 10px;
                margin: -5px 0;
                border-radius: 5px;
            }
            """
        )
        layout.addWidget(self.frame_slider)

        # Load the first frame initially
        self.update_frame_display()

    def load_next_frame(self):
        # Increment frame index and update display
        if self.current_frame_index < self.cumframes[-1]:
            self.current_frame_index += 1
            self.update_frame_display()

    def slider_changed(self, value):
        # Update frame index from slider and refresh display
        self.current_frame_index = value
        self.update_frame_display()

    def update_frame_display(self):
        # Fetch the frame based on current index and update QLabel
        frame = self.get_frame(self.current_frame_index)
        if frame is not None:
            # Convert frame data to QImage and then to QPixmap
            qimage = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.visual_area.setPixmap(pixmap.scaled(self.visual_area.size(), Qt.KeepAspectRatio))
            self.frame_slider.setValue(self.current_frame_index)

    def get_frame(self, index):
        frame = utils.get_frame(index, self.cumframes[-1], self.cumframes, self.containers)[0].squeeze()
        return frame
