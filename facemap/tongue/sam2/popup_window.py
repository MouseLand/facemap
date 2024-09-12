from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent, QPainter, QPainterPath, QColor
import numpy as np
from facemap import utils
from ..sam2.sam2_model import SAM2Model
import matplotlib.pyplot as plt

class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Dictionary to store click positions and labels for each frame
        self.frame_click_data = {}  # Structure: {frame_index: {"positions": [(x, y)], "labels": [1, 0]}}
        self.current_frame_index = 0  # Keep track of the current frame index

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            x = event.x()
            y = event.y()
            label_type = 1 if self.parent().mask_type_selection == self.parent().add_button else 0
            
            # Ensure there is an entry for the current frame before adding a click
            if self.current_frame_index not in self.frame_click_data:
                self.frame_click_data[self.current_frame_index] = {"positions": [], "labels": [], "pixmap_positions": []}

            self.store_click_for_frame(x, y, label_type)
            print(f"Clicked position: x={x}, y={y}, label={label_type} (frame {self.current_frame_index})")
            self.update()  # Repaint to reflect new click

    def store_click_for_frame(self, x, y, label_type):
        """Store click position and label for the current frame, converting to image pixel coordinates."""
        # Get the pixmap displayed in the QLabel
        pixmap = self.pixmap()
        if pixmap is None:
            return  # No image loaded, nothing to store

        # Get the actual size of the image in pixels
        image_width = self.parent().Lx
        image_height = self.parent().Ly
        print(self.parent().Lx, self.parent().Ly)

        # Get the size of the QLabel (display area)
        label_width = self.width()
        label_height = self.height()

        # Compute the scaling factors between QLabel size and actual image size
        scale_x = image_width / label_width
        scale_y = image_height / label_height

        # Convert the clicked position to image pixel coordinates
        image_x = int(x * scale_x)
        image_y = int(y * scale_y)

        # Store the converted coordinates
        if self.current_frame_index not in self.frame_click_data:
            self.frame_click_data[self.current_frame_index] = {"positions": [], "labels": [], "pixmap_positions": []}

        # Append the image pixel coordinates and the label type
        self.frame_click_data[self.current_frame_index]["positions"].append([image_x, image_y])
        self.frame_click_data[self.current_frame_index]["pixmap_positions"].append([x, y])
        self.frame_click_data[self.current_frame_index]["labels"].append(label_type)

        print(f"Stored click at image coordinates: ({image_x}, {image_y}) with label: {label_type}")

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.pixmap():
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            frame_data = self.frame_click_data.get(self.current_frame_index, {"positions": [], "labels": [], "pixmap_positions": []})
            for pos, mask in zip(frame_data["pixmap_positions"], frame_data["labels"]):
                x, y = pos
                size = 10  # Size of the star

                if mask == 1:
                    painter.setPen(QColor(0, 255, 0))  # Green for positive clicks
                    painter.setBrush(QColor(0, 255, 0))
                else:
                    painter.setPen(QColor(255, 0, 0))  # Red for negative clicks
                    painter.setBrush(QColor(255, 0, 0))

                self.draw_star(painter, x, y, size)


    def draw_star(self, painter, x, y, size):
        # Draw a simple star shape
        points = []
        for i in range(5):
            angle = i * 72
            x_offset = size * np.cos(np.radians(angle))
            y_offset = size * np.sin(np.radians(angle))
            points.append(QPoint(int(x + x_offset), int(y - y_offset)))  # Convert to int

        path = QPainterPath()
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        path.closeSubpath()

        painter.drawPath(path)
    
    def show_mask(self, mask, obj_id=None, random_color=False):
        """Display a mask on the current frame, scaled to QLabel size."""
        if random_color:
            color = np.concatenate([np.random.random(3) * 255, np.array([0.6 * 255])], axis=0).astype(int)
        else:
            # Use a predefined color map (e.g., tab10) to select a color based on obj_id
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id  # Loop over available colors
            color = np.array([*cmap(cmap_idx)[:3], 0.6]) * 255  # Convert to RGB (0-255) and set transparency

        color = QColor(int(color[0]), int(color[1]), int(color[2]), int(color[3]))  # Set color with alpha

        # Convert the mask to a format that can be displayed
        h, w = mask.shape[-2:]
        mask_image = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA image
        mask_image[:, :, 0:3] = mask.reshape(h, w, 1) * np.array([color.red(), color.green(), color.blue()])
        mask_image[:, :, 3] = mask * color.alpha()  # Set alpha channel based on the mask and color's alpha

        # Convert the NumPy array into QImage and QPixmap
        qimage = QImage(mask_image.data, mask_image.shape[1], mask_image.shape[0], mask_image.strides[0], QImage.Format_RGBA8888)
        mask_pixmap = QPixmap.fromImage(qimage)

        # Scale the mask to match the QLabel size
        label_width = self.width()
        label_height = self.height()
        scaled_mask_pixmap = mask_pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Create a new pixmap with the current image and draw the mask on it
        if self.pixmap():
            # Scale the original pixmap to match the QLabel size
            original_pixmap = self.pixmap().scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            combined_pixmap = QPixmap(label_width, label_height)
            combined_pixmap.fill(Qt.transparent)  # Start with a transparent background

            painter = QPainter(combined_pixmap)
            painter.drawPixmap(0, 0, original_pixmap)  # Draw the scaled original image
            painter.drawPixmap(0, 0, scaled_mask_pixmap)  # Overlay the scaled mask
            painter.end()

            # Update the QLabel to show the new image with the mask
            self.setPixmap(combined_pixmap)


class Sam2Popup(QDialog):
    def __init__(self, parent=None, cumframes=[], Ly=[], Lx=[], containers=None):
        super().__init__(parent)
        self.parent = parent
        self.cumframes = cumframes
        self.Ly = Ly
        self.Lx = Lx
        self.containers = containers
        self.current_frame_index = 0  # Index of the currently displayed frame
        video_path = self.parent.video_filenames[-1]
        print(f"Video Path: {video_path}")
        self.sam2 = SAM2Model(video_path)

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

    def add_points_and_masks(self):
        # Get the masks and points from the frame_click_data
        for frame_idx, data in self.visual_area.frame_click_data.items():
            if len(data["positions"]) > 0:
                points = np.array(data["positions"], dtype=np.float32)
                labels = np.array(data["labels"], np.int32)
                # pass the click positions, labels and frame idx to the SAM2 model. data is used as follows
                ann_frame_idx = frame_idx
                ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
                _, out_obj_ids, out_mask_logits = self.sam2.predictor.add_new_points_or_box(
                    inference_state=self.sam2.inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
                # show the mask on the current frame
                mask = out_mask_logits[0, 0].cpu().numpy()
                self.visual_area.show_mask(mask, obj_id=ann_obj_id)


    def track_objects(self):
        self.add_points_and_masks()
        # Create a SAM2Model object and pass the video path
        """
        if self.sam2 is None:
            video_path = self.parent.video_filenames[-1]
            print(f"Video Path: {video_path}")
            self.sam2 = SAM2Model(video_path)
        # Perform object tracking using the SAM2 model
            
        # pass the click positions, labels and frame idx to the SAM2 model. data is used as follows
        ann_frame_idx = 345  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # Let's add a positive click at (x, y) = (210, 350) to get started
        points = np.array([[100, 200]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1], np.int32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        self.sam2.track_objects() 
        """

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

    def update_frame_display(self):
        # Fetch the frame based on current index and update QLabel
        frame = self.get_frame(self.current_frame_index)
        if frame is not None:
            # Convert frame data to QImage and then to QPixmap
            qimage = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.visual_area.setPixmap(pixmap.scaled(self.visual_area.size(), Qt.KeepAspectRatio))
            
            # Update the slider to match the current frame index
            self.frame_slider.setValue(self.current_frame_index)

            # Instead of calling set_frame_index, just set the current frame index on the visual area
            self.visual_area.current_frame_index = self.current_frame_index

    def slider_changed(self, value):
        # Update frame index from slider and refresh display
        self.current_frame_index = value
        self.update_frame_display()

    def get_frame(self, index):
        frame = utils.get_frame(index, self.cumframes[-1], self.cumframes, self.containers)[0].squeeze()
        return frame
