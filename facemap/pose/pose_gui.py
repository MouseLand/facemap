"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import numpy as np
import pyqtgraph as pg
from matplotlib import cm
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from facemap import utils
from facemap.pose import pose

from . import transforms

"""
Pose subclass for generating obtaining bounding box from user input.
Currently supports single video processing only.
"""


class PoseGUI(pose.Pose):
    def __init__(self, gui=None, img_xy=(256, 256)):
        self.gui = gui
        super(PoseGUI, self).__init__(gui=self.gui)
        self.bbox_set = False
        self.bbox = []
        self.resize = False
        self.add_padding = False
        self.img_xy = img_xy
        self.cancel_bbox_selection = False

    # Draw box on GUI using user's input
    def draw_user_bbox(self):
        """
        Function for user to draw a bbox
        """
        # Get sample frame from each video in case of multiple videos
        frame_index = self.gui.cframe  # use current frame for drawing bbox
        sample_frame = utils.get_frame(
            frame_index, self.nframes, self.cumframes, self.containers
        )
        last_video = False
        for video_id, frame in enumerate(sample_frame):
            # Trigger new window for ROI selection of each frame
            if video_id == len(sample_frame) - 1:
                last_video = True
            ROI_popup(frame, video_id, self.gui, self, last_video)
        return self.bbox, self.bbox_set, self.resize, self.add_padding

    def plot_bbox_roi(self):
        # Plot bbox on GUI
        for i, bbox in enumerate(self.bbox):
            y1, y2, x1, x2 = bbox
            dy, dx = y2 - y1, x2 - x1
            xrange = np.arange(x1 + self.gui.sx[i], x2 + self.gui.sx[i]).astype(
                np.int32
            )
            yrange = np.arange(y1 + self.gui.sy[i], y2 + self.gui.sy[i]).astype(
                np.int32
            )
            y1, x1 = yrange[0], xrange[0]
            self.gui.add_ROI(
                roitype=4 + 1,
                roistr="bbox_{}".format(i),
                moveable=False,
                resizable=False,
                pos=(y1, x1, dy, dx),
                ivid=i,
                yrange=yrange,
                xrange=xrange,
            )
        self.bbox_set = True

    def adjust_bbox(self):
        # This function adjusts bbox so that it is of minimum dimension: 256, 256
        sample_frame = utils.get_frame(0, self.nframes, self.cumframes, self.containers)
        for i, bbox in enumerate(self.bbox):
            x1, x2, y1, y2 = bbox
            dy, dx = y2 - y1, x2 - x1
            if dx != dy:  # If bbox is not square then add padding to image
                self.add_padding = True
            if dy != 256 or dx != 256:  # If bbox is not 256, 256 then resize image
                self.resize = True
            self.bbox[i] = x1, x2, y1, y2
            """
            larger_dim = max(dx, dy)
            if larger_dim < self.img_xy[0] or larger_dim < self.img_xy[1]:
                # If the largest dimension of the image is smaller than the minimum required dimension,
                # then resize the image to the minimum dimension
                self.resize = True
            else:
                # If the largest dimension of the image is larger than the minimum required dimension,
                # then crop the image to the minimum dimension
                (x1, x2, y1, y2, self.resize,) = transforms.get_crop_resize_params(
                    sample_frame[i],
                    x_dims=(x1, x2),
                    y_dims=(y1, y2),
                )
                self.bbox[i] = x1, x2, y1, y2
            """
            print("BBOX after adjustment:", self.bbox)
            print("RESIZE:", self.resize)
            print("PADDING:", self.add_padding)


class ROI_popup(QDialog):
    def __init__(self, frame, video_id, gui, pose, last_video):
        super().__init__()
        window_max_size = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        fraction = 0.5
        aspect_ratio = 1.5
        self.resize(
            int(np.floor(window_max_size.width() * fraction)),
            int(np.floor(window_max_size.height() * fraction * aspect_ratio)),
        )
        self.gui = gui
        self.frame = frame.squeeze()
        self.pose = pose
        self.last_video = last_video
        self.setWindowTitle("Select face ROI for video: " + str(video_id))

        # Add image and ROI bbox
        self.verticalLayout = QVBoxLayout(self)
        self.win = pg.GraphicsLayoutWidget()
        self.win.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.win.setObjectName("Dialog " + str(video_id + 1))
        # fix image in ROI window
        ROI_win = self.win.addViewBox(invertY=True, lockAspect=True, enableMouse=False)
        self.img = pg.ImageItem(self.frame)
        ROI_win.addItem(self.img)
        shape_y, shape_x = self.frame.shape[0], self.frame.shape[1]
        ROI_win.setRange(xRange=[0, shape_x], yRange=[0, shape_y])
        self.roi = pg.RectROI(
            [0, 0],
            [int(np.floor(0.6 * shape_x)), int(np.floor(0.5 * shape_y))],
            pen=pg.mkPen("r", width=2),
            movable=True,
            resizable=True,
            maxBounds=QtCore.QRectF(0, 0, shape_x, shape_y),
        )
        ROI_win.addItem(self.roi)
        self.win.show()
        self.verticalLayout.addWidget(self.win)

        # Add buttons to dialog box
        self.done_button = QPushButton("Done")
        self.done_button.setDefault(True)
        self.done_button.clicked.connect(self.done_exec)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_exec)
        # Add a next button to the dialog box horizontally centered with cancel button and done button
        self.next_button = QPushButton("Next")
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self.next_exec)
        # Add a skip button to the dialog box horizontally centered with cancel button and done button
        self.skip_button = QPushButton("Skip (no ROI)")
        self.skip_button.setDefault(True)
        self.skip_button.clicked.connect(self.skip_exec)

        # Position buttons
        self.widget = QWidget(self)
        self.horizontalLayout = QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.addWidget(self.cancel_button)
        self.horizontalLayout.addWidget(self.skip_button)
        if self.last_video:
            self.horizontalLayout.addWidget(self.done_button)
        else:
            self.horizontalLayout.addWidget(self.next_button)
        self.verticalLayout.addWidget(self.widget)

        self.exec_()

    def get_coordinates(self):
        print("self.frame shape:", self.frame.shape)
        print("self.img shape:", self.img.shape)
        roi_tuple, _ = self.roi.getArraySlice(self.frame, self.img, returnSlice=False)
        print("roi_tuple:", roi_tuple)
        (x1, x2), (y1, y2) = roi_tuple[0], roi_tuple[1]
        return (x1, x2), (y1, y2)

    def skip_exec(self):
        self.pose.bbox = []
        self.pose.bbox_set = False
        self.close()

    def next_exec(self):
        (y1, y2), (x1, x2) = self.get_coordinates()
        self.pose.bbox.append([y1, y2, x1, x2])
        self.resize = False
        self.add_padding = False
        self.pose.adjust_bbox()
        self.close()

    def cancel_exec(self):
        self.pose.cancel_bbox_selection = True
        self.close()

    def done_exec(self):
        # User finished drawing ROI
        (y1, y2), (x1, x2) = self.get_coordinates()
        self.pose.bbox.append([y1, y2, x1, x2])
        self.resize = False
        self.add_padding = False
        self.pose.plot_bbox_roi()
        self.pose.adjust_bbox()
        self.close()


class VisualizeVideoSubset(QDialog):
    def __init__(self, gui, video_id, pose, frame_idx, bodyparts):
        super().__init__()
        print("Visualizing video subset")
        self.gui = gui
        self.video_id = video_id
        self.pose = pose
        self.frame_idx = frame_idx
        self.bodyparts = bodyparts

        print("pose shape:", self.pose.shape)
        colors = cm.get_cmap("jet")(np.linspace(0, 1.0, self.pose.shape[-2]))
        colors *= 255
        colors = colors.astype(int)
        self.brushes = np.array([pg.mkBrush(color=c) for c in colors])

        # Add image and pose prediction
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.win = pg.GraphicsLayoutWidget()
        self.win.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.win.setObjectName("Dialog " + str(video_id + 1))
        frame_win = self.win.addViewBox(invertY=True)
        self.current_frame_idx = 0
        frame0 = self.get_frame(self.frame_idx[self.current_frame_idx])
        self.img = pg.ImageItem(frame0)
        frame_win.addItem(self.img)
        self.win.show()
        self.verticalLayout.addWidget(self.win)
        self.update_window_title()

        self.button_horizontalLayout = QtWidgets.QHBoxLayout()
        # Add a next button to the dialog box horizontally centered with other buttons
        self.next_button = QPushButton("Next")
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self.next_exec)
        # Add a previous button to the dialog box horizontally centered with next button and done button
        self.previous_button = QPushButton("Previous")
        self.previous_button.setDefault(False)
        self.previous_button.clicked.connect(self.previous_exec)
        self.previous_button.setEnabled(False)
        # Add buttons to dialog box
        self.done_button = QPushButton("Done")
        self.done_button.setDefault(False)
        self.done_button.clicked.connect(self.done_exec)
        self.button_horizontalLayout.addWidget(self.previous_button)
        self.button_horizontalLayout.addWidget(self.next_button)
        self.button_horizontalLayout.addWidget(self.done_button)
        self.verticalLayout.addLayout(self.button_horizontalLayout)

        # Scatter plot for pose prediction
        self.pose_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen("r", width=2))
        x, y = (
            self.pose[self.current_frame_idx][:, 0],
            self.pose[self.current_frame_idx][:, 1],
        )

        self.pose_scatter.setData(
            x=x,
            y=y,
            size=self.gui.sizeObject.height() * 0.006,
            symbol="o",
            brush=self.brushes,
            hoverable=True,
            hoverSize=self.gui.sizeObject.height() * 0.007,
            hoverSymbol="x",
            pen=(0, 0, 0, 0),
            data=self.bodyparts,
        )
        frame_win.addItem(self.pose_scatter)

        self.exec_()

    def get_frame(self, frame_idx):
        return utils.get_frame(
            frame_idx, self.gui.nframes, self.gui.cumframes, self.gui.video
        )[0]

    def next_exec(self):
        if self.current_frame_idx < len(self.frame_idx) - 1:
            self.current_frame_idx += 1
            self.update_window_title()
            self.img.setImage(self.get_frame(self.frame_idx[self.current_frame_idx]))
            self.update_pose_scatter()
            self.previous_button.setEnabled(True)
            if self.current_frame_idx == len(self.frame_idx) - 1:
                self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(False)

    def previous_exec(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_window_title()
            self.img.setImage(self.get_frame(self.frame_idx[self.current_frame_idx]))
            self.update_pose_scatter()
            self.next_button.setEnabled(True)
            if self.current_frame_idx == 0:
                self.previous_button.setEnabled(False)
        else:
            self.previous_button.setEnabled(False)

    def done_exec(self):
        self.close()

    def update_pose_scatter(self):
        x, y = (
            self.pose[self.current_frame_idx][:, 0],
            self.pose[self.current_frame_idx][:, 1],
        )
        self.pose_scatter.setData(
            x=x,
            y=y,
            size=self.gui.sizeObject.height() * 0.006,
            symbol="o",
            brush=self.brushes,
            hoverable=True,
            hoverSize=self.gui.sizeObject.height() * 0.005,
            hoverSymbol="x",
            pen=(0, 0, 0, 0),
            data=self.bodyparts,
        )

    def update_window_title(self):
        self.setWindowTitle(
            "Frame: "
            + str(self.frame_idx[self.current_frame_idx])
            + " ({}/{})".format(self.current_frame_idx, len(self.frame_idx) - 1)
        )
