import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QDialog,
    QPushButton)

from facemap import roi, utils
from facemap.pose import pose
from . import transforms

"""
Pose subclass for generating obtaining bounding box from user input.
Currently supports single video processing only.
"""
class PoseGUI(pose.Pose):
    def __init__(self, gui=None):
        self.gui = gui
        super(PoseGUI, self).__init__(gui=self.gui)
        self.bbox_set = False
        self.bbox = []
        self.cancel = False

    # Draw box on GUI using user's input
    def draw_user_bbox(self):
        """ 
        Function for user to draw a bbox
        """
        # Get sample frame from each video in case of multiple videos
        sample_frame = utils.get_frame(0, self.nframes, self.cumframes, self.containers)
        last_video=False
        for video_id, frame in enumerate(sample_frame):         
            # Trigger new window for ROI selection of each frame
            if video_id == len(sample_frame)-1:
                last_video = True
            ROI_popup(frame, video_id, self.gui, self, last_video) 
        return self.bbox, self.bbox_set, self.cancel

    def adjust_bbox_params(self):
        # This function adjusts bbox so that it is of minimum dimension: 256,256
        sample_frame = utils.get_frame(0, self.nframes, self.cumframes, self.containers)  
        for i, bbox in enumerate(self.bbox):
            x1, x2, y1, y2, resize = transforms.get_crop_resize_params(sample_frame[i], 
                                                                    x_dims=(bbox[0], bbox[1]), 
                                                                    y_dims=(bbox[2], bbox[3]))
            self.bbox[i] = [x1, x2, y1, y2, resize]
            # If bbox_region is not square, then adjust the bbox_region to be square
            if self.bbox[i][2] - self.bbox[i][0] != self.bbox[i][3] - self.bbox[i][1]:
                if self.bbox[i][2] - self.bbox[i][0] > self.bbox[i][3] - self.bbox[i][1]:
                    self.bbox[i][2] = self.bbox[i][0] + self.bbox[i][3] - self.bbox[i][1]
                else:
                    self.bbox[i][3] = self.bbox[i][1] + self.bbox[i][2] - self.bbox[i][0]
        print("user selected bbox after adjustment:", self.bbox)

    def plot_bbox_roi(self):
        self.adjust_bbox_params()
        for i, bbox in enumerate(self.bbox):
            x1, x2, y1, y2, _ = bbox
            dy, dx = y2-y1, x2-x1
            xrange = np.arange(y1+self.gui.sx[i], y2+self.gui.sx[i]).astype(np.int32)
            yrange = np.arange(x1+self.gui.sy[i], x2+self.gui.sy[i]).astype(np.int32)
            x1, y1 = yrange[0], xrange[0]
            self.gui.add_ROI(roitype=4+1, roistr="bbox_{}".format(i), moveable=False, resizable=False,
                            pos=(x1, y1, dx, dy), ivid=i, yrange=yrange, xrange=xrange)
        self.bbox_set = True    

class ROI_popup(QDialog):
    def __init__(self, frame, video_id, gui, pose, last_video):
        super().__init__()
        self.gui = gui
        self.frame = frame
        self.pose = pose
        self.last_video = last_video
        self.setWindowTitle('Select ROI for video: '+str(video_id))

        # Add image and ROI bbox
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.win = pg.GraphicsLayoutWidget()
        self.win.setObjectName("Dialog "+str(video_id+1))
        ROI_win = self.win.addViewBox(invertY=True)
        self.img = pg.ImageItem(self.frame)
        ROI_win.addItem(self.img)
        self.roi = pg.RectROI([0,0],[100,100],pen=pg.mkPen('r',width=2), movable=True,resizable=True)
        ROI_win.addItem(self.roi)
        self.win.show()
        self.verticalLayout.addWidget(self.win)

        # Add buttons to dialog box
        self.done_button = QPushButton('Done')
        self.done_button.setDefault(True)
        self.done_button.clicked.connect(self.done_exec)
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.cancel_exec)
        # Add a next button to the dialog box horizontally centered with cancel button and done button
        self.next_button = QPushButton('Next')
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self.next_exec)
        # Add a skip button to the dialog box horizontally centered with cancel button and done button
        self.skip_button = QPushButton('Skip')
        self.skip_button.setDefault(True)
        self.skip_button.clicked.connect(self.skip_exec)

        # Position buttons
        self.widget = QtWidgets.QWidget(self)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
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
        roi_tuple, _ = self.roi.getArraySlice(self.frame, self.img, returnSlice=False)
        (x1, x2), (y1, y2) = roi_tuple[0], roi_tuple[1]
        return (x1, x2), (y1, y2)

    def skip_exec(self):
        self.pose.bbox = []
        self.pose.bbox_set = False
        self.close()

    def next_exec(self):
        (x1, x2), (y1, y2) = self.get_coordinates()
        self.pose.bbox.append([x1, x2, y1, y2, False])
        self.close()

    def cancel_exec(self):
        self.pose.cancel = True
        self.close()

    def done_exec(self):
        # User finished drawing ROI
        (x1, x2), (y1, y2) = self.get_coordinates()
        self.pose.bbox.append([x1, x2, y1, y2, False])
        self.pose.plot_bbox_roi()
        self.close()

# Following used to check cropped sections of frames
class test_popup(QDialog):
    def __init__(self, frame, gui):
        super().__init__(gui)
        self.gui = gui
        self.frame = frame

        self.setWindowTitle('Chosen ROI')
        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        # Add image and ROI bbox
        self.win = pg.GraphicsLayoutWidget()
        ROI_win = self.win.addViewBox(invertY=True)
        self.img = pg.ImageItem(self.frame)
        ROI_win.addItem(self.img)
        self.win.show()
        self.verticalLayout.addWidget(self.win)

        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.close)
        
        # Position buttons
        self.widget = QtWidgets.QWidget(self)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.addWidget(self.cancel_button)
        self.verticalLayout.addWidget(self.widget)

        self.show() 

