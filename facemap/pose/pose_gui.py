import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QDialog,
    QPushButton,
    QLabel)

from .. import roi
from .pose import Pose
from . import transforms

import cv2
from .. import utils
import time

"""
Pose subclass for generating pose estimates on GUI involving user validation for bbox.
Currently supports single video processing only.
"""
class PoseGUI(Pose):
    def __init__(self, parent=None):
        self.parent = parent
        if self.parent is not None:
            super().__init__(self.parent.filenames)
        if self.bbox is None:
            self.draw_user_bbox() #draw_suggested_bbox()  

    def draw_suggested_bbox(self):
        if self.bbox_set:
            del self.bbox_roi
            x1, x2, y1, y2 = self.bbox
            dx, dy = x2-x1, y2-y1
            self.plot_bbox_roi(y1, x1, dy, dx)
        else:
            prev_bbox = (np.nan, np.nan, np.nan, np.nan)
            while not self.bbox_set:
                self.bbox = np.round(super().estimate_bbox_region(prev_bbox)).astype(int)
                prev_bbox = self.bbox
                # plot bbox as ROI
                x1, x2, y1, y2 = 0,0,0,0#self.bbox
                dx, dy = x2-x1, y2-y1
                self.plot_bbox_roi(y1, x1, dy, dx)
                # get user validation
                qm = QtGui.QMessageBox
                ret = qm.question(self.parent,'', "Does the suggested ROI match the requirements?", 
                                    qm.Yes | qm.No)
                """
                msgBox = QtGui.QMessageBox()
                msgBox.setText('What to do?')
                msgBox.addButton(QtGui.QPushButton('Yes'))
                msgBox.addButton(QtGui.QPushButton('No'))
                msgBox.addButton(QtGui.QPushButton('Draw'))
                ret = msgBox.exec_()"""
                print("ret", ret)
                self.bbox_set = ret == qm.Yes
                if not self.bbox_set:
                    del self.bbox_roi

    # Draw box on GUI using user's input
    def draw_user_bbox(self):
        """
        Function for user to draw a bbox
        """
        self.bbox_set = False
        sample_frame = utils.get_frame(0, self.nframes, self.cumframes, self.containers)[0]  
        # Trigger new window for ROI selection
        exPopup = ROI_popup(sample_frame, self.parent, self)
        exPopup.show()        

    def adjust_bbox_params(self):
        # This function adjusts bbox so that it is of minimum dimension: 256,256
        sample_frame = utils.get_frame(0, self.nframes, self.cumframes, self.containers)[0]  
        x1, x2, y1, y2, resize = transforms.get_crop_resize_params(sample_frame, 
                                                                    x_dims=(self.bbox[0], self.bbox[1]), 
                                                                    y_dims=(self.bbox[2], self.bbox[3]))
        self.bbox = x1, x2, y1, y2, resize
        self.bbox_set = True
        print("user selected bbox after adjustment:", self.bbox)                                       

    def plot_bbox_roi(self, moveable=True, resizable=True):
        self.adjust_bbox_params()
        self.parent.nROIs += 1
        x1, x2, y1, y2, _ = self.bbox
        dy, dx = y2-y1, x2-x1
        self.bbox_roi = roi.sROI(rind=4, rtype="bbox", iROI=self.parent.nROIs, moveable=False, 
                                    resizable=False, parent=self.parent, pos=(x1, y1, dx, dy))
        self.parent.ROIs.append(self.bbox_roi)
        self.bbox_set = True      

class ROI_popup(QDialog):
    def __init__(self, frame, parent, pose):
        super().__init__(parent)
        self.parent = parent
        self.frame = frame
        self.pose = pose

        self.setWindowTitle('Select ROI')
        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        # Add image and ROI bbox
        self.win = pg.GraphicsLayoutWidget()
        ROI_win = self.win.addViewBox(invertY=True)
        self.img = pg.ImageItem(self.frame)
        ROI_win.addItem(self.img)
        self.roi = pg.RectROI([0,0],[100,100],pen=pg.mkPen('r',width=2), movable=True,resizable=True)
        ROI_win.addItem(self.roi)
        self.win.show()
        self.verticalLayout.addWidget(self.win)

        # Add buttons to dialog box
        self.done_button = QtGui.QPushButton('Done')
        self.done_button.setDefault(True)
        self.done_button.clicked.connect(lambda: self.done_exec(self, parent))
        self.cancel_button = QtGui.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.close)
        
        # Position buttons
        self.widget = QtWidgets.QWidget(self)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.addWidget(self.cancel_button)
        self.horizontalLayout.addWidget(self.done_button)
        self.verticalLayout.addWidget(self.widget)

        self.show() 
    
    def get_coordinates(self):
        roi_tuple, _ = self.roi.getArraySlice(self.frame, self.img, returnSlice=False)
        (x1, x2), (y1, y2) = roi_tuple[0], roi_tuple[1]
        return (x1, x2), (y1, y2)

    def done_exec(self, dialogBox, parent):
        # User finished drawing ROI
        (x1, x2), (y1, y2) = self.get_coordinates()
        self.pose.bbox = x1, x2, y1, y2, False
        self.pose.plot_bbox_roi()
        dialogBox.close()

# Following used to check cropped sections of frames
class test_popup(QDialog):
    def __init__(self, frame, parent):
        super().__init__(parent)
        self.parent = parent
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

        self.cancel_button = QtGui.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.close)
        
        # Position buttons
        self.widget = QtWidgets.QWidget(self)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout.addWidget(self.cancel_button)
        self.verticalLayout.addWidget(self.widget)

        self.show() 