import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

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
        super().__init__(self.parent.filenames, bbox_user_validation=True)
        if self.bbox is None:
            self.draw_user_box() #draw_suggested_bbox()  

    def run(self):
        # Get bbox coordinates   
        #y, x, dy, dx =  self.bbox_roi.pos
        #self.bbox = x, x+dx, y, y+dy
        t0 = time.time()
        self.parent.poseFilepath = super().run()
        self.plot_pose_estimates()
        print("~~~~~~~~~~~~~~~~~~~~~DONE~~~~~~~~~~~~~~~~~~~~~")
        print("Time taken:", time.time()-t0)

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
    def draw_user_box(self):
        """
        Function for user to draw a bbox
        """
        self.bbox_set = False
        frame = utils.get_frame(0, self.nframes, self.cumframes, self.containers)[0]  
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_grayscale_preprocessed = transforms.preprocess_img(frame_grayscale) 
        from_center = False
        show_crosshair = False
        user_selected_ROI = cv2.selectROI(frame_grayscale_preprocessed.squeeze(), from_center, show_crosshair)
        top_x, top_y, bottom_x, bottom_y = user_selected_ROI
        #cv2.SetImageROI(frame_grayscale_preprocessed, user_selected_ROI)
        self.plot_bbox_roi(top_y, top_x, bottom_y, bottom_x, moveable=False, resizable=False)
        cv2.destroyWindow("ROI selector")
        self.bbox = top_x, top_x+bottom_x, top_y, top_y+bottom_y
        self.bbox_set = True
        print("user selected bbox:", self.bbox)

    def plot_bbox_roi(self, y1, x1, dy, dx, moveable=True, resizable=False):
        self.parent.nROIs += 1
        self.bbox_roi = roi.sROI(rind=4, rtype="bbox", iROI=self.parent.nROIs, moveable=moveable, 
                                    resizable=resizable, parent=self.parent, pos=(y1, x1, dy, dx))
        self.parent.ROIs.append(self.bbox_roi)

    def plot_pose_estimates(self):
        # Plot labels
        self.parent.poseFileLoaded = True
        self.parent.load_labels()
        self.parent.Labels_checkBox.setChecked(True)    

