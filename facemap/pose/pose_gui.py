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

    def run(self):
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
    def draw_user_bbox(self):
        """
        Function for user to draw a bbox
        """
        self.bbox_set = False
        """
        frame = utils.get_frame(0, self.nframes, self.cumframes, self.containers)[0]  
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_grayscale_preprocessed = transforms.preprocess_img(frame_grayscale) 
        from_center = False
        show_crosshair = False
        user_selected_ROI = cv2.selectROI(frame_grayscale_preprocessed.squeeze(), from_center, show_crosshair)
        top_x, top_y, bottom_x, bottom_y = user_selected_ROI
        resize = False
        self.bbox = top_x, top_x+bottom_x, top_y, top_y+bottom_y, resize
        print("user selected bbox:", self.bbox)
        #cv2.destroyWindow("ROI selector")
        """
        sample_frame = utils.get_frame(0, self.nframes, self.cumframes, self.containers)[0]  
        exPopup = ROI_popup(sample_frame, self.parent, self)
        exPopup.show()        

    def set_bbox_params(self):
        sample_frame = utils.get_frame(0, self.nframes, self.cumframes, self.containers)[0]  
        x1, x2, y1, y2, resize = transforms.get_crop_resize_params(sample_frame, 
                                                                    x_dims=(self.bbox[0], self.bbox[1]), 
                                                                    y_dims=(self.bbox[2], self.bbox[3]))
        self.bbox = x1, x2, y1, y2, resize
        self.bbox_set = True
        print("user selected bbox after adjustment:", self.bbox)

    def plot_bbox_roi(self, moveable=True, resizable=True):
        print("bbox before", self.bbox)
        self.set_bbox_params()
        self.parent.nROIs += 1
        x1, x2, y1, y2, _ = self.bbox
        dy, dx = y2-y1, x2-x1
        self.bbox_roi = roi.sROI(rind=4, rtype="bbox", iROI=self.parent.nROIs, moveable=moveable, 
                                    resizable=resizable, parent=self.parent, pos=(x1, y1, dx, dy))
        self.parent.ROIs.append(self.bbox_roi)
        """
        # ROI dimensions
        pos0 = self.bbox_roi.ROI.getSceneHandlePositions()
        pos = self.parent.p0.mapSceneToView(pos0[0][1])
        x2, y2 = int(pos.x()), int(pos.y())
        sizex, sizey = self.bbox_roi.ROI.size()
        x1 = int(x2-sizex)
        y1 = int(y2-sizey)
        """
        self.bbox = x1, x2, y1, y2, False

    def plot_pose_estimates(self):
        # Plot labels
        self.parent.poseFileLoaded = True
        self.parent.load_labels()
        self.parent.Labels_checkBox.setChecked(True)    

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
        print("ROI selected:", (x1, x2), (y1, y2))
        self.pose.plot_bbox_roi()
        dialogBox.close()
    """
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(1000,500)
        ROI_win = self.win.addViewBox()

        imv = pg.ImageItem(frame)#pg.ImageView()
        #imv.setImage(frame)
        #self.hide_buttons(imv)
        #imv.show()
        ROI_win.addItem(imv)
        roi = pg.RectROI([0,0],[100,100],pen=pg.mkPen('r',width=2), movable=True,resizable=True)
        ROI_win.addItem(roi)
        print("here")
        self.win.show()

        def getcoordinates(roi):
            roi_tuple, _ = roi.getArraySlice(frame, imv.imageItem, returnSlice=False)
            (x1, x2), (y1, y2) = roi_tuple[0], roi_tuple[1]

        roi.sigRegionChanged.connect(getcoordinates)

    def hide_buttons(self, imv):
        imv.ui.roiBtn.hide()
        imv.ui.histogram.hide()
        imv.ui.menuBtn.hide()
        done_button = QtGui.QPushButton('Done')
        done_button.setDefault(True)

        # Add first video frame
        self.roi_frame = QLabel(self)
        image = QtGui.QImage(frame, frame.shape[1],\
                            frame.shape[0], frame.shape[1] * 3,QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(image)
        self.roi_frame.resize(frame.shape[0], frame.shape[1])
        self.roi_frame.setPixmap(pixmap.scaled(self.roi_frame.size(), QtCore.Qt.KeepAspectRatio))
        #self.verticalLayout.addWidget(self.roi_frame)
        
        self.rubberband = QtGui.QRubberBand(
            QtGui.QRubberBand.Rectangle, self)
        self.setMouseTracking(True)
        self.selector_roi = None
        color = QtGui.QPalette()
        color.setBrush(QtGui.QPalette.Highlight, QtGui.QBrush(QtCore.Qt.red))
        self.rubberband.setPalette(color)
        self.rubberband.setWindowOpacity(1.0)
        print(frame.shape)
        self.rubberband.show()
    def mousePressEvent(self, event):
        self.origin = event.pos()
        self.rubberband.setGeometry(
            QtCore.QRect(self.origin, QtCore.QSize()))
        self.rubberband.show()
        QtGui.QWidget.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if self.rubberband.isVisible():
            self.rubberband.setGeometry(
                QtCore.QRect(self.origin, event.pos()).normalized())
        QtGui.QWidget.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self, event):
        if self.rubberband.isVisible():
            self.rubberband.hide()
            selected = []
            rect = self.rubberband.geometry()
            self.rubberband.show()
            self.selector_roi = rect.getRect()
            print(self.selector_roi)
        QtGui.QWidget.mouseReleaseEvent(self, event)   
    """

