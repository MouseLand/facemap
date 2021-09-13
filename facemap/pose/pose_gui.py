from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5 import QtGui
import pyqtgraph as pg
from .. import roi

"""
Pose subclass for generating pose estimates on GUI involving user validation for bbox.
Currently supports single video processing only.
"""
class PoseGUI(Pose):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(self.parent.filenames)
        if super().bbox is None:
            self.draw_bbox()                  
        super().run()
        self.plot_pose_labels()

    def draw_bbox(self):
        if super().bbox_set:
            del self.bbox_plot
            x1, x2, y1, y2 = super().bbox
            dx, dy = x2-x1, y2-y1
            self.bbox_plot = roi.sROI(rind=1, rtype="bbox", iROI=1, moveable=False, 
                                    parent=self.parent, pos=(y1, x1, dy, dx))
        else:
            prev_bbox = (np.nan, np.nan, np.nan, np.nan)
            while not super().bbox_set:
                super().bbox = np.round(super().estimate_bbox_region(prev_bbox)).astype(int)
                prev_bbox = super().bbox
                # plot bbox as ROI
                x1, x2, y1, y2 = super().bbox
                dx, dy = x2-x1, y2-y1
                self.bbox_plot = roi.sROI(rind=1, rtype="bbox", iROI=1, moveable=False, 
                                        parent=self.parent, pos=(y1, x1, dy, dx))
                # get user validation
                qm = QtGui.QMessageBox
                ret = qm.question(self.parent,'', "Does the suggested ROI match the requirements?", qm.Yes | qm.No)
                super().bbox_set = ret == qm.Yes
                if not super().bbox_set:
                    del self.bbox_plot

    def plot_pose_labels(self):
        # Plot labels
        self.parent.poseFileLoaded = True
        self.parent.load_labels()
        self.parent.Labels_checkBox.setChecked(True)    

