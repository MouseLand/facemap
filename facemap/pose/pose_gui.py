from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5 import QtGui

class PoseGUI(Pose):
    def __init__(self, parent=None):
        self.parent = parent

    def draw_bbox(self):
        if self.bbox_set:
            del self.bbox_plot
            x1, x2, y1, y2 = self.bbox
            dx, dy = x2-x1, y2-y1
            self.bbox_plot = roi.sROI(rind=1, rtype="bbox", iROI=1, moveable=False, 
                                    parent=self.parent, pos=(y1, x1, dy, dx))
        else:
            prev_bbox = (np.nan, np.nan, np.nan, np.nan)
            while not self.bbox_set:
                self.bbox = np.round(self.estimate_bbox_region(prev_bbox)).astype(int)
                prev_bbox = self.bbox
                # plot bbox as ROI
                x1, x2, y1, y2 = self.bbox
                dx, dy = x2-x1, y2-y1
                self.bbox_plot = roi.sROI(rind=1, rtype="bbox", iROI=1, moveable=False, 
                                        parent=self.parent, pos=(y1, x1, dy, dx))
                # get user validation
                qm = QtGui.QMessageBox
                ret = qm.question(self.parent,'', "Does the suggested ROI match the requirements?", qm.Yes | qm.No)
                self.bbox_set = ret == qm.Yes
                if not self.bbox_set:
                    del self.bbox_plot

    def estimate_bbox_region(self, prev_bbox):
        """
        Obtain ROI/bbox for cropping images to use as input for model 
        """
        num_iter = 3 # select optimum value
        for it in range(num_iter):
            t0 = time.time()
            imgs = self.get_batch_imgs()
            # Get bounding box for imgs 
            bbox, _ = transforms.get_bounding_box(imgs, self.net, prev_bbox)
            prev_bbox = bbox    # Update bbox 
            adjusted_bbox = transforms.adjust_bbox(bbox, (self.parent.Ly[0], self.parent.Lx[0])) # Adjust bbox to be square instead of rectangle
            print(adjusted_bbox, "bbox", time.time()-t0) 
        return adjusted_bbox