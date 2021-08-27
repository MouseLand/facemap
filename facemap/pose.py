import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from . import transforms, roi
import torch
from . import UNet_helper_functions as utils
import cv2
import time
from PyQt5 import QtGui

class Pose():
    def __init__(self, parent):
        self.parent = parent
        self.pose_labels = None
        self.bodyparts = None
        self.bbox = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.load_model()
        self.set_bbox()
        # plot key points  on all frames 
        """
        self.keypoints_labels = [all_labels[i] for i in sorted(np.unique(all_labels, return_index=True)[1])]
        # Choose colors for each label: provide option for color blindness as well
        self.colors = cm.get_cmap('gist_rainbow')(np.linspace(0, 1., len(self.keypoints_labels)))
        self.colors *= 255
        self.colors = self.colors.astype(int)
        self.colors[:,-1] = 127
        self.brushes = np.array([pg.mkBrush(color=c) for c in self.colors])
        self.parent.Pose_scatterplot.setData(x, y, size=15, symbol='o', brush=self.brushes[filtered_keypoints],
                            hoverable=True, hoverSize=15)
        """
        # save prediction

    def set_bbox(self):
        bbox_set = False
        prev_bbox = (np.nan, np.nan, np.nan, np.nan)
        while not bbox_set:
            self.bbox = self.get_bbox_region(prev_bbox)
            prev_bbox = self.bbox
            # plot bbox as ROI
            x1, x2, y1, y2 = self.bbox
            dx, dy = x2-x1, y2-y1
            self.bbox_plot = roi.sROI(rind=1, rtype="bbox", iROI=1, moveable=True, 
                                    parent=self.parent, pos=(y1, x1, dy, dx))
            # get user validation
            qm = QtGui.QMessageBox
            ret = qm.question(self.parent,'', "Does the suggested ROI match the requirements?", qm.Yes | qm.No)
            bbox_set = ret == qm.Yes
            if not bbox_set:
                del self.bbox_plot
        print("bbox set!")

    def get_bbox_region(self, prev_bbox):
        """
        Obtain ROI/bbox for cropping images to use as input for model 
        """
        num_iter = 1 # select optimum value
        self.parent.p0.addItem(self.parent.Pose_scatterplot)
        for it in range(num_iter):
            t0 = time.time()
            imgs = self.get_batch_imgs()
            # Get bounding box regions
            bbox, lm_mean = transforms.get_bounding_box(imgs, self.net, prev_bbox)
            x = lm_mean[:,0]
            y = lm_mean[:,1]
            self.parent.Pose_scatterplot.setData(x, y, size=15, symbol='o', 
                                        hoverable=True, hoverSize=15)
            prev_bbox = bbox    # Update bbox 
            adjusted_bbox = transforms.adjust_bbox(bbox, (self.parent.Ly[0], self.parent.Lx[0])) # Adjust bbox to be square instead of rectangle
            print(adjusted_bbox, "bbox", time.time()-t0) #
            #min_x, max_x, min_y, max_y = adjusted_bbox
            #min_x, max_x, min_y, max_y = min_x-padding, max_x+padding, min_y-padding, max_y+padding
        return adjusted_bbox
        
    def load_model(self):
        model_file = "/Users/Atika/Neuroverse/Janelia/facemap-mac/facemap/example_model_small"  # Replace w/ function that downloads model from server
        self.bodyparts = None#data['landmarks']
        kernel_size = 3
        nbase = [1, 64, 64*2, 64*3, 64*4] # number of channels per layer
        nout = 14 #len(self.bodyparts)  # number of outputs
        net = utils.UNet_b(nbase, nout, kernel_size, labels_id=self.bodyparts)
        if torch.cuda.is_available():
            cpu_is_device = False
        else:
            cpu_is_device = True
        net.load_model(model_file, cpu=cpu_is_device)
        return net

    def get_batch_imgs(self, batch_size=3, nchannels=1):
        """
        Get batch of images sampled randomly from video. Note: works for single video only
        Parameters
        -------------
        self: (Pose) object
        batch_size: int number
        Returns
        --------------
        im: 1-D list
            list of images each of size (Ly x Lx) 
        """
        img_ind = np.random.randint(0, self.parent.cumframes[-1], batch_size)
        im = np.zeros((batch_size, nchannels, self.parent.Ly[0], self.parent.Lx[0]))
        for k, ind in enumerate(img_ind):
            frame = self.parent.get_frame(ind)[0]
            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_grayscale_preprocessed = transforms.preprocess_img(frame_grayscale)
            im[k] = frame_grayscale_preprocessed
        return im
