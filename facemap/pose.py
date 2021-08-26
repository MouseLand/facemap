import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from . import transforms
import torch
from . import UNet_helper_functions as utils

class Pose():
    def __init__(self, parent):
        self.parent = parent
        self.pose_labels = None
        self.bodyparts = None
        self.bbox = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.load_model()
        self.get_bbox_region()

    def get_bbox_region(self):
        """
        Obtain ROI/bbox for cropping images to use as input for model 
        """
        num_iter = 5
        prev_bbox = (np.nan, np.nan, np.nan, np.nan)
        for it in range(num_iter):
            imgs = self.get_batch_imgs()
            # Get bounding box regions
            bbox, lm_mean = transforms.get_bounding_box(imgs, self.net, prev_bbox)
            prev_bbox = bbox    # Update bbox 
            print(bbox, lm_mean)
            #adjusted_bbox = adjust_bbox(bbox, imgs.shape[-2:]) # Adjust bbox to be square instead of rectangle
            #min_x, max_x, min_y, max_y = adjusted_bbox
            #min_x, max_x, min_y, max_y = min_x-padding, max_x+padding, min_y-padding, max_y+padding
        
    def load_model(self):
        model_file = "/Users/Atika/Neuroverse/Janelia/facemap-mac/facemap/example_model"  # Replace w/ function that downloads model from server
        #data = torch.load(model_file)
        #print(data.keys())
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

    def get_batch_imgs(self, batch_size=3):
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
        im = []
        for i in img_ind:
            im.append(transforms.preprocess_img(self.parent.img[0][i]))
        return im
