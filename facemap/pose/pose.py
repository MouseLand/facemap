import os
import time

from numpy.core.fromnumeric import cumprod
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import torch

from .. import utils
from . import UNet_helper_functions as UNet_utils
from . import unet_torch
from . import transforms
from PyQt5 import QtGui 
from . import pose_gui

"""
Base class for generating pose estimates using command line interface.
Currently supports single video processing only.
"""
class Pose():
    def __init__(self, filenames=None, savepath=None):
        self.filenames = filenames
        self.cumframes, self.Ly, self.Lx, self.containers = utils.get_frame_details(self.filenames)
        self.nframes = self.cumframes[-1]
        self.pose_labels = None
        self.bodyparts = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.load_model()
        self.savepath = savepath
        self.bbox = None

    def run(self, save=True):
        # Predict and save pose
        self.set_bbox_params()
        self.dataFrame = self.predict_landmarks()
        if save:
            self.savepath = self.save_pose_prediction()
        return self.savepath

    def estimate_bbox_region(self, prev_bbox):
        """
        Obtain ROI/bbox for cropping images to use as input for model 
        """
        num_iter = 3 # select optimum value for CPU vs. GPU
        for it in range(1):#num_iter):
            t0 = time.time()
            imgs = self.get_batch_imgs()
            # Get bounding box for imgs 
            bbox, _ = transforms.get_bounding_box(imgs, self.net, prev_bbox)
            prev_bbox = bbox    # Update bbox 
            adjusted_bbox = transforms.adjust_bbox(bbox, (self.Ly[0], self.Lx[0])) # Adjust bbox to be square instead of rectangle
            print(adjusted_bbox, "bbox", time.time()-t0) 
        return adjusted_bbox

    def predict_landmarks(self, save_prediction=True):
        """
        Predict labels for all frames in video and save output as .h5 file
        """
        scorer = "Facemap" 
        bodyparts = self.net.labels_id 
        nchannels = 1
        if self.device == 'cpu':
            batch_size = 2
        else:
            batch_size = 16

        # Create an empty dataframe
        for index, bodypart in enumerate(bodyparts):
            columnindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y","likelihood"]],
                names=["scorer", "bodyparts", "coords"])
            frame = pd.DataFrame(
                                np.nan,
                                columns=columnindex,
                                index=np.arange(self.cumframes[-1]))
            if index == 0:
                dataFrame = frame
            else:
                dataFrame = pd.concat([dataFrame, frame], axis=1)

        # Store predictions in dataframe
        self.net.eval()
        start = 0
        end = batch_size
        Xstart, Xstop, Ystart, Ystop, resize = self.bbox
        with tqdm(total=self.cumframes[-1], unit='frame', unit_scale=True) as pbar:
            while start != self.cumframes[-1]:#
                # Pre-pocess images
                im = np.zeros((end-start, nchannels, 256, 256))
                for i, frame_ind in enumerate(np.arange(start,end)):
                    frame = utils.get_frame(frame_ind, self.nframes, self.cumframes, self.containers)[0]  
                    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_grayscale_preprocessed = transforms.preprocess_img(frame_grayscale)
                    im[i,0] = transforms.crop_resize(frame_grayscale_preprocessed.squeeze(), Xstart, 
                                                    Xstop, Ystart, Ystop, resize)
                
                # Network prediction 
                hm_pred, locref_pred = self.net(torch.tensor(im).to(device=self.net.DEVICE, dtype=torch.float32)) # convert to tensor and send to device
                del im
                hm_pred = UNet_utils.gaussian_smoothing(hm_pred.cpu(), sigma=2, nms_size=6, sigmoid=True)
                pose = UNet_utils.argmax_pose_predict_batch(hm_pred.cpu().detach().numpy(), 
                                                                locref_pred.cpu().detach().numpy(),
                                                                UNet_utils.STRIDE)
                # Get adjusted landmarks that fit to original image size
                landmarks = pose[:,:,:2].squeeze()
                likelihood = pose[:,:,-1].squeeze()
                del hm_pred, locref_pred
                Xlabel, Ylabel = transforms.labels_crop_resize(landmarks[:,:,0], landmarks[:,:,1], 
                                                                Xstart, Ystart,
                                                                current_size=(256, 256), 
                                                                desired_size=(self.bbox[3]-self.bbox[2], 
                                                                            self.bbox[1]-self.bbox[0]))
                dataFrame.iloc[start:end,::3] = Xlabel
                dataFrame.iloc[start:end,1::3] = Ylabel
                dataFrame.iloc[start:end,2::3] = likelihood
                if end%2==0:
                    pbar.update(batch_size)
                start = end 
                end += batch_size
                end = min(end, self.cumframes[-1])
        return dataFrame

    def save_pose_prediction(self):
        # Save prediction to .h5 file
        basename, filename = os.path.split(self.filenames[0][0])
        videoname, _ = os.path.splitext(filename)
        poseFilepath = os.path.join(basename, videoname+"_FacemapPose.h5")
        self.dataFrame.to_hdf(poseFilepath, "df_with_missing", mode="w")
        return poseFilepath  
        
    def load_model(self):
        """
        Load pre-trained UNet model for labels prediction 
        """
        # Replace following w/ a function that downloads model from a server
        model_file = "/home/stringlab/Facemap/facemap/facemap/model_state.pt" 
        model_params_file = "/home/stringlab/Facemap/facemap/facemap/model_params.pth"  
        model_params = torch.load(model_params_file)
        self.bodyparts = model_params['landmarks'] 
        kernel_size = 3
        nout = len(self.bodyparts)  # number of outputs
        net = unet_torch.FMnet(img_ch=1, output_ch=nout, labels_id=self.bodyparts, 
                                filters=64, kernel=kernel_size, device=self.device)
        if torch.cuda.is_available():
            cpu_is_device = False
        else:
            cpu_is_device = True
        net.load_model(model_file, cpu=cpu_is_device)
        net.to(self.device)
        print("Using cpu as device:", cpu_is_device)
        return net

    def get_batch_imgs(self, batch_size=1, nchannels=1):
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
        img_ind = np.random.randint(0, self.cumframes[-1], batch_size)
        im = np.zeros((batch_size, nchannels, self.Ly[0], self.Lx[0]))
        for k, ind in enumerate(img_ind):
            frame = utils.get_frame(ind, self.nframes, self.cumframes, self.containers)[0]        ## ~~~~~~~~~~~~~~~~    GUI function     ~~~~~~~~~~~~~~~~
            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_grayscale_preprocessed = transforms.preprocess_img(frame_grayscale)
            im[k] = frame_grayscale_preprocessed
        return im
