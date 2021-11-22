import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

from .. import utils
from . import UNet_helper_functions as UNet_utils
from . import unet_torch
from . import transforms

"""
Base class for generating pose estimates using command line interface.
Currently supports single video processing only.
"""
class Pose():
    def __init__(self, filenames, bbox_user_validation=False, savepath=None):
        self.filenames = filenames
        self.cumframes, self.Ly, self.Lx, self.containers = utils.get_frame_details(self.filenames)
        self.nframes = self.cumframes[-1]
        self.pose_labels = None
        self.bodyparts = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.load_model()
        self.savepath = savepath

        if not bbox_user_validation:
            self.bbox = np.round(self.estimate_bbox_region((np.nan, np.nan, np.nan, np.nan))).astype(int)
            self.bbox_set = True
        else:
            self.bbox = None
            self.bbox_set = False

    def run(self, save=True):
        self.cropped_img_slice = self.get_img_slice()
        # Predict and save pose
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
        batch_size = 1

        # Create an empty dataframe
        for index, bodypart in enumerate(bodyparts):
            columnindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y"]],#"likelihood" 
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
        img_x, img_y = self.bbox[1]-self.bbox[0], self.bbox[-1]-self.bbox[-2]
        orig_imgy, orig_imgx = self.Ly[0], self.Lx[0]
        while end != 2: # self.cumframes[-1]:
            # Pre-pocess images
            im = np.zeros((batch_size, nchannels, img_y, img_x))
            for i, frame_ind in enumerate(np.arange(start,end)):
                frame = utils.get_frame(frame_ind, self.nframes, self.cumframes, self.containers)[0]  
                frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_grayscale_preprocessed = transforms.preprocess_img(frame_grayscale)
                im[i,0] = frame_grayscale_preprocessed[0][self.cropped_img_slice]    # used cropped section only
            
            # Network prediction 
            hm_pred, locref_pred = self.net(torch.tensor(im).to(device=self.net.DEVICE, dtype=torch.float32)) # convert to tensor and send to device
            del im
            pose = UNet_utils.argmax_pose_predict_batch(hm_pred.cpu().detach().numpy(), 
                                                            locref_pred.cpu().detach().numpy(),
                                                            UNet_utils.STRIDE)
            # Get adjusted landmarks that fit to original image size
            landmarks = pose[:,:,:2].squeeze()
            likelihood = pose[:,:,2].squeeze()
            del hm_pred, locref_pred
            Xlabel, Ylabel = transforms.labels_resize(landmarks[:,0], landmarks[:,1], 
                                                    current_size=(img_x, img_y), 
                                                    desired_size=(orig_imgx, orig_imgy))
            im_pred = np.array(list(zip(Xlabel, Ylabel)))
            dataFrame.iloc[start:end] = im_pred.ravel()
            start = end 
            end += batch_size
        return dataFrame
        
    def save_pose_prediction(self):
        # Save prediction to .h5 file
        basename, filename = os.path.split(self.filenames[0][0])
        videoname, _ = os.path.splitext(filename)
        poseFilepath = os.path.join(basename, videoname+"_FacemapPose.h5")
        self.dataFrame.to_hdf(poseFilepath, "df_with_missing", mode="w")
        return poseFilepath  
    
    def get_img_slice(self):
        x1, x2, y1, y2 = self.bbox
        slc = [slice(y1, y2), slice(x1, x2)]   # slice img_y and img_x
        return slc
        
    def load_model(self):
        """
        Load pre-trained UNet model for labels prediction 
        """
        # Replace following w/ a function that downloads model from a server
        model_file = "/Users/Atika/Neuroverse/Janelia/facemap-mac/model_state.pt" 
        model_params_file = "/Users/Atika/Neuroverse/Janelia/facemap-mac/model_params.pth"  
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
