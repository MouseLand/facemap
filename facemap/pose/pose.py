import os
import time

from numpy.core.fromnumeric import argmax

from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import torch

from .. import utils
from . import UNet_helper_functions as UNet_utils
from . import unet_torch
from . import transforms

import time

"""
Base class for generating pose estimates using command line interface.
Currently supports single video processing only.
"""
class Pose():
    def __init__(self, gui=None, filenames=None, bbox=[], bbox_set=False):
        self.gui = gui
        if self.gui is not None:
            self.filenames = self.gui.filenames
        else:
            self.filenames = filenames
        self.cumframes, self.Ly, self.Lx, self.containers = utils.get_frame_details(self.filenames)
        self.nframes = self.cumframes[-1]
        self.pose_labels = None
        self.bodyparts = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.load_model()
        self.bbox = bbox
        self.bbox_set = bbox_set

    def run(self, plot=True):
        # Predict and save pose
        if not self.bbox_set:
            resize = True 
            for i in range(len(self.Ly)):
                x1, x2, y1, y2 = 0, self.Ly[i], 0, self.Lx[i]
                self.bbox.append([x1, x2, y1, y2, resize])
            print("No bbox set. Using full image size:", self.bbox)
            self.bbox_set = True    
        t0 = time.time()
        for video_id in range(len(self.bbox)):
            print("Processing video:", self.filenames[0][video_id])
            dataFrame = self.predict_landmarks(video_id)
            savepath = self.save_pose_prediction(dataFrame, video_id)
            if self.gui is not None:
                self.gui.poseFilepath.append(savepath)
        print("~~~~~~~~~~~~~~~~~~~~~DONE~~~~~~~~~~~~~~~~~~~~~")
        print("Time taken:", time.time()-t0)
        if plot:
            self.plot_pose_estimates()

    def predict_landmarks(self, video_id, verbose=True):
        """
        Predict labels for all frames in video and save output as .h5 file
        """
        scorer = "Facemap" 
        bodyparts = self.net.labels_id 
        nchannels = 1
        if torch.cuda.is_available():
            batch_size = 1
        else:
            batch_size = 1
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
        Xstart, Xstop, Ystart, Ystop, resize = self.bbox[video_id]
        img_time = []
        with tqdm(total=100, unit='frame', unit_scale=True) as pbar:
            im = torch.zeros((end-start, nchannels, 256, 256), device=self.device)
            while start<=100:#!= self.cumframes[-1]:
                # Pre-pocess images
                for i, frame_ind in enumerate(np.arange(start,end)):
                    frame = utils.get_frame(frame_ind, self.nframes, self.cumframes, self.containers)[video_id]  
                    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_grayscale = transforms.crop_resize(frame_grayscale.squeeze(), Ystart, Ystop,
                                                    Xstart, Xstop, resize)
                    im[i,0] = transforms.preprocess_img(frame_grayscale, device=self.device) 

                # Network prediction 
                hm_pred, locref_pred = self.net(im) # convert to tensor and send to device

                # Post-process predictions
                hm_pred = UNet_utils.gaussian_smoothing(hm_pred, sigmoid=True, device=self.device)
                pose = UNet_utils.argmax_pose_predict_batch(hm_pred.cpu().detach().numpy(), 
                                                                locref_pred.cpu().detach().numpy(),
                                                                UNet_utils.STRIDE)
                # Get adjusted landmarks that fit to original image size
                landmarks = pose[:,:,:2]#.squeeze()
                likelihood = pose[:,:,-1]#.squeeze()
                Xlabel, Ylabel = transforms.labels_crop_resize(landmarks[:,:,0], landmarks[:,:,1], 
                                                                Ystart, Xstart,
                                                                current_size=(256, 256), 
                                                                desired_size=(Xstop-Xstart, Ystop-Ystart))
                # Save predictions to dataframe
                dataFrame.iloc[start:end,::3] = Xlabel
                dataFrame.iloc[start:end,1::3] = Ylabel
                dataFrame.iloc[start:end,2::3] = likelihood
                
                # Update progress bar
                start = end 
                end += batch_size
                end = min(end, self.cumframes[-1])
                pbar.update(batch_size)

            if verbose:
                print("img proc", np.round(1/np.mean(img_time),2))
        return dataFrame

    def save_pose_prediction(self, dataFrame, video_id):
        # Save prediction to .h5 file
        if self.gui is not None:
            basename = self.gui.save_path
            _, filename = os.path.split(self.filenames[0][video_id])
            videoname, _ = os.path.splitext(filename)
        else:
            basename, filename = os.path.split(self.filenames[0][video_id])
            videoname, _ = os.path.splitext(filename)
        poseFilepath = os.path.join(basename, videoname+"_FacemapPose.h5")
        dataFrame.to_hdf(poseFilepath, "df_with_missing", mode="w")
        return poseFilepath  
    
    def plot_pose_estimates(self):
        # Plot labels
        self.gui.poseFileLoaded = True
        self.gui.load_labels()
        self.gui.Labels_checkBox.setChecked(True)    
        
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

    def load_model(self):
        """
        Load pre-trained UNet model for labels prediction 
        """
        # Replace following w/ a function that downloads model from a server
        model_file = os.getcwd()+"/model_state.pt"
        model_params_file = os.getcwd()+"/model_params.pth"
        model_params = torch.load(model_params_file)
        self.bodyparts = model_params['landmarks'] 
        kernel_size = 3
        nout = len(self.bodyparts)  # number of outputs
        net = unet_torch.FMnet(img_ch=1, output_ch=nout, labels_id=self.bodyparts, 
                                filters=64, kernel=kernel_size, device=self.device)
        if torch.cuda.is_available():
            cpu_is_device = False
            print("Using cuda as device")
        else:
            cpu_is_device = True
            print("Using cpu as device")
        net.load_model(model_file, cpu=cpu_is_device)
        net.to(self.device)
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
