import numpy as np
import pyqtgraph as pg
from . import transforms
from . import UNet_helper_functions as UNet_utils
from .. import roi, utils
import torch
import cv2
import time, os
import pandas as pd

"""
Base class for generating pose estimates. Currently supports single video processing only.
"""
class Pose():
    def __init__(self, filenames, savepath=None, parent=None, bbox=None):
        self.cumframes, self.Ly, self.Lx, _ = utils.get_frame_details(filenames)
        self.pose_labels = None
        self.bodyparts = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.load_model()
        self.bbox = None 
        self.bbox_set = False
        self.draw_bbox()                            ## ~~~~~~~~~~~~~~~~    GUI function     ~~~~~~~~~~~~~~~~
        self.cropped_img_slice = self.get_img_slice()
        # Plot key points on all frames: predict, save, and plot!
        self.dataFrame = self.predict_landmarks()
        self.save_pose_prediction()
        self.plot_pose_labels()
        print("predicted labels plotted!")

    def predict_landmarks(self, save_prediction=True):
        """
        Predict labels for all frames in video and save output as .h5 file
        """
        scorer = "Facemap" 
        bodyparts = np.arange(self.net.nout) # temporary, switch to self.net.labels_id
        nchannels = 1
        batch_size = 1

        # Create an empty dataframe
        for index, bodypart in enumerate(bodyparts):
            columnindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y"]], 
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
        while end != 20: # self.cumframes[-1]:
            # Prepocess images
            im = np.zeros((batch_size, nchannels, img_y, img_x))
            for i, frame_ind in enumerate(np.arange(start,end)):
                frame = self.parent.get_frame(frame_ind)[0]     ## ~~~~~~~~~~~~~~~~    GUI function     ~~~~~~~~~~~~~~~~
                frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_grayscale_preprocessed = transforms.preprocess_img(frame_grayscale)
                im[i,0] = frame_grayscale_preprocessed[0][self.cropped_img_slice]    # used cropped section only
            
            # Network prediction 
            hm_pred = self.net(torch.tensor(im).to(device=self.net.DEVICE, dtype=torch.float32)) # convert to tensor and send to device
            del im
            
            # Get adjusted landmarks that fit to original image size
            landmarks = np.squeeze(UNet_utils.heatmap2landmarks(hm_pred.cpu().detach().numpy()))
            del hm_pred
            Xlabel, Ylabel = transforms.labels_resize(landmarks[:,0], landmarks[:,1], 
                                                    current_size=(img_x, img_y), desired_size=(orig_imgx, orig_imgy))
            landmarks = np.array(list(zip(Xlabel, Ylabel)))
            dataFrame.iloc[start:end] = landmarks.ravel()
            start = end 
            end += batch_size
        return dataFrame
        
    def save_pose_prediction(self):
        # Save prediction to .h5 file
        basename, filename = os.path.split(self.parent.filenames[0][0])
        videoname, _ = os.path.splitext(filename)
        self.parent.poseFilepath = os.path.join(basename, videoname+"_FacemapPose.h5")
        self.dataFrame.to_hdf(self.parent.poseFilepath, "df_with_missing", mode="w") 
        
    def plot_pose_labels(self):
        # Plot labels
        self.parent.poseFileLoaded = True
        self.parent.load_labels()
        self.parent.Labels_checkBox.setChecked(True)         
    
    def get_img_slice(self):
        x1, x2, y1, y2 = self.bbox
        slc = [slice(y1, y2), slice(x1, x2)]   # slice img_y and img_x
        return slc

        
    def load_model(self):
        """
        Load pre-trained UNet model for labels prediction 
        """
        # Replace w/ function that downloads model from server
        model_file = "/Users/Atika/Neuroverse/Janelia/facemap-mac/facemap/example_model_small"  
        self.bodyparts = None # data['landmarks']
        kernel_size = 3
        nbase = [1, 64, 64*2, 64*3, 64*4] # number of channels per layer
        nout = 14 #len(self.bodyparts)  # number of outputs
        net = UNet_utils.UNet_b(nbase, nout, kernel_size, labels_id=self.bodyparts)
        if torch.cuda.is_available():
            cpu_is_device = False
        else:
            cpu_is_device = True
        net.load_model(model_file, cpu=cpu_is_device)
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
            frame = self.parent.get_frame(ind)[0]       ## ~~~~~~~~~~~~~~~~    GUI function     ~~~~~~~~~~~~~~~~
            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_grayscale_preprocessed = transforms.preprocess_img(frame_grayscale)
            im[k] = frame_grayscale_preprocessed
        return im




