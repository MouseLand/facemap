import os
import time
from sklearn.covariance import log_likelihood

from tqdm import tqdm

import cv2
from zmq import device
import numpy as np
import pandas as pd
import torch
import pickle

from .. import utils
from . import UNet_helper_functions as UNet_utils
from . import unet_torch
from . import transforms

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
            dataFrame, metadata = self.predict_landmarks(video_id)
            savepath = self.save_pose_prediction(dataFrame, video_id)
            print("Saved pose prediction to:", savepath)
            # Save metadata to a pickle file
            metadata_file = os.path.splitext(savepath)[0]+"_Facemap_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
            if self.gui is not None:
                self.gui.poseFilepath.append(savepath)
                self.gui.Labels_checkBox.setChecked(True)
                self.gui.start()
        print("~~~~~~~~~~~~~~~~~~~~~DONE~~~~~~~~~~~~~~~~~~~~~")
        print("Time taken:", time.time()-t0)
        if plot:
            self.plot_pose_estimates()

    def predict_landmarks(self, video_id):
        """
        Predict labels for all frames in video and save output as .h5 file
        """
        scorer = "Facemap" 
        bodyparts = self.net.bodyparts 
        nchannels = 1
        if torch.cuda.is_available():
            batch_size = 1
        else:
            batch_size = 1
        # Create an empty dataframe
        for index, bodypart in enumerate(bodyparts):
            columnindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y", "likelihood"]],
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
        print("Predicting pose for video:", self.filenames[0][video_id])
        start_time = time.time()

        self.net.eval()
        start = 0
        end = batch_size
        Xstart, Xstop, Ystart, Ystop, resize = self.bbox[video_id]
        
        img_load_time = 0
        preprocess_time = 0
        inference_time = 0
        postprocess_time = 0

        with tqdm(total=self.cumframes[-1], unit='frame', unit_scale=True) as pbar:
            #while start <500:  # for checking bbox
            while start != self.cumframes[-1]: #  for analyzing entire video
                
                # Pre-pocess images
                t0 = time.time()
                imall = np.zeros((end-start, nchannels, 256, 256))
                cframes = np.arange(start, end)
                utils.get_frames(imall, self.containers, cframes, self.cumframes)
                img_load_time += time.time() - t0
                t0 = time.time()
                frame_grayscale = torch.tensor(transforms.crop_resize(imall.squeeze(), Ystart, Ystop,
                                Xstart, Xstop, resize)).to(self.net.device, dtype=torch.float32)
                imall = transforms.preprocess_img(frame_grayscale)
                preprocess_time += time.time() - t0

                t0 = time.time()
                # Network prediction 
                Xlabel, Ylabel, likelihood = UNet_utils.get_predicted_landmarks(self.net, imall, 
                                                                        batchsize=batch_size, smooth=False)
                inference_time += time.time() - t0

                t0 = time.time()
                # Get adjusted landmarks that fit to original image size
                Xlabel, Ylabel = transforms.labels_crop_resize(Xlabel, Ylabel, 
                                                                Ystart, Xstart,
                                                                current_size=(256, 256), 
                                                                desired_size=(self.bbox[video_id][1]-self.bbox[video_id][0],
                                                                            self.bbox[video_id][3]-self.bbox[video_id][2]))
                postprocess_time += time.time() - t0

                dataFrame.iloc[start:end,::3] = Xlabel.cpu().numpy()
                dataFrame.iloc[start:end,1::3] = Ylabel.cpu().numpy()
                dataFrame.iloc[start:end,2::3] = likelihood.cpu().numpy()
                pbar.update(batch_size)

                start = end 
                end += batch_size
                end = min(end, self.cumframes[-1])

        stop = time.time()
        time_elapsed = stop - start_time
        if batch_size == 1:
            print("Elapsed time:", time_elapsed)
            print("cumframes:", self.cumframes[-1])
            print("Total speed:", self.cumframes[-1]/time_elapsed, "fps")
            print("Image load speed:", self.cumframes[-1]/img_load_time, "fps")
            print("Preprocess speed:", self.cumframes[-1]/preprocess_time, "fps")
            print("Inference speed:", self.cumframes[-1]/inference_time, "fps")
            print("Postprocess speed:", self.cumframes[-1]/postprocess_time, "fps")

        metadata = {"batch_size": batch_size,
                    "run_duration": time_elapsed,
                    "image_size": (self.Ly, self.Lx),
                    "bbox": self.bbox[video_id],
                    "total_frames": self.cumframes[-1],
                    "bodyparts": bodyparts,
                    "inference_speed": self.cumframes[-1]/time_elapsed
                    }
        return dataFrame, metadata

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
        model_file = os.getcwd()+"/facemap/pose/facemap_model_state.pt"
        model_params_file = os.getcwd()+"/facemap/pose/facemap_model_params.pth"
        print("LOADING MODEL....", model_file)
        model_params = torch.load(model_params_file)
        self.bodyparts = model_params['params']['bodyparts'] 
        channels = model_params['params']['channels']
        kernel_size = 3
        nout = len(self.bodyparts)  # number of outputs
        net = unet_torch.FMnet(img_ch=1, output_ch=nout, labels_id=self.bodyparts, 
                                channels=channels, kernel=kernel_size, device=self.device)
        if torch.cuda.is_available():
            cpu_is_device = False
            print("Using cuda as device")
        else:
            cpu_is_device = True
            print("Using cpu as device")
        net.load_state_dict(torch.load(model_file))#(model_file, cpu=cpu_is_device)
        net.to(self.device);
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
