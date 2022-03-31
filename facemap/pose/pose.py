import os
import time
from tkinter import N
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import pickle
from io import StringIO

from facemap import utils

from . import FMnet_torch, pose_helper_functions as pose_utils
from . import transforms, models, model_training

"""
Base class for generating pose estimates.
Contains functions that can be used through CLI or GUI
Currently supports single video processing and multi-videos as processed sequentially.
"""

# TO-DO:
# Write a function to re-train the model
# Add a feature for using the retrained model for the next pose prediction

class Pose():
    def __init__(self, filenames=None, bodyparts=None, bbox=[], bbox_set=False, gui=None, GUIobject=None):
        self.gui = gui
        self.GUIobject = GUIobject
        if self.gui is not None:
            self.filenames = self.gui.filenames
        else:
            self.filenames = filenames
        self.cumframes, self.Ly, self.Lx, self.containers = utils.get_frame_details(self.filenames)
        self.nframes = self.cumframes[-1]
        self.pose_labels = None
        self.bodyparts = bodyparts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bbox = bbox
        self.bbox_set = bbox_set
        self.net =  None

    def pose_prediction_setup(self):
        # Setup the model
        if self.net is None:
            self.net = self.load_model()
        # Setup the bounding box
        if not self.bbox_set:
            resize = True 
            for i in range(len(self.Ly)):
                x1, x2, y1, y2 = 0, self.Ly[i], 0, self.Lx[i]
                self.bbox.append([x1, x2, y1, y2, resize])
                prompt = "No bbox set. Using entire frame view: {} and resize={}".format(self.gui.bbox, resize)
                utils.update_mainwindow_message(MainWindow=self.gui, GUIobject=self.GUIobject, 
                                                prompt=prompt, hide_progress=True)
            self.bbox_set = True   

    def run_all(self, plot=True):
        start_time = time.time()
        self.pose_prediction_setup()
        for video_id in range(len(self.bbox)):
            utils.update_mainwindow_message(MainWindow=self.gui, GUIobject=self.GUIobject, 
                                    prompt="Processing video: {}".format(self.filenames[0][video_id]), hide_progress=True)
            pred_data, metadata = self.predict_landmarks(video_id)
            dataFrame = self.write_dataframe(pred_data)
            savepath = self.save_pose_prediction(dataFrame, video_id)
            utils.update_mainwindow_message(MainWindow=self.gui, GUIobject=self.GUIobject, 
                        prompt="Saved pose prediction outputs to: {}".format(savepath),  hide_progress=True)
            print("Saved pose prediction outputs to:", savepath)
            # Save metadata to a pickle file
            metadata_file = os.path.splitext(savepath)[0]+"_Facemap_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
            if self.gui is not None:
                self.gui.poseFilepath.append(savepath)
                self.gui.Labels_checkBox.setChecked(True)
                self.gui.start()
        if plot and self.gui is not None:
            self.plot_pose_estimates()
        end_time = time.time()
        print("Pose estimation time elapsed:", end_time-start_time, "seconds")
        utils.update_mainwindow_message(MainWindow=self.gui, GUIobject=self.GUIobject, 
                    prompt="Pose estimation time elapsed: {} seconds".format(end_time-start_time),  hide_progress=True)

    def run_subset(self):
        """
        Run pose estimation on a random subset of frames
        """
        self.pose_prediction_setup()
        # Select a random subset of frames
        subset_size = int(self.nframes/10)
        subset_ind = np.random.choice(self.nframes, subset_size, replace=False)
        subset_ind = np.sort(subset_ind)

        for video_id in range(len(self.bbox)):
            utils.update_mainwindow_message(MainWindow=self.gui, GUIobject=self.GUIobject, 
                                    prompt="Processing video: {}".format(self.filenames[0][video_id]), hide_progress=True)
            pred_data, _ = self.predict_landmarks(video_id, frame_ind=subset_ind)
            utils.update_mainwindow_message(MainWindow=self.gui, GUIobject=self.GUIobject, 
                        prompt="Finished processing subset of video",  hide_progress=True)
            return pred_data, subset_ind, video_id

    # Retrain model using refined pose data
    def retrain_model(self, pose_data, selected_frame_ind):
        video_path = self.filenames[0][0]
        imgs = model_training.load_images_from_video(video_path, selected_frame_ind)
        imgs, keypoints = model_training.preprocess_images_landmarks(imgs, pose_data, self.bbox[0])
        # Save images and landmarks to a numpy file
        savepath = os.path.splitext(video_path)[0]+"_Facemap_refined_images_landmarks.npy"
        np.save(savepath, {"imgs": imgs,
                            "keypoints": keypoints,
                            "bbox": self.bbox[0],
                            "bodyparts": self.bodyparts,
                            "frame_ind": selected_frame_ind})
        print("Preprocessed images and landmarks saved to:", savepath)
        #self.net = model_training.finetune_model(imgs, keypoints, self.net, batch_size=1)
        return
    
    def write_dataframe(self, data):
        scorer = "Facemap" 
        bodyparts = self.bodyparts 
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

        # Fill dataframe with data
        dataFrame.iloc[:,::3] = data[:,:,0].cpu().numpy()
        dataFrame.iloc[:,1::3] = data[:,:,1].cpu().numpy()
        dataFrame.iloc[:,2::3] = data[:,:,2].cpu().numpy()

        return dataFrame

    def predict_landmarks(self, video_id, frame_ind=None):
        """
        Predict labels for all frames in video and save output as .h5 file
        """
        nchannels = 1
        # TO-DO:
            # Support other batch sizes depending on GPU memory and CPU
        if torch.cuda.is_available():
            batch_size = 1
        else:
            batch_size = 1

        if frame_ind is None:
            total_frames = self.cumframes[-1]
            frame_ind = np.arange(total_frames)
        else:
            total_frames = len(frame_ind)

        # Create array for storing predictions
        pred_data = torch.zeros(total_frames, len(self.net.bodyparts), 3)

        # Store predictions in dataframe
        self.net.eval()
        start = 0
        end = batch_size
        Xstart, Xstop, Ystart, Ystop, resize = self.bbox[video_id]
        inference_time = 0

        progress_output = StringIO()
        with tqdm(total=total_frames, unit='frame', unit_scale=True, file=progress_output) as pbar:
            while start != total_frames: #  for analyzing entire video
                
                # Pre-pocess images
                imall = np.zeros((batch_size, nchannels, self.Ly[video_id], self.Lx[video_id]))
                if frame_ind is None:
                    cframes = np.arange(start, end)
                else:
                    cframes = np.array(frame_ind[start:end])
                utils.get_frames(imall, self.containers, cframes, self.cumframes)

                # Inference time includes: pre-processing, inference, post-processing
                t0 = time.time()
                imall = torch.from_numpy(imall).to(self.net.device, dtype=torch.float32)
                frame_grayscale = transforms.crop_resize(imall, Ystart, Ystop,
                                                        Xstart, Xstop, resize).clone().detach()
                imall = transforms.preprocess_img(frame_grayscale)

                # Network prediction 
                Xlabel, Ylabel, likelihood = pose_utils.get_predicted_landmarks(self.net, imall, batchsize=batch_size,
                                                                                smooth=False)

                # Get adjusted landmarks that fit to original image size
                Xlabel, Ylabel = transforms.labels_crop_resize(Xlabel, Ylabel, 
                                                                Ystart, Xstart,
                                                                current_size=(256, 256), 
                                                                desired_size=(self.bbox[video_id][1]-self.bbox[video_id][0],
                                                                            self.bbox[video_id][3]-self.bbox[video_id][2]))

                # Assign predictions to dataframe
                pred_data[start:end, :, 0] = Xlabel
                pred_data[start:end, :, 1] = Ylabel
                pred_data[start:end, :, 2] = likelihood
                inference_time += time.time() - t0

                pbar.update(batch_size)
                start = end 
                end += batch_size
                end = min(end, total_frames)
                # Update progress bar for every 5% of the total frames
                if (end) % np.floor(total_frames*.05) == 0:
                    utils.update_mainwindow_progressbar(MainWindow=self.gui,
                                                        GUIobject=self.GUIobject, s=progress_output, 
                                                        prompt="Pose prediction progress:")

        if batch_size == 1:
            inference_speed = total_frames / inference_time
            print("Inference speed:", inference_speed, "fps")

        metadata = {"batch_size": batch_size,
                    "image_size": (self.Ly, self.Lx),
                    "bbox": self.bbox[video_id],
                    "total_frames": total_frames,
                    "bodyparts": self.net.bodyparts,
                    "inference_speed": inference_speed,
                    }
        return pred_data, metadata

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

    def load_model(self):
        """
        Load pre-trained UNet model for labels prediction 
        """
        model_params_file = models.get_model_params_path()       
        model_state_file = models.get_model_state_path()   
        if torch.cuda.is_available():
            print("Using cuda as device")
        else:
            print("Using cpu as device")
        print("LOADING MODEL....", model_params_file)
        utils.update_mainwindow_message(MainWindow=self.gui, GUIobject=self.GUIobject,
                                        prompt="Loading model... {}".format(model_params_file))
        model_params = torch.load(model_params_file, map_location=self.device)
        self.bodyparts = model_params['params']['bodyparts'] 
        channels = model_params['params']['channels']
        kernel_size = 3
        nout = len(self.bodyparts)  # number of outputs from the model
        net = FMnet_torch.FMnet(img_ch=1, output_ch=nout, labels_id=self.bodyparts, 
                                channels=channels, kernel=kernel_size, device=self.device)
        net.load_state_dict(torch.load(model_state_file, map_location=self.device))
        net.to(self.device);
        return net

