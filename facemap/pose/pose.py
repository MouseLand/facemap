"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os
import pickle
import time
from io import StringIO

import h5py
import numpy as np
import torch
from tqdm import tqdm

from facemap import utils

from . import datasets, facemap_network, model_loader
from . import pose_helper_functions as pose_utils
from . import transforms

try:
    from . import model_training
except:
    train = False

"""
Base class for generating pose estimates.
Contains functions that can be used through CLI or GUI
Currently supports single video processing, whereas multi-view videos recorded simultaneously are processed sequentially.
"""


class Pose:
    """Pose estimation for single video processing

    Parameters
    ----------
    filenames: 2D-list
        List of filenames to be processed.
    bbox: list
        Bounding box for cropping the video [x1, x2, y1, y2]. If not set, the entire frame is used.
    bbox_set: bool
        Flag to indicate whether the bounding box has been set. Default is False.
    resize: bool
        Flag to indicate whether the video needs to be resized
    add_padding: bool
        Flag to indicate whether the video needs to be padded. Default is False.
    gui: object
        GUI object.
    GUIobject: object
        GUI mainwindow object.
    net: object
        PyTorch model object.
    model_name: str
        Name of the model to be used for pose estimation. Default is None which uses the pre-trained model.
    """

    def __init__(
        self,
        filenames=None,
        bbox=[],
        bbox_set=False,
        resize=False,
        add_padding=False,
        gui=None,
        GUIobject=None,
        net=None,
        model_name=None,
    ):
        self.gui = gui
        self.GUIobject = GUIobject
        if self.gui is not None:
            self.filenames = self.gui.filenames
            self.batch_size = self.gui.batch_size_spinbox.value()
        else:
            self.filenames = filenames
            self.batch_size = 1
        self.cumframes, self.Ly, self.Lx, self.containers = utils.get_frame_details(
            self.filenames
        )
        self.nframes = self.cumframes[-1]
        self.pose_labels = None
        if gui is not None:
            self.device = self.gui.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bbox = bbox
        self.bbox_set = bbox_set
        self.resize = resize
        self.add_padding = add_padding
        self.net = net
        self.model_name = model_name
        self.bodyparts = [
            "eye(back)",
            "eye(bottom)",
            "eye(front)",
            "eye(top)",
            "lowerlip",
            "mouth",
            "nose(bottom)",
            "nose(r)",
            "nose(tip)",
            "nose(top)",
            "nosebridge",
            "paw",
            "whisker(I)",  # "whisker(c1)",
            "whisker(III)",  # "whisker(d2)",
            "whisker(II)",  # "whisker(d1)",
        ]

    def pose_prediction_setup(self):
        # Setup the model
        self.load_model()
        # Setup the bounding box
        if not self.bbox_set:
            for i in range(len(self.Ly)):
                x1, x2, y1, y2 = 0, self.Ly[i], 0, self.Lx[i]
                self.bbox.append([x1, x2, y1, y2])

                # Update resize and add padding flags
                if x2 - x1 != y2 - y1:  # if not a square frame view then add padding
                    self.add_padding = True
                if x2 - x1 != 256 or y2 - y1 != 256:  # if not 256x256 then resize
                    self.resize = True
                prompt = (
                    "No bbox set. Using entire frame view: {} and resize={}".format(
                        self.bbox, self.resize
                    )
                )
                utils.update_mainwindow_message(
                    MainWindow=self.gui,
                    GUIobject=self.GUIobject,
                    prompt=prompt,
                    hide_progress=True,
                )
            self.bbox_set = True

    def set_model(self, model_selected=None):
        """Set model to use for pose estimation

        Args:
            model_selected (str, optional): Path of trained model weights to use. Default value of None sets the model_name to Base model/facemap_model_state which uses pre-trained weights.
        """
        if model_selected is None:
            if self.gui is None:
                model_selected = "Base model"
            else:
                model_selected = self.gui.pose_model_combobox.currentText()
        # Get all model names
        model_paths = model_loader.get_model_states_paths()
        if len(model_paths) == 0:  # No models found, set default model
            self.model_name = model_loader.get_basemodel_state_path()
        model_names = [os.path.splitext(os.path.basename(m))[0] for m in model_paths]
        for model in model_names:  # Find selected model and update model name
            if (model == model_selected) or (
                model_selected == "Base model" and "facemap_model_state" in model
            ):
                print("Setting model name to:", model)
                self.model_name = model_paths[model_names.index(model)]
                break
        print("Loading model state from:", self.model_name)
        self.net.load_state_dict(torch.load(self.model_name, map_location=self.device, weights_only=False))
        self.net.to(self.device)

    def load_model(self):
        """
        Load model for keypoints prediction. Uses default model unless set_model is used to update the model name and load the model state.
        """
        model_params_file = model_loader.get_model_params_path()
        print("{} set as device".format(self.device))
        print("Loading model parameters from:", model_params_file)
        utils.update_mainwindow_message(
            MainWindow=self.gui,
            GUIobject=self.GUIobject,
            prompt="Loading model... {}".format(model_params_file),
        )
        model_params = torch.load(model_params_file, map_location=self.device, weights_only=False)
        # self.bodyparts = model_params["params"]["bodyparts"]
        channels = model_params["params"]["channels"]
        kernel_size = 3
        nout = len(self.bodyparts)  # number of outputs from the model
        self.net = facemap_network.FMnet(
            img_ch=1,
            output_ch=nout,
            labels_id=self.bodyparts,
            channels=channels,
            kernel=kernel_size,
            device=self.device,
        )
        if self.model_name is None:
            self.set_model()
        else:
            self.set_model(self.model_name)

    def train(
        self,
        image_data,
        keypoints_data,
        num_epochs,
        batch_size,
        learning_rate,
        weight_decay,
        bbox,
    ):
        """
        Train the model
        Parameters
        ----------
        image_data: ND-array
            Array of images of shape (nframes, Ly, Lx)
        keypoints_data: ND-array
            Array of keypoints of shape (nframes, nkeypoints, 2)
        num_epochs: int
            Number of epochs for training
        batch_size: int
            Batch size for training
        learning_rate: float
            Learning rate for training
        weight_decay: float
            Weight decay for training
        Returns
        -------
        model: torch.nn.Module
            Trained/finetuned model
        """
        # Create a dataset object for training
        dataset = datasets.FacemapDataset(
            image_data=image_data,
            keypoints_data=keypoints_data,
            bbox=bbox,
        )
        # Create a dataloader object for training
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        # Use preprocessed data to train the model
        self.net = model_training.train(
            dataloader,
            self.net,
            num_epochs,
            learning_rate,
            weight_decay,
            gui=self.gui,
            gui_obj=self.GUIobject,
        )
        print("Model training complete!")
        return self.net

    def predict_landmarks(self, video_id, frame_ind=None):
        """
        Predict keypoints/landmarks for all frames in video and save output as .h5 file. If frame_ind is specified, only predict keypoints/landmarks for those frames.
        Parameters
        ----------
        video_id: int
            Index of video in filenames list to be used for prediction.
        frame_ind: list
            List of frame indices for keypoints/landmarks prediction.
        """
        nchannels = 1
        if frame_ind is None:
            total_frames = self.cumframes[-1]
            frame_ind = np.arange(total_frames)
        else:
            total_frames = len(frame_ind)

        # Create array for storing predictions
        pred_data = torch.zeros(total_frames, len(self.bodyparts), 3)

        # Store predictions in dataframe
        self.net.eval()
        start = 0
        end = self.batch_size
        # Get bounding box for the video
        y1, _, x1, _ = self.bbox[video_id]
        inference_time = 0

        print("Using params:")
        print("\tbbox:", self.bbox[video_id])
        print("\tbatch size:", self.batch_size)
        print("\tresize:", self.resize)
        print("\tpadding:", self.add_padding)
        # FIXME: Plotting keypoints after batch processing is not working properly
        progress_output = StringIO()
        with tqdm(
            total=total_frames, unit="frame", unit_scale=True, file=progress_output
        ) as pbar:
            while start != total_frames:  #  for analyzing entire video
                # Pre-pocess images
                imall = np.zeros(
                    (self.batch_size, nchannels, self.Ly[video_id], self.Lx[video_id])
                )
                cframes = np.array(frame_ind[start:end])
                # utils.get_frames(imall, self.containers, cframes, self.cumframes)
                imall = utils.get_batch_frames(
                    cframes,
                    total_frames,
                    self.cumframes,
                    self.containers,
                    video_idx=video_id,
                    grayscale=True,
                )
                # Inference time includes: pre-processing, inference, post-processing
                t0 = time.time()

                # Pre-process images
                imall, postpad_shape, pads = transforms.preprocess_img(
                    imall,
                    self.bbox[video_id],
                    self.add_padding,
                    self.resize,
                    device=self.net.device,
                )

                # Run inference
                xlabels, ylabels, likelihood = pose_utils.predict(
                    self.net, imall, smooth=False
                )

                xlabels, ylabels = transforms.adjust_keypoints(
                    xlabels,
                    ylabels,
                    crop_xy=(x1, y1),
                    padding=pads,
                    current_size=(256, 256),
                    desired_size=postpad_shape,
                )

                # Add predictions to array
                pred_data[start:end, :, 0] = xlabels
                pred_data[start:end, :, 1] = ylabels
                pred_data[start:end, :, 2] = likelihood

                # Update progress bar and inference time
                inference_time += time.time() - t0
                pbar.update(self.batch_size)
                start = end
                end += self.batch_size
                end = min(end, total_frames)
                # Update progress bar for every 5% of the total frames
                percent_frames = int(np.floor(total_frames * 0.05))
                if percent_frames != 0 and (end) % percent_frames == 0:
                    utils.update_mainwindow_progressbar(
                        MainWindow=self.gui,
                        GUIobject=self.GUIobject,
                        s=progress_output,
                        prompt="Pose prediction progress:",
                    )

        inference_speed = total_frames / inference_time
        print("Inference speed:", inference_speed, "fps")

        metadata = {
            "batch_size": self.batch_size,
            "image_size": (self.Ly, self.Lx),
            "bbox": self.bbox[video_id],
            "total_frames": total_frames,
            "bodyparts": self.bodyparts,
            "inference_speed": inference_speed,
        }
        return pred_data, metadata

    def save_model(self, model_filepath):
        """
        Save model to file
        Parameters
        ----------
        model_filepath: str
            Path to save model weights
        """
        torch.save(self.net.state_dict(), model_filepath)
        model_loader.copy_to_models_dir(model_filepath)
        return model_filepath

    def save_data_to_hdf5(self, data, video_id, selected_frame_ind=None):
        """save_data_to_hdf5: Save data to an HDF5 file

        Args:
            data (2D-array): Data to save (nframes x nbodyparts x 3)
            selected_frame_ind (list): Indices of selected frames
        """
        # Create a multi-index dict to store data in HDF5 file. First index is the scorer name, second index is the bodypart names, and third index is the coordinates (x, y, likelihood)
        scorer = "Facemap"
        bodyparts = self.bodyparts
        data_dict = {}
        data_dict[scorer] = {}
        if selected_frame_ind is None:
            indices = np.arange(self.cumframes[-1])
        else:
            indices = selected_frame_ind
        for index, bodypart in enumerate(bodyparts):
            data_dict[scorer][bodypart] = {}
            data_dict[scorer][bodypart]["x"] = data[:, index, 0][indices]
            data_dict[scorer][bodypart]["y"] = data[:, index, 1][indices]
            data_dict[scorer][bodypart]["likelihood"] = data[:, index, 2][indices]

        if self.gui is not None:
            basename = self.gui.save_path
            _, filename = os.path.split(self.filenames[0][video_id])
            videoname, _ = os.path.splitext(filename)
        else:
            basename, filename = os.path.split(self.filenames[0][video_id])
            videoname, _ = os.path.splitext(filename)

        hdf5_filepath = os.path.join(basename, videoname + "_FacemapPose.h5")
        with h5py.File(hdf5_filepath, "w") as f:
            self.save_dict_to_hdf5(f, "", data_dict)
        return hdf5_filepath

    def save_dict_to_hdf5(self, h5file, path, data_dict):
        """
        Saves dictionary to an HDF5 file. Adapted from https://github.com/talmolab/sleap/blob/391bc0421fe3820ddd6b5d07e31311d60b129fe3/sleap/util.py#L116
        Calls itself recursively if items in dictionary are not
        `np.ndarray`, `np.int64`, `np.float64`, `str`, or bytes.
        Objects must be iterable.
        Args:
            h5file: The HDF5 filename object to save the data to.
                Assume it is open.
            path: The path to group save the dict under.
            data_dict: The dict containing data to save.
        Raises:
            ValueError: If type for item in dict cannot be saved.
        Returns:
            None
        """
        for key, item in list(data_dict.items()):
            if item is None:
                h5file[path + key] = ""
            elif isinstance(item, bool):
                h5file[path + key] = int(item)
            elif isinstance(item, list):
                items_encoded = []
                for it in item:
                    if isinstance(it, str):
                        items_encoded.append(it.encode("utf8"))
                    else:
                        items_encoded.append(it)

                h5file[path + key] = np.asarray(items_encoded)
            elif isinstance(item, (str)):
                h5file[path + key] = item.encode("utf8")
            elif isinstance(
                item, (np.ndarray, np.int64, np.float64, str, bytes, float)
            ):
                h5file[path + key] = item
            elif isinstance(item, dict):
                self.save_dict_to_hdf5(h5file, path + key + "/", item)
            elif isinstance(item, int):
                h5file[path + key] = item
            else:
                raise ValueError("Cannot save %s type" % type(item))

    def run(self):
        start_time = time.time()
        self.pose_prediction_setup()
        for video_id in range(len(self.filenames[0])):
            utils.update_mainwindow_message(
                MainWindow=self.gui,
                GUIobject=self.GUIobject,
                prompt="Processing video: {}".format(self.filenames[0][video_id]),
                hide_progress=True,
            )
            print("\nProcessing video: {}".format(self.filenames[0][video_id]))
            pred_data, metadata = self.predict_landmarks(video_id)

            # Save the data using h5py
            savepath = self.save_data_to_hdf5(pred_data.cpu().numpy(), video_id)

            utils.update_mainwindow_message(
                MainWindow=self.gui,
                GUIobject=self.GUIobject,
                prompt="Saved pose prediction outputs to: {}".format(savepath),
                hide_progress=True,
            )
            print("Saved keypoints:", savepath)
            # Save metadata to a pickle file
            metadata_file = os.path.splitext(savepath)[0] + "_metadata.pkl"
            with open(metadata_file, "wb") as f:
                pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
            print("Saved metadata:", metadata_file)
            if self.gui is not None:
                self.gui.poseFilepath.append(savepath)
        end_time = time.time()
        utils.update_mainwindow_message(
            MainWindow=self.gui,
            GUIobject=self.GUIobject,
            prompt="Pose estimation time elapsed: {} seconds".format(
                end_time - start_time
            ),
            hide_progress=True,
        )

    def run_subset(self, subset_ind=None):
        print("Using {} for pose estimation".format(self.model_name))
        if subset_ind is None:
            # Select a random subset of frames
            subset_size = int(self.nframes / 10)
            subset_ind = np.random.choice(self.nframes, subset_size, replace=False)
            # subset_ind = np.sort(subset_ind)

        utils.update_mainwindow_message(
            MainWindow=self.gui,
            GUIobject=self.GUIobject,
            prompt="Processing video: {}".format(self.filenames[0][0]),
            hide_progress=True,
        )
        pred_data, _ = self.predict_landmarks(0, frame_ind=subset_ind)
        utils.update_mainwindow_message(
            MainWindow=self.gui,
            GUIobject=self.GUIobject,
            prompt="Finished processing subset of video",
            hide_progress=True,
        )
        return pred_data, subset_ind, self.bbox

    def save_pose_prediction(self, dataFrame, video_id):
        # Save prediction to .h5 file
        if self.gui is not None:
            basename = self.gui.save_path
            _, filename = os.path.split(self.filenames[0][video_id])
            videoname, _ = os.path.splitext(filename)
        else:
            basename, filename = os.path.split(self.filenames[0][video_id])
            videoname, _ = os.path.splitext(filename)
        poseFilepath = os.path.join(basename, videoname + "_FacemapPose.h5")
        dataFrame.to_hdf(poseFilepath, "df_with_missing", mode="w")
        return poseFilepath

    def plot_pose_estimates(self):
        # Plot labels
        self.gui.is_pose_loaded = True
        self.gui.load_keypoints()
        self.gui.keypoints_checkbox.setChecked(True)
        self.gui.start()
