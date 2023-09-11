"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os
from glob import glob

import cv2
import numpy as np
import torch

from . import pose_helper_functions as pose_utils
from . import transforms

"""
Facemap dataset for training the Facemap model. 
----------
A dataset object to load image and keypoints data
"""


class FacemapDataset(torch.utils.data.Dataset):
    """
    FacemapDataset
    ----------
    A dataset object to load image and keypoints data
    """

    def __init__(
        self,
        datadir=None,
        image_data=None,
        keypoints_data=None,
        bbox=None,
        train=True,
        img_size=(256, 256),
        scorer="All",
    ):
        self.datadir = datadir
        self.scorer = scorer
        self.img_size = img_size
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
            "whisker(III)",  # "whisker(c2)",  # "whisker(d2)",
            "whisker(II)",  # "whisker(d1)",
        ]
        # Set image and keypoints data
        if datadir is None:
            self.images = self.preprocess_imgs(image_data)
            if keypoints_data is None:
                self.keypoints = None
            else:
                self.keypoints = torch.from_numpy(keypoints_data)
        else:  # Load data from directory - not used for GUI
            self.images = self.load_images()
            # self.keypoints = self.load_keypoints_h5()
        if self.keypoints is not None:
            self.num_keypoints = self.keypoints.shape[1]
        self.num_images = self.__len__()
        # Set bounding box
        if bbox is not None:
            # Create a list of bounding boxes for each image by repeating the bbox for each image
            self.bbox = torch.from_numpy(np.tile(bbox, (self.num_images, 1)))
        else:
            self.bbox = self.estimate_bbox_from_keypoints()

        if train:
            self.augmentation = True
        else:
            self.augmentation = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        """
        Get the image and keypoints for the item confined in the img_size range
        Parameters
        ----------
        item : int
            Index of the item.
        Returns
        -------
        image : ND-array of shape (C, W, H)
            Image.
        keypoints : ND-array of shape (15, 2)
            Keypoints.
        """
        image, keypoints = self.preprocess_data(
            self.images[item].clone().detach(),
            self.keypoints[item].clone().detach(),
            self.bbox[item].clone().detach(),
        )

        if self.augmentation:
            image, keypoints = transforms.augment_data(image, keypoints)

        # If not a tensor, convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.from_numpy(keypoints)

        data = {
            "image": image,
            "keypoints": keypoints,
            "bbox": self.bbox[item],
            "item": item,
        }
        return data

    def preprocess_imgs(self, image_data):
        """
        Preprocess images to be in the range [0, 1] and normalize99
        Parameters
        ----------
        image_data : list of ND-array of shape (C, W, H)
            List of images.
        Returns
        -------
        image_data : list of ND-array of shape (C, W, H)
            List of images.
        """
        imgs = []
        for im in image_data:
            im = torch.from_numpy(im)
            # Normalize
            im = pose_utils.normalize99(im)
            imgs.append(im)
        return imgs

    def preprocess_data(self, image, keypoints, bbox):
        """
        Preproccesing of image involves:
            1. Cropping image to select bounding box (bbox) region
            2. Padding image size to be square
            3. Resize image to Lx x Ly for model input
        Parameters
        -------------
        image: ND-array
            image of size [(Lz) x Ly x Lx]
        add_padding: bool
            whether to add padding to image
        resize: bool
            whether to resize image
        bbox: tuple of size (4,)
            bounding box positions in order x1, x2, y1, y2
        Returns
        --------------
        image: ND-array
            preprocessed image of size [1 x Ly x Lx]
        """
        # 1. Crop image
        # if self.augmentation: #randomize bbox/cropping during training
        bbox = transforms.randomize_bbox_coordinates(bbox, image.shape[-2:])
        image = transforms.crop_image(image, bbox)
        y1, _, x1, _ = bbox
        keypoints[:, 0] = keypoints[:, 0] - x1
        keypoints[:, 1] = keypoints[:, 1] - y1

        # 2. Pad image to square
        image, (pad_y_top, _, pad_x_left, _) = transforms.pad_img_to_square(image)
        keypoints = transforms.pad_keypoints(keypoints, pad_y_top, pad_x_left)

        # 3. Resize image to resize_shape for model input
        keypoints = transforms.resize_keypoints(
            keypoints, desired_shape=self.img_size, original_shape=image.shape[-2:]
        )
        image = transforms.resize_image(image, self.img_size)

        return image, keypoints

    def load_images(self):
        """
        Load images from the directory or subdirectories provided containing .png files and convert to float32 and normalize99
        Returns
        -------
        images : List ND-arrays of shape (C, W, H)
            List of images.
        """
        # Check if the directory contains .png files
        img_files = sorted(glob(os.path.join(self.datadir, "*.png")))
        if len(img_files) == 0:  # If not, check if it contains subdirectories
            img_files = sorted(glob(os.path.join(self.datadir, "*/*.png")))
        if len(img_files) == 0:
            raise ValueError("No .png files found in the directory")

        imgs = []
        for file in img_files:
            # Normalize images
            im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            # Add channel dimension
            im = im[np.newaxis, ...]
            # Convert numpy array to tensor
            im = torch.from_numpy(im)

            # Convert to float32 in the range 0 to 1
            if im.dtype == float:
                pass
            elif im.dtype == torch.uint8:
                im = im.float() / 255.0
            elif im.dtype == torch.uint16:
                im = im.float() / 65535.0
            else:
                print("Cannot handle im type " + str(im.dtype))
                raise TypeError

            # Normalize
            im = pose_utils.normalize99(im)
            imgs.append(im)

        return imgs

    """
    def load_keypoints_h5(self):
        """ """
        Load landmarks/keypoints from the directory provided containing .h5 files.
        Returns
        -------
        landmarks : pandas multi-index dataframe
            Dataframe containing the following:
                scorer: str
                    Name of the scorer.
                bodyparts: str
                    Name of the bodyparts. Contains a total of 15 bodyparts.
                x-coord: float
                    x-coordinate of the landmark.
                y-coord: float
                    y-coordinate of the landmark.
        """ """
        # Landmarks/key points info
        annotation_files = sorted(
            glob(os.path.join(self.datadir, "CollectedData_{}.h5".format(self.scorer)))
        )
        if len(annotation_files) == 0:
            annotation_files = sorted(
                glob(
                    os.path.join(
                        self.datadir, "*/CollectedData_{}.h5".format(self.scorer)
                    )
                )
            )
        if len(annotation_files) == 0:
            raise ValueError("No .h5 files found in the directory")

        # Landmarks dataframe concatentation
        landmarks = pd.DataFrame()
        for f in annotation_files:
            df = pd.read_hdf(f)
            # Remove likelihood column
            df = df.T[df.columns.get_level_values("coords") != "likelihood"].T
            df = self.fix_labels(df, scorer=self.scorer)
            landmarks = pd.concat([landmarks, df])
        landmarks = landmarks.to_numpy().reshape(-1, 15, 2)
        # Convert to tensor
        landmarks = torch.from_numpy(landmarks)
        return landmarks
    """

    def estimate_bbox_from_keypoints(self):
        """
        Return a list of bounding boxes from a list of landmarks for each image.
        Returns
        -------
        bbox: list of size [N x 4]
            bounding box positions in order x1, x2, y1, y2 for each image
        """
        bbox = []
        for i in range(self.keypoints.shape[0]):
            x_min = np.nanmin(self.keypoints[i, :, 0])
            x_max = np.nanmax(self.keypoints[i, :, 0])
            y_min = np.nanmin(self.keypoints[i, :, 1])
            y_max = np.nanmax(self.keypoints[i, :, 1])
            bbox.append([y_min, y_max, x_min, x_max])
        # Convert to tensor
        bbox = torch.from_numpy(np.array(bbox))
        return bbox

    """
    def fix_labels(self, df, scorer):
        # Change scorer label to All
        df = df.rename(columns=dict(zip(df.columns.levels[0], [scorer])), level=0)

        # Rename labels to merge points from different views
        for label in pd.unique(df.columns.get_level_values("bodyparts")):
            label_lower = label.lower()
            if "eye" in label_lower:  # ~~~ Eye points ~~~
                if "back" in label_lower:
                    df = df.rename(columns={label: "eye(back)"})
                elif "bottom" in label_lower:
                    df = df.rename(columns={label: "eye(bottom)"})
                elif "front" in label_lower:
                    df = df.rename(columns={label: "eye(front)"})
                elif "top" in label_lower:
                    df = df.rename(columns={label: "eye(top)"})
            elif "tear" in label_lower:  # Teargland -> eye(front)
                df = df.rename(columns={label: "eye(front)"})
            elif "nose" in label_lower:  # ~~~ Nose points ~~~
                nose_points = [
                    kp in label_lower
                    for kp in ["(r)", "(l)", "tip", "top", "bottom", "bridge"]
                ]
                if sum(nose_points) == 1:
                    if "tip" in label_lower:
                        df = df.rename(columns={label: "nose(tip)"})
                    elif "(l)" in label_lower:
                        df = df.rename(columns={label: "nose(bottom)"})
                    else:
                        df = df.rename(columns={label: label_lower})
            elif "whisker" in label_lower:  # ~~~ Whisker points ~~~
                if "c1" in label_lower or "u1" in label_lower:
                    df = df.rename(columns={label: "whisker(I)"})  # "whisker(c1)"})
                elif "c2" in label_lower or "u2" in label_lower:
                    df = df.rename(columns={label: "whisker(III)"})  # "whisker(c2)"})
                elif "d1" in label_lower:
                    df = df.rename(columns={label: "whisker(II)"})  # "whisker(d1)"})
            elif "mouth" in label_lower:  # ~~~ Mouth points ~~~~
                df = df.rename(columns={label: "mouth"})
            elif "lowerlip" in label_lower:
                df = df.rename(columns={label: label_lower})
            elif "paw" in label_lower:  # ~~~ Paw points ~~~~
                df = df.rename(columns={label: label_lower})

        # Remove any labels not required
        for adjusted_label in pd.unique(df.columns.get_level_values("bodyparts")):
            if adjusted_label not in self.bodyparts:
                print("remove %s" % adjusted_label)
                df = df.drop(columns=[adjusted_label], level=1)
            df = df[sorted(df)]
        return df
    """
