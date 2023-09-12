"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
from facemap.tongue.dataset import TongueDataset
import cv2, random
from scipy import ndimage
import numpy as np
import torch
from facemap.pose import pose_helper_functions as pose_utils
from facemap.pose import transforms
from facemap.pose import pose_helper_functions as pose_utils
import torch.nn.functional as F


def create_dataset(video_file, img_size=(256,256), train=False):
    imgs = get_img_from_video(video_file)
    bbox = [0, imgs.shape[1], 0, imgs.shape[2]] # Bounding box for the frames [x1, x2, y1, y2]
    dat = TongueDataset(imgs, bbox=bbox, img_size=img_size, train=train)
    return dat

def get_img_from_video(video_path):
    """Load movie and return as numpy array

    Args:
        filepath (str): path to movie file
    Returns:
        movie (ND-array): movie as numpy array
    """
    cap = cv2.VideoCapture(video_path)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frame_idx in range(framecount):
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
        else:
            print("Error reading frame")
    frames = np.array(frames)
    return frames

def preprocess_imgs(image_data, resize_shape=None,bbox=None):
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
        # 1. Crop image
        if bbox is not None:
            im = pose_utils.crop_image(im, bbox)
        # 2. Pad image to square
        im, _ = transforms.pad_img_to_square(im)
        # 3. Resize image to resize_shape for model input
        if resize_shape is not None:
            im = resize_image(im, resize_shape)
        imgs.append(im)
    return imgs
    
def resize_image(im, resize_shape):
    """
    Resize image to given height and width.
    Parameters
    ----------
    im : ND-array
        image of size [Ly x Lx]
    resize_shape : tuple of size (2,)
        desired shape of image
    Returns
    -------
    im : ND-array
        resized image of size [h x w]
    """
    h, w = resize_shape
    if im.ndim == 3:
        im = torch.unsqueeze(im, dim=0)
    elif im.ndim == 2:
        im = torch.unsqueeze(im, dim=0)
        im = torch.unsqueeze(im, dim=0)
    im = F.interpolate(im, size=(h, w), mode="bilinear", align_corners=True).squeeze(dim=0)
    return im
