"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
"""
Facemap functions for: 
- bounding box: (suggested ROI) for UNet input images
- image preprocessing 
- image augmentation
"""
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from . import pose_helper_functions as pose_utils


def preprocess_img(im, bbox, add_padding, resize, device=None):
    """
    Preproccesing of image involves:
        1. Conversion to float32 and normalize99
        2. Cropping image to select bounding box (bbox) region
        3. padding image size to be square
        4. Resize image to 256x256 for model input
     Parameters
    -------------
    im: ND-array
        image of size [(Lz) x Ly x Lx]
    bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2
    add_padding: bool
        whether to add padding to image
    resize: bool
        whether to resize image
     Returns
    --------------
    im: ND-array
        preprocessed image of size [1 x Ly x Lx] if input dimensions==2, else [Lz x Ly x Lx]
    postpad_shape: tuple of size (2,)
        shape of padded image
    pads: tuple of size (4,)
        padding values for (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right)
    """
    # 1. Convert to float32 in range 0-1
    if im.ndim == 2:
        im = im[np.newaxis, ...]

    # Convert numpy array to tensor
    if device is not None:
        im = torch.from_numpy(im)
        if device.type != "cpu":
            im = im.pin_memory()
        im = im.to(device, dtype=torch.float32)

    # Normalize
    im = pose_utils.normalize99(im, device=device)

    # 2. Crop image
    im = crop_image(im, bbox)

    # 3. Pad image to square
    if add_padding:
        im, pads = pad_img_to_square(im, bbox)
    else:
        pads = (0, 0, 0, 0)

    # 4. Resize image to 256x256 for model input
    postpad_shape = im.shape[-2:]
    if resize:
        im = F.interpolate(im, size=(256, 256), mode="bilinear")

    return im, postpad_shape, pads


def randomize_bbox_coordinates(bbox, im_shape, random_factor_range=(0.1, 0.3)):
    """
    Randomize bounding box by a random amount to increase/expand the bounding box region while staying within the image region.
    Parameters
    ----------
    bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2
    im_shape: tuple of size (2,)
        image shape in order Ly, Lx
    random_factor_range: tuple of size (2,)
        range of random factor to use for expaning bounding box
    Returns
    -------
    bbox: tuple of size (4,)
        randomized bounding box positions in order x1, x2, y1, y2
    """
    bbox = np.array(bbox)
    x1, x2, y1, y2 = bbox
    x_range = x2 - x1
    y_range = y2 - y1
    x_min = int(x1 - x_range * get_random_factor(random_factor_range))
    x_max = int(x2 + x_range * get_random_factor(random_factor_range))
    y_min = int(y1 - y_range * get_random_factor(random_factor_range))
    y_max = int(y2 + y_range * get_random_factor(random_factor_range))
    x_min = max(0, x_min)
    x_max = min(im_shape[0], x_max)
    y_min = max(0, y_min)
    y_max = min(im_shape[1], y_max)
    bbox = np.array([x_min, x_max, y_min, y_max])
    return bbox


def get_random_factor(factor_range):
    """
    Get a random factor within the range provided.
    Parameters
    ----------
    factor_range: tuple of size (2,)
        factor range in order min, max
    Returns
    -------
    factor: float
        random factor
    """
    factor = np.random.uniform(factor_range[0], factor_range[1])
    return factor


def get_cropped_imgs(imgs, bbox):
    """
    Preproccesing of image involves: conversion to float32 in range0-1, normalize99, and padding image size to be
    compatible with UNet model input
    Parameters
    -------------
    imgs: ND-array
        images of size [batch_size x nchan x Ly x Lx]
    bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2
    Returns
    --------------
    cropped_imgs: ND-array
        images of size [batch_size x nchan x Ly' x Lx'] where Ly' = y2-y1 and Lx'=x2-x1
    """
    x1, x2, y1, y2 = (np.round(bbox)).astype(int)
    batch_size = imgs.shape[0]
    nchannels = imgs.shape[1]
    cropped_imgs = np.empty((batch_size, nchannels, x2 - x1, y2 - y1))
    for i in range(batch_size):
        for n in range(nchannels):
            cropped_imgs[i, n] = imgs[i, n, x1:x2, y1:y2]
    return cropped_imgs


def pad_keypoints(keypoints, pad_h, pad_w):
    """
    Pad keypoints using padding values for width and height.
    Parameters
    ----------
    keypoints : ND-array
        keypoints of size [N x 2]
    pad_h : int
        height padding
    pad_w : int
        width padding
    Returns
    -------
    keypoints : ND-array
        padded keypoints of size [N x 2]
    """
    keypoints[:, 0] += pad_h
    keypoints[:, 1] += pad_w
    return keypoints


def pad_img_to_square(img, bbox=None):
    """
    Pad image to square.
    Parameters
    ----------
    im : ND-array
        image of size [c x h x w]
    bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2 used for cropping image
    Returns
    -------
    im : ND-array
        padded image of size [c x h x w]
    (pad_w, pad_h) : tuple of int
        padding values for width and height
    """
    if bbox is not None:  # Check if bbox is square
        x1, x2, y1, y2 = bbox
        dx, dy = x2 - x1, y2 - y1
    else:
        dx, dy = img.shape[-2:]

    if dx == dy:
        return img, (0, 0, 0, 0)

    largest_dim = max(dx, dy)
    if (dx < largest_dim and abs(dx - largest_dim) % 2 != 0) or (
        dy < largest_dim and abs(dy - largest_dim) % 2 != 0
    ):
        largest_dim += 1

    if dx < largest_dim:
        pad_x = abs(dx - largest_dim)
        pad_x_left = pad_x // 2
        pad_x_right = pad_x - pad_x_left
    else:
        pad_x_left = 0
        pad_x_right = 0

    if dy < largest_dim:
        pad_y = abs(dy - largest_dim)
        pad_y_top = pad_y // 2
        pad_y_bottom = pad_y - pad_y_top
    else:
        pad_y_top = 0
        pad_y_bottom = 0

    if img.ndim > 3:
        pads = (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, 0, 0, 0, 0)
    elif img.ndim == 3:
        pads = (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, 0, 0)
    else:
        pads = (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right)

    img = F.pad(
        img,
        pads,
        mode="constant",
        value=0,
    )

    return img, (pad_y_top, pad_y_bottom, pad_x_left, pad_x_right)


def resize_keypoints(keypoints, desired_shape, original_shape):
    """
    Resize keypoints to desired shape.
    Parameters
    ----------
    keypoints: ND-array
        keypoints of size [batch x N x 2]
    desired_shape: tuple of size (2,)
        desired shape of image
    original_shape: tuple of size (2,)
        original shape of image
    Returns
    -------
    keypoints: ND-array
        keypoints of size [batch x N x 2]
    """
    x_scale = desired_shape[1] / original_shape[1]  # scale factor for x coordinates
    y_scale = desired_shape[0] / original_shape[0]  # scale factor for y coordinates
    xlabels, ylabels = keypoints[:, 0], keypoints[:, 1]
    xlabels = xlabels * x_scale
    ylabels = ylabels * y_scale
    # Stack the x and y coordinates together using torch.stack
    keypoints = torch.stack([xlabels, ylabels], dim=1)
    return keypoints


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
    im = F.interpolate(im, size=(h, w), mode="bilinear", align_corners=True).squeeze(
        dim=0
    )
    return im


def get_crop_resize_params(img, x_dims, y_dims, xy=(256, 256)):
    """
    Get cropped and resized image dimensions
    Input:-
        img: image
        x_dims: min,max x pos
        y_dims: min,max y pos
        xy: final (desired) image size
    Output:-
        x1: (int) x dim start pos
        x2: (int) x dim stop pos
        y1: (int) y dim start pos
        y2: (int) y dim stop pos
        resize: (bool) whether to resize image
    """
    x1 = int(x_dims[0])
    x2 = int(x_dims[1])
    y1 = int(y_dims[0])
    y2 = int(y_dims[1])

    resize = False
    if abs(y2 - y1) > xy[0]:  # if cropped image larger than desired size
        # crop image then resize image and landmarks
        resize = True
    else:  # if cropped image smaller than desired size then add padding accounting for labels in view
        y_pad = abs(abs(y2 - y1) - xy[0])
        if y_pad % 2 == 0:
            y_pad = y_pad // 2
            y1, y2 = y1 - y_pad, y2 + y_pad
        else:  # odd number division so add 1
            y_pad = y_pad // 2
            y1, y2 = y1 - y_pad, y2 + y_pad + 1

    if abs(x2 - x1) > xy[1]:  # if cropped image larger than desired size
        resize = True
    else:
        x_pad = abs(abs(x2 - x1) - xy[1])
        if x_pad % 2 == 0:
            x_pad = x_pad // 2
            x1, x2 = x1 - x_pad, x2 + x_pad
        else:
            x_pad = x_pad // 2
            x1, x2 = x1 - x_pad, x2 + x_pad + 1

    if y2 > img.shape[1]:
        y1 -= y2 - img.shape[1]
    if x2 > img.shape[0]:
        x1 -= x2 - img.shape[0]

    y2, x2 = min(y2, img.shape[1]), min(x2, img.shape[0])
    y1, x1 = max(0, y1), max(0, x1)
    y2, x2 = max(y2, xy[0]), max(x2, xy[1])

    return x1, x2, y1, y2, resize


def crop_image(im, bbox=None):
    """
    Crop image to bounding box.
    Parameters
    ----------
    im : ND-array
        image of size [(Lz) x Ly x Lx]
    bbox : tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2
    Returns
    -------
    im : ND-array
        cropped image of size [1 x Ly x Lx]
    """
    if bbox is None:
        return im
    y1, y2, x1, x2 = bbox
    if im.ndim == 2:
        im = im[y1:y2, x1:x2]
    elif im.ndim == 3:
        im = im[:, y1:y2, x1:x2]
    elif im.ndim == 4:
        im = im[:, :, y1:y2, x1:x2]
    else:
        raise ValueError("Cannot handle image with ndim=" + str(im.ndim))
    return im


def adjust_keypoints(xlabels, ylabels, crop_xy, padding, current_size, desired_size):
    """
    Adjust raw keypoints (x,y coordinates) obtained from model to plot on original image
    Parameters
    -------------
    xlabels: ND-array
        x coordinates of keypoints
    ylabels: ND-array
        y coordinates of keypoints
    crop_xy: tuple of size (2,)
        initial coordinates of bounding box (x1,y1) for cropping
    padding: tuple of size (4,)
        padding values for bounding box (x1,x2,y1,y2)
    Returns
    --------------
    xlabels: ND-array
        x coordinates of keypoints
    ylabels: ND-array
        y coordinates of keypoints
    """
    # Rescale keypoints to original image size
    xlabels, ylabels = rescale_keypoints(xlabels, ylabels, current_size, desired_size)
    xlabels, ylabels = adjust_keypoints_for_padding(xlabels, ylabels, padding)
    # Adjust for cropping
    x1, y1 = crop_xy[0], crop_xy[1]
    xlabels += x1
    ylabels += y1
    return xlabels, ylabels


def rescale_keypoints(xlabels, ylabels, current_size, desired_size):
    """
    Rescale keypoints to original image size
    Parameters
    -------------
    xlabels: ND-array
        x coordinates of keypoints
    ylabels: ND-array
        y coordinates of keypoints
    current_size: tuple of size (2,)
        current size of image (h,w)
    desired_size: tuple of size (2,)
        desired size of image (h,w)
    Returns
    --------------
    xlabels: ND-array
        x coordinates of keypoints
    ylabels: ND-array
        y coordinates of keypoints
    """
    xlabels *= desired_size[1] / current_size[1]  # x_scale
    ylabels *= desired_size[0] / current_size[0]  # y_scale
    return xlabels, ylabels


def adjust_keypoints_for_padding(xlabels, ylabels, pads):
    """
    Adjust keypoints for padding. Adds padding to the top and left of the image only.
    Parameters
    -------------
    xlabels: ND-array
        x coordinates of keypoints
    ylabels: ND-array
        y coordinates of keypoints
    pads: tuple of size (4,)
        padding values for bounding box (y1,y2,x1,x2)
    Returns
    --------------
    xlabels: ND-array
        x coordinates of keypoints after padding
    ylabels: ND-array
        y coordinates of keypoints after padding
    """
    pad_y_top, pad_y_bottom, pad_x_left, pad_x_right = pads
    xlabels -= pad_y_top
    ylabels -= pad_x_left
    return xlabels, ylabels


def adjust_bbox(prev_bbox, img_yx, div=16, extra=1):
    """
    Takes a bounding box as an input and the original image size. Adjusts bounding box to be square
    instead of a rectangle. Uses longest dimension of prev_bbox for final image size that cannot
    exceed img_yx
    Parameters
    -------------
    prev_bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2
    img_yx: tuple of size (2,)
        image size for y and x dimensions
    Returns
    --------------
    bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2
    """
    x1, x2, y1, y2 = np.round(prev_bbox)
    xdim, ydim = (x2 - x1), (y2 - y1)

    # Pad bbox dimensions to be divisible by div
    Lpad = int(div * np.ceil(xdim / div) - xdim)
    xpad1 = extra * div // 2 + Lpad // 2
    xpad2 = extra * div // 2 + Lpad - Lpad // 2
    Lpad = int(div * np.ceil(ydim / div) - ydim)
    ypad1 = extra * div // 2 + Lpad // 2
    ypad2 = extra * div // 2 + Lpad - Lpad // 2

    x1, x2, y1, y2 = x1 - xpad1, x2 + xpad2, y1 - ypad1, y2 + ypad2
    xdim = min(x2 - x1, img_yx[1])
    ydim = min(y2 - y1, img_yx[0])

    # Choose largest dimension for image size
    if xdim > ydim:
        # Adjust ydim
        ypad = xdim - ydim
        if ypad % 2 != 0:
            ypad += 1
        y1 = max(0, y1 - ypad // 2)
        y2 = min(y2 + ypad // 2, img_yx[0])
    else:
        # Adjust xdim
        xpad = ydim - xdim
        if xpad % 2 != 0:
            xpad += 1
        x1 = max(0, x1 - xpad // 2)
        x2 = min(x2 + xpad // 2, img_yx[1])
    adjusted_bbox = (x1, x2, y1, y2)
    return adjusted_bbox


def augment_data(
    image,
    keypoints,
    scale=False,
    scale_range=0.5,
    rotation=False,
    rotation_range=10,
    flip=True,
    contrast_adjust=True,
):
    """
    Augments data by randomly scaling, rotating, flipping, and adjusting contrast
    Parameters
    ----------
    image: ND-array
        image of size nchan x Ly x Lx
    keypoints: ND-array
        keypoints of size nkeypoints x 2
    scale: bool
        whether to scale the image
    scale_range: float
        range of scaling factor
    rotation: bool
        whether to rotate the image
    rotation_range: float
        range of rotation angle
    flip: bool
        whether to flip the image horizontally
    contrast_adjust: bool
        whether to adjust contrast of image
    Returns
    -------
    image: ND-array
        image of size nchan x Ly x Lx
    keypoints: ND-array
        keypoints of size nkeypoints x 2
    """
    if scale:
        scale_range = max(0, min(2, float(scale_range)))
        scale_factor = (np.random.rand() - 0.5) * scale_range + 1
        image = image.squeeze() * scale_factor
        keypoints = keypoints * scale_factor
    if rotation:
        theta = np.random.rand() * rotation_range - rotation_range / 2
        print("rotating by {}".format(theta))
        image = ndimage.rotate(image, theta, axes=(-2, -1), reshape=True)
        keypoints = rotate_points(keypoints, theta)  # TODO: Add rotation function
    if flip and np.random.rand() > 0.5:
        keypoints[:, 0] = 256 - keypoints[:, 0]
        image = ndimage.rotate(image, 180, axes=(-1, 0), reshape=True)
    if contrast_adjust and np.random.rand() > 0.5:
        image = pose_utils.randomly_adjust_contrast(image)

    return image, keypoints
