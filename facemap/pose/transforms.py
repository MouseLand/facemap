"""
Facemap functions for: 
- bounding box: (suggested ROI) for UNet input images
- image preprocessing 
- image augmentation
"""
import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.nn import functional as F

from . import pose_helper_functions


def preprocess_img(im):
    """
    Preproccesing of image involves: conversion to float32 in range0-1, normalize99, and padding image size to be
    compatible with UNet model input i.e. divisible by 16
     Parameters
    -------------
    im: ND-array
        image of size [(Lz) x Ly x Lx]
     Returns
    --------------
    im: ND-array
        preprocessed image of size [1 x Ly x Lx] if input dimensions==2, else [Lz x Ly x Lx]
    """
    if im.ndim == 2:
        im = im[np.newaxis, ...]
    # Adjust image contrast
    im = pose_helper_functions.normalize99(im)
    return im


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


def get_crop_resize_params(img, x_dims, y_dims, xy=(256, 256)):
    """
    Get cropped and resized image dimensions
    Input:-
        img: image
        x_dims: min,max x pos
        y_dims: min,max y pos
        xy: final (desired) image size
    Output:-
        Xstart: (int) x dim start pos
        Xstop: (int) x dim stop pos
        Ystart: (int) y dim start pos
        Ystop: (int) y dim stop pos
        resize: (bool) whether to resize image
    """
    Xstart = int(x_dims[0])
    Xstop = int(x_dims[1])
    Ystart = int(y_dims[0])
    Ystop = int(y_dims[1])

    resize = False
    if abs(Ystop - Ystart) > xy[0]:  # if cropped image larger than desired size
        # crop image then resize image and landmarks
        resize = True
    else:  # if cropped image smaller than desired size then add padding accounting for labels in view
        y_pad = abs(abs(Ystop - Ystart) - xy[0])
        if y_pad % 2 == 0:
            y_pad = y_pad // 2
            Ystart, Ystop = Ystart - y_pad, Ystop + y_pad
        else:  # odd number division so add 1
            y_pad = y_pad // 2
            Ystart, Ystop = Ystart - y_pad, Ystop + y_pad + 1

    if abs(Xstop - Xstart) > xy[1]:  # if cropped image larger than desired size
        resize = True
    else:
        x_pad = abs(abs(Xstop - Xstart) - xy[1])
        if x_pad % 2 == 0:
            x_pad = x_pad // 2
            Xstart, Xstop = Xstart - x_pad, Xstop + x_pad
        else:
            x_pad = x_pad // 2
            Xstart, Xstop = Xstart - x_pad, Xstop + x_pad + 1

    if Ystop > img.shape[1]:
        Ystart -= Ystop - img.shape[1]
    if Xstop > img.shape[0]:
        Xstart -= Xstop - img.shape[0]

    Ystop, Xstop = min(Ystop, img.shape[1]), min(Xstop, img.shape[0])
    Ystart, Xstart = max(0, Ystart), max(0, Xstart)
    Ystop, Xstop = max(Ystop, xy[0]), max(Xstop, xy[1])

    return Xstart, Xstop, Ystart, Ystop, resize


def crop_resize(img, Xstart, Xstop, Ystart, Ystop, resize, xy=[256, 256]):
    """
    Crop and resize image using dimensions provided
    Input:-
        img: (2D array) image
        Xstart: (int) x dim start pos
        Xstop: (int) x dim stop pos
        Ystart: (int) y dim start pos
        Ystop: (int) y dim stop pos
        resize: (bool) whether to resize image
    Output:-
        im_cropped: (2D array) cropped image
    """
    # Crop image and landmarks
    im_cropped = img[:, :, Ystart:Ystop, Xstart:Xstop]
    # Resize image
    if resize:
        im_cropped = F.interpolate(im_cropped, size=(256, 256), mode="bilinear")
    return im_cropped


def labels_crop_resize(Xlabel, Ylabel, Xstart, Ystart, current_size, desired_size):
    """
    Adjust x,y labels on a 2D image to perform a resize operation
    Parameters
    -------------
    Xlabel: ND-array
    Ylabel: ND-array
    current_size: tuple or array of size(2,)
    desired_size: tuple or array of size(2,)
    Returns
    --------------
    Xlabel: ND-array
            adjusted x values on new/desired_size of image
    Ylabel: ND-array
            adjusted y values on new/desired_size of image
    """
    # Xlabel, Ylabel = Xlabel.astype(float), Ylabel.astype(float)
    Xlabel *= desired_size[1] / current_size[1]  # x_scale
    Ylabel *= desired_size[0] / current_size[0]  # y_scale
    Xlabel = Xlabel + Xstart
    Ylabel = Ylabel + Ystart
    return Xlabel, Ylabel


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


#  Following Function adopted from cellpose:
#  https://github.com/MouseLand/cellpose/blob/35c16c94e285a4ec2fa17f148f06bbd414deb5b8/cellpose/transforms.py#L187
def normalize99(img):
    """
    Normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile
     Parameters
    -------------
    img: ND-array
        image of size [Ly x Lx]
    Returns
    --------------
    X: ND-array
        normalized image of size [Ly x Lx]
    """
    X = img.copy()
    x01 = np.percentile(X, 1)
    x99 = np.percentile(X, 99)
    X = (X - x01) / (x99 - x01)
    return X


def random_rotate_and_resize(
    X,
    landmarks,
    scale_range=0.5,
    xy=(256, 256),
    do_flip=True,
    rotation=30,
    contrast_adjustment=True,
    gamma_aug=False,
    gamma_range=0.5,
    motion_blur=False,
    gaussian_blur=False,
):

    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim > 2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)

    lbl = np.zeros((nimg, landmarks[0].shape[0], 2), np.float32)
    scale = np.zeros(nimg, np.float32)
    dg = gamma_range / 2

    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]
        imgi[n], lbl[n] = X[n].copy(), landmarks[n].copy()

        if contrast_adjustment:
            imgi[n] = pose_helper_functions.randomly_adjust_contrast(imgi[n])

        # generate random augmentation parameters
        theta = np.pi * (2 * np.random.rand() - 1) * rotation / 180
        scale[n] = (np.random.rand() - 0.5) * scale_range + 1
        flip = np.random.rand() > 0.5

        dxy = 32 * np.ones(
            2,
        )
        dxy = (
            np.random.rand(
                2,
            )
            - 0.5
        ) * dxy

        # create affine transform
        cc = np.array([Lx / 2, Ly / 2])
        cc1 = cc - np.array([Lx - xy[1], Ly - xy[0]]) / 2 + dxy
        pts1 = np.float32([cc, cc + np.array([1, 0]), cc + np.array([0, 1])])
        pts2 = np.float32(
            [
                cc1,
                cc1 + scale[n] * np.array([np.cos(theta), np.sin(theta)]),
                cc1
                + scale[n]
                * np.array([np.cos(np.pi / 2 + theta), np.sin(np.pi / 2 + theta)]),
            ]
        )
        M = cv2.getAffineTransform(pts1, pts2)

        if flip and do_flip:
            imgi[n] = np.flip(imgi[n], axis=-1)
            lbl[n, :, 0] = Lx - lbl[n, :, 0]

        # affine transform on labels
        lbl[n] = lbl[n] @ M[:, :2].T + M[:, 2]
        lbl[n, lbl[n, :, 0] > xy[0] - 1, 0] = np.nan
        lbl[n, lbl[n, :, 1] > xy[1] - 1, 1] = np.nan
        lbl[n, lbl[n] < 0] = np.nan

        # affine transform on image
        for k in range(nchan):
            I = cv2.warpAffine(imgi[n][k], M, (xy[1], xy[0]), flags=cv2.INTER_LINEAR)
            if gamma_aug:
                gamma = np.random.uniform(low=1 - dg, high=1 + dg)
                imgi[n] = np.sign(I) * (np.abs(I)) ** gamma
            else:
                imgi[n] = I

    return imgi, lbl, scale
