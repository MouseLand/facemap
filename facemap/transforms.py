"""
Facemap functions for: 
- bounding box: (suggested ROI) for UNet input images
- image preprocessing 
- image augmentation
"""
import numpy as np
import cv2
import torch
from . import UNet_helper_functions

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
    # convert to float32 in the range 0. to 1.
    if im.dtype == float:
        pass
    elif im.dtype == np.uint8:
        im = im.astype(float)/255.
    elif im.dtype == np.uint16:
        im = im.astype(float)/65535.
    else:
        print('Cannot handle im type '+str(im.dtype))
        raise TypeError
    # Normalize images
    im = normalize99(im)   
    if im.ndim < 3:
        im = im[np.newaxis,...]
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
    cropped_imgs = np.empty((batch_size, nchannels, x2-x1, y2-y1))
    for i in range(batch_size):
        for n in range(nchannels):
            cropped_imgs[i,n] = imgs[i, n, x1:x2, y1:y2]
    return cropped_imgs
    
def get_bounding_box(imgs, net, prev_bbox):
    """
    Predicts bounding box for face ROI used as input for UNet model.
    Parameters
    -------------
    imgs: ND-array
        images of size [batch_size x nchan x Ly x Lx]
    net: NA
        UNet model/network
    prev_bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2 
    Returns
    --------------
    bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2 
    lm_mean: ND-array
        avg. position of all landmarks/key points
    """
    prev_minx, prev_maxx, prev_miny, prev_maxy = prev_bbox
    net.eval()
    # pad image if not divisible by 16 
    if imgs.shape[-1]%16!=0 or imgs.shape[-2]%16!=0:  
        imgs, ysub, xsub = pad_image_ND(imgs)
    # Network prediction using padded image
    hm_pred = net(torch.tensor(imgs).to(device=net.DEVICE, dtype=torch.float32)) # convert to tensor and send to device
    # slices from padding
    slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
    slc[-3] = slice(0, hm_pred.shape[-3])
    slc[-2] = slice(ysub[0], ysub[-1]+1)
    slc[-1] = slice(xsub[0], xsub[-1]+1)
    # Slice out padding
    hm_pred = hm_pred[slc]
    # Get landmark positions
    print("sliced hm", hm_pred.shape)
    lm = UNet_helper_functions.heatmap2landmarks(hm_pred.cpu().detach().numpy())
    lm_mean = lm.mean(axis=0) # avg. position of all landmarks/key points
    # Estimate bbox positions using landmark positions b/w 5th and 95th percentile 
    min_x = np.nanmean([np.percentile(lm.T[0,:],5),prev_minx])
    max_x = np.nanmean([np.percentile(lm.T[0,:],90),prev_maxx])
    min_y = np.nanmean([np.percentile(lm.T[1,:],5),prev_miny])
    max_y = np.nanmean([np.percentile(lm.T[1,:],90),prev_maxy])
    bbox = (min_x, max_x, min_y, max_y)
    return bbox, lm_mean

def adjust_bbox(prev_bbox, img_xy):
    """
    Takes a bounding box as an input and desired image size. Adjusts bounding box to be square
    instead of a rectangle. Uses longest dimension of prev_bbox for final image size that cannot
    exceed img_xy
    Parameters
    -------------
    prev_bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2 
    img_xy: tuple of size (2,)
        image size for x and y dimensions
    Returns
    --------------
    bbox: tuple of size (4,)
        bounding box positions in order x1, x2, y1, y2 
    """
    padding = 20
    x1, x2, y1, y2 = np.round(prev_bbox)
    xdim, ydim = (x2-x1)+padding, (y2-y1)+padding
    xdim = min(xdim, img_xy[0])
    ydim = min(ydim, img_xy[1])
    if xdim > ydim:
        # Adjust ydim
        ypad = xdim-ydim 
        if ypad%2!=0:
            ypad+=1
        y1 = max(0, y1-ypad//2)
        y2 = min(y2+ypad//2, img_xy[1])
    else:
        # Adjust xdim
        xpad = ydim-xdim
        if xpad%2!=0:
            xpad+=1
        x1 = max(0, x1-xpad//2)
        x2 = min(x2+xpad//2, img_xy[0])
    bbox = (x1, x2, y1, y2)
    return bbox

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

#  Following Function adopted from cellpose:
#  https://github.com/MouseLand/cellpose/blob/35c16c94e285a4ec2fa17f148f06bbd414deb5b8/cellpose/transforms.py#L547
def pad_image_ND(img0, div=16, extra = 1):
    """ 
    Pad image for test-time so that its dimensions are a multiple of 16 (2D or 3D)
    Parameters
    -------------
    img0: ND-array
        image of size [nchan (x Lz) x Ly x Lx]
    div: int (optional, default 16)
    Returns
    --------------
    I: ND-array
        padded image
    ysub: array, int
        yrange of pixels in I corresponding to img0
    xsub: array, int
        xrange of pixels in I corresponding to img0
    """
    Lpad = int(div * np.ceil(img0.shape[-2]/div) - img0.shape[-2])
    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    Lpad = int(div * np.ceil(img0.shape[-1]/div) - img0.shape[-1])
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2

    if img0.ndim>3:
        pads = np.array([[0,0], [0,0], [xpad1,xpad2], [ypad1, ypad2]])
    else:
        pads = np.array([[0,0], [xpad1,xpad2], [ypad1, ypad2]])

    I = np.pad(img0, pads, mode='constant')

    Ly, Lx = img0.shape[-2:]
    ysub = np.arange(xpad1, xpad1+Ly)
    xsub = np.arange(ypad1, ypad1+Lx)
    return I, ysub, xsub