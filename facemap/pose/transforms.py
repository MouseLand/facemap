"""
Facemap functions for: 
- bounding box: (suggested ROI) for UNet input images
- image preprocessing 
- image augmentation
"""
import cv2
import numpy as np
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
    im = im.astype('uint8') # APT method only
    if im.ndim == 2:
        im = im[np.newaxis,np.newaxis,...]
    # Adjust image contrast
    im = UNet_helper_functions.clahe_adjust_contrast(im)
    im = UNet_helper_functions.normalize_mean(im)
    for i in range(im.shape[0]):
        im[i,0] = normalize99(im[i,0]) 
    im = np.squeeze(im, axis=0)
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
    hm_pred, locref_pred = net(torch.tensor(imgs).to(device=net.DEVICE, dtype=torch.float32)) # convert to tensor and send to device
    # Slice out padding
    slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
    slc[-3] = slice(0, hm_pred.shape[-3])
    slc[-2] = slice(ysub[0], ysub[-1]+1)
    slc[-1] = slice(xsub[0], xsub[-1]+1)
    hm_pred = hm_pred[slc]
    slc[-3] = slice(0, locref_pred.shape[-3])
    locref_pred = locref_pred[slc]
    # Get landmark positions
    pose = UNet_helper_functions.argmax_pose_predict_batch(hm_pred.cpu().detach().numpy(), 
                                                            locref_pred.cpu().detach().numpy(),
                                                            UNet_helper_functions.STRIDE)
    lm = pose[:,:,:2].squeeze()
    # avg. position of all landmarks/key points
    lm_mean = lm.mean(axis=0) 
    # Estimate bbox positions using landmark positions b/w 5th and 95th percentile 
    pad = 10
    min_x = np.nanmean([np.percentile(lm.T[0,:],5)-pad, prev_minx])
    max_x = np.nanmean([np.percentile(lm.T[0,:],90)+pad, prev_maxx])
    min_y = np.nanmean([np.percentile(lm.T[1,:],5)-pad, prev_miny])
    max_y = np.nanmean([np.percentile(lm.T[1,:],90)+pad, prev_maxy])
    bbox = (min_x, max_x, min_y, max_y)
    return bbox, lm_mean

def get_crop_resize_params(img, x_dims, y_dims, xy=(256,256)):
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
    xdiff, ydiff = x_dims[1] - x_dims[0], y_dims[1] - y_dims[0]
    
    Xstart = int(x_dims[0])
    Xstop = int(x_dims[1])
    Ystart = int(y_dims[0])
    Ystop = int(y_dims[1])
    
    resize = False
    if abs(Ystop-Ystart) > xy[0]:   # if cropped image larger than desired size
        # crop image then resize image and landmarks 
        resize = True
    else:    # if cropped image smaller than desired size then add padding accounting for labels in view
        y_pad = abs(abs(Ystop-Ystart) - xy[0]) 
        if y_pad % 2 == 0:
            y_pad = y_pad//2
            Ystart, Ystop = Ystart-y_pad, Ystop+y_pad
        else:  # odd number division so add 1
            y_pad = y_pad//2
            Ystart, Ystop = Ystart-y_pad, Ystop+y_pad+1
        
    if abs(Xstop-Xstart) > xy[1]:  # if cropped image larger than desired size
        resize = True
    else:
        x_pad = abs(abs(Xstop-Xstart) - xy[1]) 
        if x_pad % 2 == 0 :
            x_pad = x_pad//2 
            Xstart, Xstop = Xstart-x_pad, Xstop+x_pad
        else:
            x_pad = x_pad//2 
            Xstart, Xstop = Xstart-x_pad, Xstop+x_pad+1
    
    if Ystop > img.shape[0]:
        Ystart -= (Ystop - img.shape[0])
    if Xstop > img.shape[1]:
        Xstart -= (Xstop - img.shape[1])
    
    Ystop, Xstop = min(Ystop, img.shape[0]), min(Xstop, img.shape[1])
    Ystart, Xstart = max(0, Ystart), max(0, Xstart) 
    Ystop, Xstop = max(Ystop, xy[0]), max(Xstop, xy[1])    

    return Xstart, Xstop, Ystart, Ystop, resize

def crop_resize(img, Xstart, Xstop, Ystart, Ystop, resize, xy=(256,256)):
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
    im_cropped = img[Ystart:Ystop,Xstart:Xstop]
    # Resize image 
    if resize:
        im_cropped = cv2.resize(im_cropped, xy)
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
    Xlabel, Ylabel = Xlabel.astype(float), Ylabel.astype(float)
    Xlabel *= (desired_size[1]/current_size[1])  # x_scale
    Ylabel *= (desired_size[0]/current_size[0])  # y_scale
    Xlabel = Xlabel+Xstart
    Ylabel = Ylabel+Ystart
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
    xdim, ydim = (x2-x1), (y2-y1)

    # Pad bbox dimensions to be divisible by div
    Lpad = int(div * np.ceil(xdim/div) - xdim)
    xpad1 = extra*div//2 + Lpad//2
    xpad2 = extra*div//2 + Lpad - Lpad//2
    Lpad = int(div * np.ceil(ydim/div) - ydim)
    ypad1 = extra*div//2 + Lpad//2
    ypad2 = extra*div//2+Lpad - Lpad//2

    x1, x2, y1, y2  = x1-xpad1, x2+xpad2, y1-ypad1, y2+ypad2
    xdim = min(x2-x1, img_yx[1])
    ydim = min(y2-y1, img_yx[0])

    # Choose largest dimension for image size
    if xdim > ydim:
        # Adjust ydim
        ypad = xdim-ydim 
        if ypad%2!=0:
            ypad+=1
        y1 = max(0, y1-ypad//2)
        y2 = min(y2+ypad//2, img_yx[0])
    else:
        # Adjust xdim
        xpad = ydim-xdim
        if xpad%2!=0:
            xpad+=1
        x1 = max(0, x1-xpad//2)
        x2 = min(x2+xpad//2, img_yx[1])
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
