# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Import packages ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

print('numpy version: %s'%np.__version__)
import cv2  # opencv
import torch  # pytorch
import numbers
import math
import time
print('CUDA available: %d'%torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('torch version: %s'%torch.__version__)
import itertools
import os  # file path stuff
import random
from glob import glob  # listing files
from platform import python_version

import pandas as pd
from tqdm import tqdm  # waitbar
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from scipy.ndimage import gaussian_filter

print("python version:", python_version())
import matplotlib

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Global variables~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
DIST_THRESHOLD = 17/2
LOCREF_STDEV = 7.2801/2
STRIDE = 4
print("Global varaibles set:")
print("dist threshold:", DIST_THRESHOLD)
print("locref stdev:", LOCREF_STDEV)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helper functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def gaussian_smoothing(hm, sigma, nms_size, sigmoid=False):
    num_bodyparts = hm.shape[1]
    filters = get_2d_gaussian_kernel_map(sigma, nms_size, num_bodyparts)
    if sigmoid:
        s_fn = torch.nn.Sigmoid()
        image = s_fn(hm)
    else:
        image = hm
    cin = image.shape[1]
    features = F.conv2d(image, filters, groups=cin, padding='same')
    return features

def get_2d_gaussian_kernel_map(sigma, nms_radius, num_landmarks):
    size = nms_radius * 2 + 1
    k = torch.range(-size // 2 + 1, size // 2 + 1)
    k = k ** 2
    sm = torch.nn.Softmax()
    k = sm(-k / (2 * (sigma ** 2)))
    kernel = torch.einsum('i,j->ij', k, k)
    kernel = torch.unsqueeze(kernel, dim=0)
    kernel = torch.unsqueeze(kernel, dim=0)
    kernel_sc = kernel.repeat([num_landmarks, 1, 1, 1])
    return kernel_sc

def argmax_pose_predict_batch(scmap_batch, offmat_batch, stride):
    """Combine scoremat and offsets to the final pose."""
    pose_batch = []
    for b in range(scmap_batch.shape[0]):
        scmap, offmat = scmap_batch[b].T, offmat_batch[b].T
        ny, nx, num_joints = offmat.shape
        offmat = offmat.reshape(ny,nx,num_joints//2,-1)
        offmat *= LOCREF_STDEV
        num_joints = scmap.shape[2]
        pose = []
        for joint_idx in range(num_joints):
            maxloc = np.unravel_index(
                np.argmax(scmap[:, :, joint_idx]), scmap[:, :, joint_idx].shape
            )
            offset = np.array(offmat[maxloc][joint_idx])[::-1]
            pos_f8 = np.array(maxloc).astype("float") * stride + 0.5 * stride + offset
            pose.append(np.hstack((pos_f8, [scmap[maxloc][joint_idx]])))
        pose_batch.append(pose)
    return np.array(pose_batch)

def clahe_adjust_contrast(in_img):
    """
    in_img : LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]
    """
    in_img = np.array(in_img)
    clahe_grid_size = 20
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(clahe_grid_size, clahe_grid_size))
    simg = np.zeros(in_img.shape)
    if in_img.shape[1] == 1:
        for ndx in range(in_img.shape[0]): # for each image ndx
            simg[ndx,0,:,:] = clahe.apply(in_img[ndx, 0,:,:].astype('uint8')).astype('float')
    else:
        for ndx in range(in_img.shape[0]):
            lab = cv2.cvtColor(in_img[ndx,...], cv2.COLOR_RGB2LAB)
            lab_planes = cv2.split(lab)
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            simg[ndx,...] = rgb
    return simg

def normalize_mean(in_img):
    zz = in_img.astype('float')
    # subtract mean for each img.
    mm = zz.mean(axis=(2,3))
    xx = zz - mm[:, :, np.newaxis, np.newaxis]
    return xx

def normalize99(X):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    x01 = torch.quantile(X, .01)
    x99 = torch.quantile(X, .99)
    X = (X - x01) / (x99 - x01)
    return X

n_factor =  2**4 // (2 ** 2)
sigma  = 3 * 4 / n_factor
Lx = 64

def get_predicted_landmarks(net, im_input, batchsize=1, smooth=True):
    
    xmesh, ymesh = np.meshgrid(torch.arange(net.image_shape[0]/n_factor), torch.arange(net.image_shape[1]/n_factor))
    ymesh = torch.from_numpy(ymesh).to(device)
    xmesh = torch.from_numpy(xmesh).to(device)

    # Predict
    with torch.no_grad():
        if im_input.ndim == 3:
            im_input = im_input[np.newaxis, ...]
        hm_pred, locx_pred, locy_pred = net(im_input)

        hm_pred = hm_pred.squeeze()
        locx_pred = locx_pred.squeeze()
        locy_pred = locy_pred.squeeze()

        if smooth:
            hm_smo = gaussian_filter(hm_pred.cpu().numpy(), [0, 1, 1])
            hm_smo = hm_smo.reshape(hm_smo.shape[0], hm_smo.shape[1], Lx*Lx)
            imax = torch.argmax(hm_smo, -1)
            likelihood = torch.diag(hm_smo[:,:,imax])
        else:
            hm_pred = hm_pred.reshape(hm_pred.shape[0], Lx*Lx)
            imax = torch.argmax(hm_pred, 1)
            likelihood = torch.diag(hm_pred[:,imax])

        # this part computes the position error on the training set
        locx_pred = locx_pred.reshape(locx_pred.shape[0], Lx*Lx)
        locy_pred = locy_pred.reshape(locy_pred.shape[0], Lx*Lx)

        nn = hm_pred.shape[0]
        x_pred = ymesh.flatten()[imax] - (2*sigma) * locx_pred[torch.arange(nn), imax]
        y_pred = xmesh.flatten()[imax] - (2*sigma) * locy_pred[torch.arange(nn), imax]

    return y_pred*n_factor, x_pred*n_factor, likelihood

def analyze_frames(frames_dir, bodyparts, scorer, net, img_xy):
    """
    Input:-
    - frames_dir: path containing .png files 
    - bodyparts: landmark names
    - scorer: Name of scorer
    - net: UNet model for predictions
    - img_xy: size of images for resizing before predictions
    Output:-
    - dataFrame: predictions from network stored as multi-level dataframe
    """
    frames = glob(frames_dir+"/*.png")
    # Create an empty dataframe
    for index, bodypart in enumerate(bodyparts):
        columnindex = pd.MultiIndex.from_product(
            [[scorer], [bodypart], ["x", "y"]], 
            names=["scorer", "bodyparts", "coords"])
        frame = pd.DataFrame(
            np.nan,
            columns=columnindex,
            index=[os.path.join(fn.split("/")[-2], fn.split("/")[-1]) for fn in frames])
        if index == 0:
            dataFrame = frame
        else:
            dataFrame = pd.concat([dataFrame, frame], axis=1)
    # Add predicted values to dataframe
    net.eval()
    for ind, img_file in enumerate(frames):
        im = cv2.imread(img_file, 0)
        im = im.astype('uint8')
        im = im[np.newaxis,np.newaxis,:,:]
        # Adjust image contrast
        im = clahe_adjust_contrast(im)
        im = normalize_mean(im)
        for i in range(im.shape[0]):
            im[i,0] = normalize99(im[i,0])

        hm_pred, locref_pred = net(torch.Tensor(im).to(device=net.DEVICE, dtype=torch.float32))
        hm_pred = gaussian_smoothing(hm_pred.cpu(), sigma=3, nms_size=9, sigmoid=True)
        pose = argmax_pose_predict_batch(hm_pred.cpu().detach().numpy(), locref_pred.cpu().detach().numpy(),
                                          stride=STRIDE)
        landmarks = pose[:,:,:2].ravel()
        dataFrame.iloc[ind] = landmarks
    return dataFrame

def heatmap2landmarks(hms):
    idx = np.argmax(hms.reshape(hms.shape[:-2]+(hms.shape[-2]*hms.shape[-1],)),axis=-1)
    locs = np.zeros(hms.shape[:-2]+(2,))
    locs[...,1],locs[...,0] = np.unravel_index(idx,hms.shape[-2:])
    return locs.astype(int)

def heatmap2image(hm,cmap='jet',colors=None):
    """
    heatmap2image(hm,cmap='jet',colors=None)
    Creates and returns an image visualization from landmark heatmaps. Each 
    landmark is colored according to the input cmap/colors. 
    Inputs:
    hm: nlandmarks x height x width ndarray, dtype=float in the range 0 to 1. 
    hm[p,i,j] is a score indicating how likely it is that the pth landmark 
    is at pixel location (i,j).
    cmap: string.
    Name of colormap for defining colors of landmark points. Used only if colors
    is None. 
    Default: 'jet'
    colors: list of length nlandmarks. 
    colors[p] is an ndarray of size (4,) indicating the color to use for the 
    pth landmark. colors is the output of matplotlib's colormap functions. 
    Default: None
    Output:
    im: height x width x 3 ndarray
    Image representation of the input heatmap landmarks.
    """
    hm = np.maximum(0.,np.minimum(1.,hm))
    im = np.zeros((hm.shape[1],hm.shape[2],3))
    if colors is None:
        if isinstance(cmap,str):
            cmap = matplotlib.cm.get_cmap(cmap)
        colornorm = matplotlib.colors.Normalize(vmin=0,vmax=hm.shape[0])
        colors = cmap(colornorm(np.arange(hm.shape[0])))
    for i in range(hm.shape[0]):
        color = colors[i]
        for c in range(3):
            im[...,c] = im[...,c]+(color[c]*.7+.3)*hm[i,...]
    im = np.minimum(1.,im)
    return im

def add_motion_blur(img, kernel_size=None, vertical=True, horizontal=True):
    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size
    
    if vertical:
        # Apply the vertical kernel.
        img = cv2.filter2D(img, -1, kernel_v)

    if horizontal:
        # Apply the horizontal kernel.
        img = cv2.filter2D(img, -1, kernel_h)
    
    return img

def randomly_adjust_contrast(img):
    """
    Randomly adjusts contrast of image
    img: ND-array of size nchan x Ly x Lx
    Assumes image values in range 0 to 1
    """
    brange = [-0.2,0.2]
    bdiff = brange[1] - brange[0]
    crange = [0.7,1.3]
    cdiff = crange[1] - crange[0]
    imax = 1
    if (bdiff<0.01) and (cdiff<0.01):
        return img
    bfactor = np.random.rand() * bdiff + brange[0]
    cfactor = np.random.rand() * cdiff + crange[0]
    mm = img.mean()
    jj = img + bfactor * imax
    jj = np.minimum(imax, (jj - mm) * cfactor + mm)
    jj = jj.clip(0, imax)
    img = normalize99(jj)
    return img

# Cellpose augmentation method (https://github.com/MouseLand/cellpose/blob/35c16c94e285a4ec2fa17f148f06bbd414deb5b8/cellpose/transforms.py#L590)
def random_rotate_and_resize(X, Y, scale_range=1., xy=(256,256),do_flip=True, rotation=10,
                            contrast_adjustment=True, gamma_aug=True, gamma_range=0.5,
                            motion_blur=True, gaussian_blur=True):
    """ augmentation by random rotation and resizing
        X and Y are lists or arrays of length nimg, with dims channels x Ly x Lx (channels optional)
        Parameters
        ----------
        X: LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]
        Y: LIST of ND-arrays, float (optional, default None)
            list of image labels of size [nlabels x Ly x Lx] or [Ly x Lx]. The 1st channel
            of Y is always nearest-neighbor interpolated (assumed to be masks or 0-1 representation).
            If Y.shape[0]==3 and not unet, then the labels are assumed to be [cell probability, Y flow, X flow]. 
            If unet, second channel is dist_to_bound.
        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by
            (1-scale_range/2) + scale_range * np.random.rand()
        xy: tuple, int (optional, default (224,224))
            size of transformed images to return
        do_flip: bool (optional, default True)
            whether or not to flip images horizontally
        rescale: array, float (optional, default None)
            how much to resize images by before performing augmentations
        unet: bool (optional, default False)
        Returns
        -------
        imgi: ND-array, float
            transformed images in array [nimg x nchan x xy[0] x xy[1]]
        lbl: ND-array, float
            transformed labels in array [nimg x nchan x xy[0] x xy[1]]
        scale: array, float
            amount each image was resized by
    """
    scale_range = max(0, min(2, float(scale_range)))
    nimg = len(X)
    if X[0].ndim>2:
        nchan = X[0].shape[0]
    else:
        nchan = 1
    imgi  = np.zeros((nimg, nchan, xy[0], xy[1]), np.float32)

    lbl = []
    if Y[0].ndim>2:
        nt = Y[0].shape[0]
    else:
        nt = 1
    lbl = np.zeros((nimg, nt, xy[0], xy[1]), np.float32)
    scale = np.zeros(nimg, np.float32)
    dg = gamma_range/2 
    
    for n in range(nimg):
        Ly, Lx = X[n].shape[-2:]
        imgi[n], lbl[n] = X[n].copy(), Y[n].copy()
        
        # generate random augmentation parameters
        flip = np.random.rand()>.5
        theta = random.random()*rotation # random rotation in degrees (float)
        scale[n] = (1-scale_range/2) + scale_range * np.random.rand()
        
        if contrast_adjustment:
            imgi[n] = randomly_adjust_contrast(imgi[n])
        if motion_blur and np.random.rand()>.5:
            blur_pixels = np.random.randint(1,5)
            imgi[n] = add_motion_blur(imgi[n], kernel_size=blur_pixels)
        if gaussian_blur and np.random.rand()>.5:
            kernel = random.randrange(1,10,2)#np.random.randint(1,9)
            imgi[n] = cv2.GaussianBlur(imgi[n], (kernel, kernel), cv2.BORDER_CONSTANT )
        # create affine transform
        c = (xy[0] * 0.5 - 0.5, xy[1] * 0.5 - 0.5)  # Center of image
        M = cv2.getRotationMatrix2D(c, theta, scale[n])

        if flip and do_flip:
            imgi[n] = np.flip(imgi[n], axis=-1)
            lbl[n] =  np.flip(lbl[n], axis=-1)
            
        for k in range(nchan):
            I = cv2.warpAffine(imgi[n][k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR)
            if gamma_aug:
                gamma = np.random.uniform(low=1-dg,high=1+dg) 
                imgi[n] = np.sign(I) * (np.abs(I)) ** gamma
            else:
                imgi[n] = I
        
        for k in range(nt):
            lbl[n,k] = cv2.warpAffine(lbl[n][k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR)
    
    return imgi, lbl, scale

