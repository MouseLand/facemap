# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Import packages ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

print('numpy version: %s'%np.__version__)
import cv2  # opencv
import torch  # pytorch

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

FILTERS = get_2d_gaussian_kernel_map(sigma=2, nms_radius=6, num_landmarks=15)

def gaussian_smoothing(hm, sigmoid=False, device=None):
    if sigmoid:
        s_fn = torch.nn.Sigmoid()
        image = s_fn(hm)
    else:
        image = hm
    cin = image.shape[1]
    features = F.conv2d(image, FILTERS.to(device=device), groups=cin, padding='same')
    return features

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
            pos_f8 = np.array(maxloc) * stride + 0.5 * stride + offset
            pose.append(np.hstack((pos_f8, [scmap[maxloc][joint_idx]])))
        pose_batch.append(pose)
    return np.array(pose_batch)

clahe_grid_size = 20
CLAHE = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(clahe_grid_size, clahe_grid_size)) 

def clahe_adjust_contrast(in_img):
    """
    in_img : ND-arrays, float
            image arrays of size [nchan x Ly x Lx] or [Ly x Lx] where nchan is 1
    """
    for ndx in range(in_img.shape[0]): # for each image index
        in_img[ndx,0,:,:] = CLAHE.apply(in_img[ndx, 0,:,:])
    return in_img

def normalize_mean(in_img):
    zz = in_img#.astype('float')
    # subtract mean for each img.
    mm = zz.mean(axis=(2,3))
    xx = zz - mm[:, :, np.newaxis, np.newaxis]
    return xx

def normalize99(img):
    """ normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile """
    x01 = torch.quantile(img, .01)
    x99 = torch.quantile(img, .99)
    img = (img - x01) / (x99 - x01)
    return img

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dataset loader ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class COCODataset(torch.utils.data.Dataset):
    """
    COCODataset
    Torch Dataset based on the COCO keypoint file format.
    """
    def __init__(self, datadir, sigma, multiview=False, img_xy=(256,256),
                 scale=0.5, flip=True, rotation=10):
        
        self.label_filter = None
        self.label_sigma = sigma #8 
        self.init_label_filter()
        self.flip = flip
        self.img_xy = img_xy
        self.rotation = rotation
        self.scale = scale
        
        if multiview:
            print("processing multiview files")
            views = glob(os.path.join(datadir,"*"))
            self.img_files = []
            for v in views:
                view = v.split("/")[-1]
                self.img_files.append(sorted(glob(os.path.join(datadir,"{}/*/*.png".format(view)))))
                annfiles = sorted(glob(os.path.join(datadir,"{}/*/*.h5".format(view))))
                self.img_files = list(itertools.chain(*self.img_files))
        else:
            print("processing single view files")
            self.datadir = datadir
            self.img_files = sorted(glob(os.path.join(self.datadir,'*/*.png')))
            # Lanmarks/key points info
            annfiles = sorted(glob(os.path.join(self.datadir,'*/*.h5')))
        
        # Landmarks dataframe concatentation
        self.landmarks = pd.DataFrame()        
        for f in annfiles:
            df = pd.read_hdf(f)
            df = df.iloc[np.argsort(df.T.columns)] # sort annotations to match img_file order
            self.landmarks = self.landmarks.append(df)
        
        self.im = []
        for file in self.img_files:
            im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            im = im.astype('uint8') # APT method only
            if im.ndim < 3:
                self.im.append(im[np.newaxis,...])
    
        # Adjust image contrast
        self.im = clahe_adjust_contrast(self.im)
        self.im = normalize_mean(self.im)
        for i in range(self.im.shape[0]):
            self.im[i,0] = normalize99(self.im[i,0]) 

        self.landmark_names = pd.unique(self.landmarks.columns.get_level_values("bodyparts"))
        self.nlandmarks = len(self.landmark_names)
        # Create heatmap target prediction
        self.landmark_heatmaps = []
        for i in range(len(self.im)):
            # locs: y_pos x x_pos for data augmentation
            locs = np.array([self.landmarks.values[i][::2], self.landmarks.values[i][1::2]]).T
            target = self.make_heatmap_target(locs, np.squeeze(self.im[i]).shape)
            self.landmark_heatmaps.append(target.detach().numpy())
            if self.im[i].ndim < 3:
                self.im[i] = self.im[i][np.newaxis,...]
        
    def __len__(self):
        return len(self.img_files)
    

    def __getitem__(self, item):
        """ 
        Input :- 
            item: scalar integer. 
        Output (dict):
            image: torch float32 tensor of size ncolors x height x width
            landmarks: nlandmarks x 2 float ndarray
            heatmaps: torch float32 tensor of size nlandmarks x height x width
            id: scalar integer, contains item
        """
        # Data augmentation
        im, lm_heatmap, _ = random_rotate_and_resize([self.im[item]],
                                                    [self.landmark_heatmaps[item]],                         
                                                    xy = self.img_xy,
                                                    scale_range=self.scale,
                                                    do_flip=self.flip,
                                                    rotation=self.rotation)
        lm = heatmap2landmarks(lm_heatmap)
        
        locs = np.squeeze(lm)
        scmap, locref_map, _ = compute_locref_maps(locs, self.img_xy,
                                               self.landmark_names, lowres=True)
        # Return tensors only
        im = np.squeeze(im,axis=0)
        lm_heatmap = np.squeeze(lm_heatmap,axis=0)
        lm = np.squeeze(lm,axis=0)
        locref_map = torch.tensor(locref_map, dtype=torch.float32) 
        scmap = torch.tensor(scmap, dtype=torch.float32) 

        features = {'image': im,
                   'landmarks': lm,
                    'heatmap' : lm_heatmap, 
                    'locref_map' : locref_map,
                    'scmap' : scmap,
                    'id': item}
        return features

    @staticmethod
    def get_landmarks(d,i=None):
        if i is None:
            locs = d['landmarks']
        else:
            locs = d['landmarks'][i]
        return locs

    @staticmethod
    def get_heatmap_image(d,i,cmap='jet',colors=None):
        if i is None:
            hm = d['heatmap']
        else:
            hm = d['heatmap'][i,...]
        hm = hm.numpy()
        im = heatmap2image(hm,cmap=cmap,colors=colors)
        return im
    
    @staticmethod
    def get_image(d,i=None):
        """
        static function, used for visualization
        COCODataset.get_image(d,i=None)
        Returns an image usable with plt.imshow()
        Inputs: 
        d: if i is None, item from a COCODataset. 
        if i is a scalar, batch of examples from a COCO Dataset returned 
        by a DataLoader. 
        i: Index of example into the batch d, or None if d is a single example
        Returns the ith image from the patch as an ndarray plottable with 
        plt.imshow()
        """
        if i is None:
            im = np.squeeze(np.transpose(d['image'].numpy(),(1,2,0)),axis=2)
        else:
            im = np.squeeze(np.transpose(d['image'][i,...].numpy(),(1,2,0)),axis=2)
        return im

    def make_heatmap_target(self,locs,imsz):
        """
        Inputs:
            locs: nlandmarks x 2 ndarray 
            imsz: image shape
        Returns:
            target: torch tensor of size nlandmarks x imsz[0] x imsz[1]
        """
        # allocate the tensor
        target = torch.zeros((locs.shape[0],imsz[0],imsz[1]),dtype=torch.float32)
        # loop through landmarks
        for i in range(locs.shape[0]):
            # location of this landmark to the nearest pixel
            if ~np.isnan(locs[i,0]) and ~np.isnan(locs[i,1]):
                # location of this landmark to the nearest pixel
                x = int(np.round(locs[i,0])) # losing sub-pixel accuracy
                y = int(np.round(locs[i,1]))
                # edges of the Gaussian filter to place, minding border of image
                x0 = np.maximum(0,x-self.label_filter_r)
                x1 = np.minimum(imsz[1]-1,x+self.label_filter_r)
                y0 = np.maximum(0,y-self.label_filter_r)
                y1 = np.minimum(imsz[0]-1,y+self.label_filter_r)
                # crop filter if it goes outside of the image
                fil_x0 = self.label_filter_r-(x-x0)
                fil_x1 = self.label_filter_d-(self.label_filter_r-(x1-x))
                fil_y0 = self.label_filter_r-(y-y0)
                fil_y1 = self.label_filter_d-(self.label_filter_r-(y1-y))
                # copy the filter to the relevant part of the heatmap image
                if len(np.arange(y0,y1+1)) != len(np.arange(fil_y0,fil_y1+1)) or len(np.arange(x0,x1+1)) != len(np.arange(fil_x0,fil_x1+1)):
                    target[i,y0:y1+1,x0:x1+1] = self.label_filter[fil_y0:fil_y0+len(np.arange(y0,y1+1)),fil_x0:fil_x0+len(np.arange(x0,x1+1))]
                else:
                    target[i,y0:y1+1,x0:x1+1] = self.label_filter[fil_y0:fil_y1+1,fil_x0:fil_x1+1]
        return target
    
    def init_label_filter(self):
        """
        init_label_filter(self)
        Helper function
        Create a Gaussian filter for the heatmap target output
        """
        # radius of the filter
        self.label_filter_r = max(int(round(3 * self.label_sigma)),1)
        # diameter of the filter
        self.label_filter_d = 2*self.label_filter_r+1

        # allocate
        self.label_filter = np.zeros([self.label_filter_d,self.label_filter_d])
        # set the middle pixel to 1. 
        self.label_filter[self.label_filter_r,self.label_filter_r] = 1.
        # blur with a Gaussian
        self.label_filter = cv2.GaussianBlur(self.label_filter, (self.label_filter_d,self.label_filter_d), self.label_sigma)
        # normalize
        self.label_filter = self.label_filter / np.max(self.label_filter) 
        # convert to torch tensor
        self.label_filter = torch.from_numpy(self.label_filter)

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DeeperCut's methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Local refinement
def set_locref(scmap, locref_map, locref_mask, locref_scale, j, i, j_id, dx, dy):
    locref_mask[j, i, j_id * 2 + 0] = 1
    locref_mask[j, i, j_id * 2 + 1] = 1
    locref_map[j, i, j_id * 2 + 0] = dx * locref_scale
    locref_map[j, i, j_id * 2 + 1] = dy * locref_scale
    scmap[j, i, j_id] = 1
    
def compute_locref_maps(landmarks, size, landmark_names, lowres=False):
    # Set params
    stride = STRIDE
    half_stride = STRIDE/2
    scale = 1
    scaled_img_size = np.array(size)*scale
    size = np.ceil(scaled_img_size / (stride * 2)).astype(int) * 2
    dist_thresh = DIST_THRESHOLD * scale
    locref_stdev = LOCREF_STDEV
    
    width = size[1]
    height = size[0]
    dist_thresh_sq = dist_thresh ** 2
    
    nlandmarks = len(landmark_names)
    scmap = np.zeros(np.concatenate([size, np.array([nlandmarks])]))
    locref_shape = np.concatenate([size, np.array([nlandmarks * 2])])
    locref_mask = np.zeros(locref_shape)
    locref_map = np.zeros(locref_shape)
    
    for k, j_id in enumerate(landmark_names):
        joint_pt = landmarks[k][::-1]
        j_x = joint_pt[0]
        j_y = joint_pt[1]
        if ~np.isnan(j_x) or ~np.isnan(j_y):
            # don't loop over entire heatmap, but just relevant locations
            j_x_sm = round((j_x - half_stride) / stride)
            j_y_sm = round((j_y - half_stride) / stride)
            min_x = int(round(max(j_x_sm - dist_thresh - 1, 0)))
            max_x = int(round(min(j_x_sm + dist_thresh + 1, width - 1)))
            min_y = int(round(max(j_y_sm - dist_thresh - 1, 0)))
            max_y = int(round(min(j_y_sm + dist_thresh + 1, height - 1)))
            
            for j in range(min_y, max_y + 1):  # range(height):
                pt_y = j * stride + half_stride
                for i in range(min_x, max_x + 1):  # range(width):
                    pt_x = i * stride + half_stride
                    dx = j_x - pt_x
                    dy = j_y - pt_y
                    dist = dx ** 2 + dy ** 2
                    if dist <= dist_thresh_sq:
                        locref_scale = 1.0 / locref_stdev
                        set_locref(scmap, locref_map, locref_mask, locref_scale, j, i, k, dx, dy)
    return scmap.T, locref_map.T, locref_mask.T
