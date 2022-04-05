
## Import packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from . import FMnet_torch as model
from . import pose_helper_functions as pose_utils
from . import pose
from . import transforms
import cv2

"""
Fine-tuning the model using the pre-trained weights and refined training data
provided by the user.
"""

# TO-DO 
#    Use the user selected bounding box settings to crop the images and landmarks before training

def load_images_from_video(video_path, selected_frame_ind):
    """
    Load images from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for frame_ind in selected_frame_ind:
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_ind:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print("Error reading frame")    
    frames = np.array(frames)
    return frames

def preprocess_images_landmarks(imgs, landmarks, bbox_region):
    """
    The function preprocesses the images and landmarks by cropping the images and
    landmarks to the bounding box region and resizing the images to 256x256.
    Parameters:
        imgs: ND-array of images of size (num_frames, height, width)
        landmarks: ND-array of landmarks of size (num_frames, num_landmarks, 2)
        bbox_region: list of bounding box regions for each frame of size (num_frames, Xstart, Xstop, Ystart, Ystop, resize)
    Returns:
        imgs_preprocessed: ND-array of images of size (num_frames, 1, 256, 256)
        landmarks_preprocessed: ND-array of landmarks of size (num_frames, num_landmarks, 2)
    """

    # Write a fo loop that iterates over the frames and preprocesses them one by one
    imgs_preprocessed = []
    landmarks_preprocessed = []
    for i in range(len(imgs)):
        img = imgs[i]
        landmark = landmarks[i]
        bbox_region_i = bbox_region[i]
        img_preprocessed, landmark_preprocessed = preprocess_image_landmark(img, landmark, bbox_region_i)
        imgs_preprocessed.append(img_preprocessed)
        landmarks_preprocessed.append(landmark_preprocessed)
    imgs_preprocessed = np.array(imgs_preprocessed)
    landmarks_preprocessed = np.array(landmarks_preprocessed)
    return imgs_preprocessed, landmarks_preprocessed

def preprocess_image_landmark(img, landmark, bbox_region):
    """
    The function takes one image and the respective keypoints for the image and adjusts both of them using the 
    bounding box region supplied.
    Parameters:
        img: ND-array of images of size (height, width)
        landmark: ND-array of landmarks of size (num_landmarks, 2)
        bbox_region: list of bounding box regions for each frame of size (Xstart, Xstop, Ystart, Ystop, resize)
    Returns:
        img_preprocessed: ND-array of images of size (1, 256, 256)
        landmark_preprocessed: ND-array of landmarks of size (num_landmarks, 2)
    """
    # If bbox_region is not square, then adjust the bbox_region to be square
    if bbox_region[2] - bbox_region[0] != bbox_region[3] - bbox_region[1]:
        if bbox_region[2] - bbox_region[0] > bbox_region[3] - bbox_region[1]:
            bbox_region[2] = bbox_region[0] + bbox_region[3] - bbox_region[1]
        else:
            bbox_region[3] = bbox_region[1] + bbox_region[2] - bbox_region[0]

    # Adjust corrected annotations to the cropped region
    landmark.T[::3] = (landmark.T[::3]- bbox_region[0])/ ((bbox_region[1]-bbox_region[0])/256)
    landmark.T[1::3] = (landmark.T[1::3]- bbox_region[2])/ ((bbox_region[3]-bbox_region[2])/256)
    landmark = landmark.T[landmark.columns.get_level_values("coords")!='likelihood'].T.to_numpy().reshape(-1,15,2)
    # Pre-processing using grayscale imagenet stats
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.multiply(img, 1 / 255.0).astype(np.float32) # np.ndarray float type with values normalized to [0, 1]
    img = transforms.normalize99(img)
    img = img[bbox_region[2]:bbox_region[3], bbox_region[0]:bbox_region[1]]
    img = cv2.resize(img, (256,256))
    return img, landmark

def finetune_model(imgs, landmarks, net, batch_size, n_epochs=36):

    # Train the model on a subset of the corrected annotations
    nimg = len(imgs)
    n_factor =  2**4 // (2 ** net.n_upsample)
    xmesh, ymesh = np.meshgrid(np.arange(256/n_factor), np.arange(256/n_factor))
    ymesh = torch.from_numpy(ymesh).to(device)
    xmesh = torch.from_numpy(xmesh).to(device)
    sigma  = 3 * 4 / n_factor
    Lx = 64

    learning_rate = 1e-4
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    ggmax = 50
    LR = learning_rate * np.ones(n_epochs,)
    LR[-6:-3] = learning_rate/10
    LR[-3:] = learning_rate/25

    for epoch in range(n_epochs): 
        for param_group in optimizer.param_groups:
            param_group['lr'] = LR[epoch]
            
        pose_utils.set_seed(epoch)
        net.train()
        inds = np.random.permutation(nimg)
        train_loss = 0
        train_mean = 0
        n_batches = 0
        
        gnorm_max = 0
        for k in np.arange(0, nimg, batch_size):
            kend = min(nimg, k+batch_size)
            imgi, lbl, _ = transforms.random_rotate_and_resize(imgs[inds[k:kend]], 
                                                        landmarks[inds[k:kend]], 
                                                        contrast_adjustment=False, do_flip=True,
                                                        scale_range=0.2, rotation=0, gamma_aug=False)
            
            #### run the network FIRST for asynchronous CPU work below ##########
            img_batch = torch.from_numpy(imgi).to(device=device, dtype=torch.float32)
            hm_pred, locx_pred, locy_pred = net(img_batch)
            ######################################################################
            
            # do a lot of preparations for the true heatmaps and the location graphs
            lbl_mask = np.isnan(lbl).sum(axis=-1)
            is_nan = lbl_mask > 0
            lbl[is_nan] = 0
            lbl_nan = torch.from_numpy(lbl_mask == 0).to(device=device)
            lbl_batch = torch.from_numpy(lbl).to(device=device, dtype=torch.float32)
            
            # divide by the downsampling factor (typically 4)
            y_true = (lbl_batch[:,:,0]) / n_factor
            x_true = (lbl_batch[:,:,1]) / n_factor
            
            # relative locationsof keypoints
            locx = (ymesh - x_true.unsqueeze(-1).unsqueeze(-1))
            locy = (xmesh - y_true.unsqueeze(-1).unsqueeze(-1))
                    
            # normalize the true heatmaps
            hm_true = torch.exp(-(locx**2 + locy**2) / (2*sigma**2))
            hm_true = 10 * hm_true / (1e-3 + hm_true.sum(axis=(-2,-1)).unsqueeze(-1).unsqueeze(-1))
            
            # mask over which to train the location graphs
            mask = (locx**2 + locy**2)**.5 <= sigma
            
            # normalize the location graphs for prediction       
            locx = locx/(2*sigma)
            locy = locy/(2*sigma)
            
            # mask out nan's
            hm_true = hm_true[lbl_nan]
            y_true = y_true[lbl_nan]
            x_true = x_true[lbl_nan]
            locx = locx[lbl_nan]
            locy = locy[lbl_nan]
            mask = mask [lbl_nan]
            
            # subsample the non-nan heatmaps and location graphs
            hm_pred   = hm_pred[lbl_nan]
            locx_pred = locx_pred[lbl_nan]
            locy_pred = locy_pred[lbl_nan]
            
            # heatmap loss
            loss = ((hm_true - hm_pred).abs()).sum(axis=(-2,-1))
            
            # loss from the location graphs, masked with mask
            # I use a weighting of 0.5. Much smaller or much bigger worked almost as well (0.05 and 5)
            loss += .5 * (mask * ((locx - locx_pred)**2 + (locy - locy_pred)**2)**.5).sum(axis = (-2,-1))
            
            with torch.no_grad():    
                # this part computes the position error on the training set
                hm_pred   = hm_pred.reshape(hm_pred.shape[0], Lx*Lx)            
                locx_pred = locx_pred.reshape(locx_pred.shape[0], Lx*Lx)
                locy_pred = locy_pred.reshape(locy_pred.shape[0], Lx*Lx)

                nn = hm_pred.shape[0]
                imax = torch.argmax(hm_pred, 1)

                x_pred = ymesh.flatten()[imax] - (2*sigma) * locx_pred[np.arange(nn), imax]
                y_pred = xmesh.flatten()[imax] - (2*sigma) * locy_pred[np.arange(nn), imax]

                y_err = (y_true - y_pred).abs()
                x_err = (x_true - x_pred).abs()
                
                train_mean += ((y_err + x_err)/2).mean().item()

            loss = loss.mean()
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()        
            
            # this operation clips the gradient and returns its original norm
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), ggmax)
            # keep track of the largest gradient norm on this epoch
            gnorm_max = np.maximum(gnorm_max, gnorm.cpu())     
            
            optimizer.step()
            
            n_batches+=1
        
        train_loss /= n_batches
        train_mean /= n_batches

        if epoch % 10 == 0:
            print('Epoch %d: loss %f, mean %f, gnorm %f' % (epoch, train_loss, train_mean, gnorm_max))

    return net

