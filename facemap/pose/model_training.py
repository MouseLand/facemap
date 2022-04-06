
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

def preprocess_images_landmarks(imgs, landmarks, bbox_region):
    """
    The function preprocesses the images and landmarks by cropping the images and
    landmarks to the bounding box region and resizing the images to 256x256.
    Parameters:
        imgs: A list of multiple images of size (height, width): [(num_images, height, width), ...]
        landmarks: A list of multiple landmarks of size (num_landmarks, 2): [(num_images, num_landmarks, 3), ...]
        bbox_region: A list of bounding box regions for each set of frames [(Xstart, Xstop, Ystart, Ystop, resize), ...]
    Returns:
        imgs_preprocessed: ND-array of images of size (num_frames, 1, 256, 256)
        landmarks_preprocessed: ND-array of landmarks of size (num_frames, num_landmarks, 2)
    """
    imgs_preprocessed = []
    landmarks_preprocessed = []
    # Loop through each list of frames and landmarks
    for set in range(len(imgs)):
        Xstart, Xstop, Ystart, Ystop, resize = bbox_region[set]
        # Loop through each frame
        for frame in range(len(imgs[set])):
            Xlabel, Ylabel = landmarks[set][frame].T[::3], landmarks[set][frame].T[1::3]
            Xlabel, Ylabel = transforms.labels_crop_resize(Xlabel, Ylabel, Ystart, Xstart, current_size=imgs[set][frame].shape, 
                                                            desired_size=(256, 256))
            # Adjust corrected annotations to the cropped region
            #landmarks[set][frame].T[::3] = (landmarks[set][frame].T[::3]- bbox_region[set][0])/ ((bbox_region[set][1]-bbox_region[set][0])/256)
            #landmarks[set][frame].T[1::3] = (landmarks[set][frame].T[1::3]- bbox_region[set][2])/ ((bbox_region[set][3]-bbox_region[set][2])/256)
            landmarks = np.hstack((Xlabel, Ylabel))
            landmarks = landmarks.reshape((-1, 2))
            landmarks_preprocessed.append(landmarks) #landmarks[set][frame])
            # Pre-processing using grayscale imagenet stats
            im = imgs[set][frame]
            if im.ndim==2:
                im = im[np.newaxis, np.newaxis, :, :]
            im = torch.from_numpy(im).to(dtype=torch.float32)
            im = transforms.crop_resize(im, Ystart, Ystop, Xstart, Xstop, resize).clone().detach()
            im = transforms.preprocess_img(im).numpy()
            """
            im = np.multiply(imgs[set][frame], 1 / 255.0).astype(np.float32)
            im = transforms.normalize99(im)
            im = im[bbox_region[set][2]:bbox_region[set][3], bbox_region[set][0]:bbox_region[set][1]]
            im = cv2.resize(im, (256,256))
            """
            imgs_preprocessed.append(im.squeeze())
    imgs_preprocessed = np.array(imgs_preprocessed)
    landmarks_preprocessed = np.array(landmarks_preprocessed)
    return imgs_preprocessed, landmarks_preprocessed

def finetune_model(imgs, landmarks, net, batch_size, n_epochs=36):

    # Train the model on a subset of the corrected annotations
    nimg = imgs.shape[0]
    if imgs.ndim==3:
        imgs = imgs[:, np.newaxis, :, :]
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

