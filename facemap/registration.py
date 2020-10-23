import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import scipy.stats
from scipy.ndimage import filters
from math import pi
import sklearn.cluster
import time
import skimage.transform 
import skimage.registration
from scipy.ndimage import gaussian_filter
import pdb

from . import utils, process



'''
MOTION TRACES
'''

def run_movie_proc(filename, save_path=None):
    """ compute movie masks and movie SVDs for single movie"""
    cumframes, Ly, Lx, containers = utils.get_frame_details([[filename]])        
    avgframe, avgmotion = process.subsampled_mean(containers, cumframes, Ly, Lx, 1)
    nframes = cumframes[-1]

    # load in chunks of up to 1000 frames
    nt0 = min(1000, nframes)
    nsegs = int(min(np.floor(15000 / nt0), np.floor(nframes / nt0)))
    nc = int(250) # <- how many PCs to keep in each chunk
    nc = min(nc, nt0-1)
    if nsegs==1:
        nc = min(ncomps, nt0-1)
    # what times to sample
    tf = np.floor(np.linspace(0, nframes-nt0-1, nsegs)).astype(int)
    Lyb, Lxb = Ly[0], Lx[0]
    U = np.zeros((Lyb*Lxb, nsegs*nc), np.float32)
    ni = [0]
    ns = 0

    for n in range(nsegs):
        img = [np.zeros((nt0, Lyb, Lxb), np.float32)]
        t = tf[n]
        utils.get_frames(img, containers, np.arange(t,t+nt0), cumframes, Ly, Lx)
        ii = 0
        imall = img[ii].reshape(-1, Lyb*Lxb)
        imall -= avgframe[ii]
        ncb = min(nc, imall.shape[-1])
        usv  = utils.svdecon(imall.T, k=ncb)
        ncb = usv[0].shape[-1]
        U[:, ni[0]:ni[0]+ncb] = usv[0] * usv[1]
        ni[0] += ncb
        ns+=1
        if ns%5==0:
            print(ns)

    U = U[:, :ni[0]]
    ncomps = 500
    usv = utils.svdecon(U, k = min(ncomps, U.shape[1]-1))
    U = usv[0]

    print('U computed')

    # compute in chunks of 500
    nt0 = 500
    nsegs = int(np.ceil(nframes / nt0))
    t,nt1 = 0,0
    V = np.zeros((nframes, ncomps), np.float32)
    for n in range(nsegs):
        t += nt1
        img = [np.zeros((nt0, Lyb, Lxb), np.float32)]
        utils.get_frames(img, containers, np.arange(t,t+nt0), cumframes, Ly, Lx)
        nt1 = img[0].shape[0]
        
        imall = img[ii].reshape(-1, Lyb*Lxb)
        imall -= avgframe[ii]
        vproj = imall @ U
        V[t:t+vproj.shape[0], :] = vproj

    print('V computed')

    if save_path is None:
        froot = os.path.splitext(filename)[0]
        np.save(froot+'_proc.npy', {'avgframe': avgframe, 'movMask': [U], 
                                    'movSVD': [V], 'Ly': Ly, 'Lx': Lx})
    else:
        froot = os.path.splitext(os.path.split(filename)[-1])[0]
        np.save(os.path.join(save_path, froot+'_proc.npy'), {'avgframe': avgframe, 'movMask': [U], 
                                    'movSVD': [V], 'Ly': Ly, 'Lx': Lx})
        



def imall_init(nfr, Ly, Lx):
    imall = []
    for n in range(len(Ly)):
        imall.append(np.zeros((nfr,Ly[n],Lx[n]), 'uint8'))
    return imall

def get_newV_movie(filenames, U_new, crop_vals, tform='none', nframes='none'):
    ''' recalculates V after warping U, up to nframes frames '''
    if U_new.ndim==3:
        U_new = U_new.reshape(-1, U_new.shape[-1])
    start = time.time()
    cumframes, Ly_old, Lx_old, containers = utils.get_frame_details(filenames)
    Ly_old = Ly_old[0]; Lx_old = Lx_old[0]
    avgframe,_ = process.subsampled_mean(containers, cumframes, [Ly_old], [Lx_old], sbin=1)
    avgframe = np.reshape(avgframe[0],(Ly_old,Lx_old))
    
    V_new = np.zeros([cumframes[-1],U_new.shape[1]])
    chunk_len = 1000
    nt = int(np.ceil(cumframes[-1]/chunk_len)) #how many chunks (including last incomplete one)
    if not isinstance(nframes,str):
        nt= int(np.ceil(nframes/chunk_len))
    for i in range(nt):
        cframes = range(i*chunk_len,i*chunk_len+chunk_len)
        this_V = calc_newV(containers, cumframes, avgframe, U_new, 
                            Ly_old, Lx_old, 
                            crop_vals, cframes, 
                            False, tform)
        V_new[i*chunk_len:i*chunk_len+len(this_V),:] = this_V
    
        if i%100 == 0:
            print('Projection {} of {}, time {}s'.format(i+1,nt,(time.time() - start)))
    
    if not isinstance(nframes,str):
        V_new = V_new[:nframes,:]
    
    return V_new

def get_newV(filenames, U_new, crop_vals, tform='none', nframes='none'):
    ''' recalculates V after warping U, up to nframes frames '''
    if U_new.ndim==3:
        U = U_new.reshape(-1, U_new.shape[-1])
    start = time.time()
    cumframes, Ly_old, Lx_old = utils.get_frame_details(filenames)
    Ly_old = Ly_old[0]; Lx_old = Lx_old[0]
    _, avgmotion = process.subsampled_mean(filenames, cumframes, [Ly_old], [Lx_old], sbin=1)
    avgmotion = np.reshape(avgmotion[0],(Ly_old,Lx_old))
    
    V_new = np.zeros([cumframes[-1],U_new.shape[1]])
    chunk_len = 1000
    nt = int(np.ceil(cumframes[-1]/chunk_len)) #how many chunks (including last incomplete one)
    if not isinstance(nframes,str):
        nt= int(np.ceil(nframes/chunk_len))
    for i in range(nt):
        cframes = range(i*chunk_len,i*chunk_len+chunk_len)
        this_V = calc_newV(filenames, cumframes, avgmotion, U_new, Ly_old, Lx_old, crop_vals, cframes, tform)
        V_new[i*chunk_len:i*chunk_len+len(this_V),:] = this_V
    
        if i%20 == 0:
            print('Projection {} of {}, time {}s'.format(i+1,nt,(time.time() - start)))
    
    if not isinstance(nframes,str):
        V_new = V_new[:nframes,:]
    
    return V_new


def calc_newV(containers, cumframes, avgf, U_new, Ly_old, Lx_old, 
                crop_vals, cframes, motion=True,
              tform='none'):
    ''' this function calculates post-warp V for the chunk of times specified by cframes '''
    ''' note: remember first frame of V is a filler '''
    
    xl, xr, yl, yr = crop_vals.astype(int)
    Ly = yr - yl + 1
    Lx = xr - xl + 1
    
    # cframes adjustments
    nframes = cumframes[-1]
    cframes = np.maximum(0, np.minimum(nframes-1, cframes)) #make sure not going over video time
    cframes = np.arange(cframes[0]-1*motion, cframes[-1]+1).astype(int) #add onto the beginning to take diff
    
    # let's get X
    firstframe=0
    imall = imall_init(cframes.shape[0],[Ly_old],[Lx_old])
    if cframes[0] == -1: #this is the first frame
        cframes = cframes[1:]
        firstframe = 1
    utils.get_frames(imall, containers, cframes, cumframes, [Ly_old], [Lx_old])
    #fm.get_frames_cv2(imall, [vidfile], cframes, cumframes, [Ly_old], [Lx_old])
    if motion:
        X = np.abs(np.diff(imall[0],axis=0))
    else:
        X = imall[0].astype(np.float32)
    X -= avgf
    
    # trim
    X = X[:,yl:yr+1,xl:xr+1]
    
    X = np.reshape(X,(X.shape[0],-1))
    X = np.transpose(X, (1,0)).astype(np.float32)

    # calculate new V
    V_new = X.T @ U_new
    
    if firstframe and motion:
        V_new = np.insert(V_new,0,V_new[0,:],axis=0)
    
    return V_new


#%%

'''
EIGENFACES
'''


def warp_U(U, Ly, Lx, rigid_tform, crop_data, warp_mat):
    '''
    
    Parameters
    ----------
    U : U to be warped
    Ly : Ly of image
    Lx : Lx of image
    rigid_tform : scikit-image AffineTransform object for rigid transformations
    warp_mat : Ly x Lx warp matrix; output of the get_warp_mat function

    Returns
    -------
    U_warp : Warped U matrix

    '''
    
    U_ims = z_score_U(U, Ly, Lx, return_im=1)
    xl,xr,yl,yr = np.array(crop_data,dtype=int)
    Ly_new = yr - yl + 1
    Lx_new = xr - xl + 1
    U_warp = np.zeros((Ly_new, Lx_new, U_ims.shape[2]))

    for i in range(U_ims.shape[2]):
        U_im = skimage.transform.warp(U_ims[:,:,i], rigid_tform, preserve_range=True) #rigid transform
        U_im = U_im[yl:yr+1,xl:xr+1]
        U_warp[:,:,i] = skimage.transform.warp(U_im, warp_mat, mode='constant', preserve_range=True) #nonrigid
    
    # get indices in original
    Y, X = np.meshgrid(np.arange(0, Ly, 1, int), np.arange(0, Lx, 1, int))
    yout = skimage.transform.warp(Y, rigid_tform.inverse, order=0, preserve_range=True)
    xout = skimage.transform.warp(X, rigid_tform.inverse, order=0, preserve_range=True)
    crop_ref = np.array([xout.min(), xout.max(), yout.min(), yout.max()], int)
    print(crop_ref)

    return U_warp, crop_ref

def transform_U(videos, procs, plot=0):
    '''
    for now assumes NO DOWNSAMPLING

    Parameters
    ----------
    vidname0 : name of reference video (should have smallest Ly x Lx)
        can set to 'none' to calculate the smallest video
    other_vidnames : list of names of videos to align to
    plot : option to plot a few steps of the morphing process and final U images
    
    Returns
    -------
    warpedU : list of U's (flat), first index is U of reference cropped to same size as the other U's
    warp_info : list of dictionaries with information useful for reproducing warping outside of the function
    crop_vals : final xl, xr, yl, yr values used to crop from original to warped size
    
    '''
    
    n_videos = len(videos)
    
    warpedU = []
    warp_info = []
    
    # now get data for the reference image
    reference_video_file, reference_proc_file = videos[0], procs[0]
    cumframes, Ly0, Lx0, _ = utils.get_frame_details([[reference_video_file]])
    Ly_ref, Lx_ref = Ly0[0], Lx0[0]
    proc_ref = np.load(reference_proc_file, allow_pickle=True).item()
    
    avg_ref = z_score_im(proc_ref['avgframe'][0], Ly_ref, Lx_ref, return_im=1)
    V_ref = proc_ref['movSVD'][0]
    print('getting representative reference image')
    repim_ref = get_rep_image(reference_video_file, avg_ref, V_ref, Ly_ref, Lx_ref, cutoff = 0.0002, plot=plot)
    U_ref = z_score_U(proc_ref['movMask'][0], Ly_ref, Lx_ref, return_im=0)
    warpedU.append(U_ref) #first idx will be U from the reference
    warp_info.append([1, 1])
    del proc_ref
    
    crop_align = np.zeros((n_videos,4), int) #space for xl,xr,yl,yr
    crop_ref = np.zeros((n_videos,4), int) #space for xl,xr,yl,yr
    
    # now do the warping for the other videos
    for k in range(1,n_videos):
        print(videos[k])
        #load these individually so don't load in too much data at once
        proc_align = np.load(procs[k], allow_pickle=True).item()
        Ly_align = proc_align['Ly'][0]
        Lx_align = proc_align['Lx'][0]
        avg_align = z_score_im(proc_align['avgframe'][0], Ly_align, Lx_align, return_im=1)
        V_align = proc_align['movSVD'][0]
        print('getting representative align-to image')
        repim_align = get_rep_image(videos[k], avg_align, V_align, Ly_align, Lx_align, cutoff = 0.0002, plot=plot)
        repim_align_orig = repim_align.copy()
        U_orig = z_score_U(proc_align['movMask'][0], Ly_ref, Lx_ref, return_im=0)
        del proc_align

        # make sure sizes match, and if they don't, resize
        yadd, xadd = 0, 0
        if Ly_ref != Ly_align or Lx_ref != Lx_align:
            print('{} has size {} x {} instead of {} x {}'.format(videos[k],Ly_align,Lx_align,Ly_ref,Lx_ref))
            if Ly_ref > Ly_align:
                repim_align  = np.concatenate((repim_align, np.zeros((Ly_ref-Ly_align, Lx_align), np.uint8)), axis=0)
                yadd = Ly_ref-Ly_align
            if Lx_ref > Lx_align:
                repim_align  = np.concatenate((repim_align, np.zeros((Ly_ref, Lx_ref-Lx_align), np.uint8)), axis=1)
                xadd = Lx_ref-Lx_align
            print(repim_align.shape)
            
        # now calculate matrices for transformation (rigid, crop, then nonrigid)
        rigid_tform, rigid_im_ref = get_rigid_warp_mat(repim_align, repim_ref)
        print(rigid_im_ref.shape)
        rigid_im_crop_ref, Lx_crop, Ly_crop, xl, xr, yl, yr = crop_image(rigid_im_ref, Ly_ref, Lx_ref)
        print('cropped inds {}:{}, {}:{}'.format(yl, yr, xl, xr))
        repim_align = repim_align[yl:yr+1,xl:xr+1]
        nonrigid_tform = get_nonrigid_warp_mat(repim_align, rigid_im_crop_ref, plot=plot)
        crop_align[k,:] = np.array([xl,xr,yl,yr], dtype=int)
        
        warp_info.append([[Ly_ref, Lx_ref], rigid_tform, nonrigid_tform])

        # warp U's using matrices calculated above
        U_align, crop_refk = warp_U(U_ref, Ly_ref, Lx_ref, rigid_tform, crop_align[k,:], nonrigid_tform)
        U_align = z_score_U(U_align, Ly_crop, Lx_crop, return_im=0)
        U_align = np.reshape(U_align, (Ly_crop,Lx_crop,500))
        yadd = max(0, yadd - (Ly_ref - crop_align[k,3] - 1))
        xadd = max(0, xadd - (Lx_ref - crop_align[k,1] - 1))
        if yadd>0:
            U_align = U_align[:-yadd]
        if xadd>0:
            U_align = U_align[:,:-xadd]
        U_align /= (U_align**2).sum(axis=(0,1))
        print(U_align.shape, repim_align_orig[yl:yr+1,xl:xr+1].shape)
        warpedU.append(U_align)
        crop_ref[k] = crop_refk
        
        # adjust the reference image to this crop
        U_crop = np.reshape(U_ref.copy(), (Ly_ref,Lx_ref,500))
        xl, xr, yl, yr = crop_ref[k]
        U_crop = U_crop[yl:yr+1,xl:xr+1,:]
        Ly_crop_ref, Lx_crop_ref = U_crop.shape[:2]
        U_crop = z_score_U(U_crop, Ly_crop_ref, Lx_crop_ref, return_im=0)
        U_crop /= (U_crop**2).sum(axis=0)
        U_crop = np.reshape(U_crop, (Ly_crop_ref, Lx_crop_ref, 500))
        #if len(other_vidnames) == 1:
        #    warpedU[0] = vid0_U_crop # replace 1st idx with cropped one
        #    crop_vals = crop_data[idx,:]
        
        # plot the warped and unwarped U's for comparison
        if plot:
            plt.figure(figsize=(12,9))
            U0_im = np.reshape(U_ref,(Ly_ref,Lx_ref,-1))
            U1_im = np.reshape(U_orig,(Ly_align,Lx_align,-1))
            for i in range(0,12,4):
                ax=plt.subplot(3,4,i+1)
                ax.imshow(U0_im[:,:,i])#, vmin=-2, vmax=2)
                ax.set_title('reference mask')
                ax.axis('off')
                ax=plt.subplot(3,4,i+2)
                ax.imshow(U_crop[:,:,i])#, vmin=-2, vmax=2)
                ax.set_title('reference mask cropped')
                ax.axis('off')
                #ax=plt.subplot(3,4,i+2)
                #ax.imshow(repim_align)
                ax=plt.subplot(3,4,i+3)
                ax.imshow(U_align[:,:,i])#, vmin=-2, vmax=2)
                ax.set_title('reference mask warped')
                ax.axis('off')
                ax=plt.subplot(3,4,i+4)
                ax.imshow(U1_im[:,:,i])#, vmin=-2, vmax=2)
                ax.set_title('mask1')
                ax.axis('off')
            plt.suptitle('{} movie masks warped to {} axes'.format(reference_video_file, videos[1]))
            plt.show()
    return warpedU, warp_info, crop_align, crop_ref

    

def get_warped_Us(reference_files, other_vidnames,plot=0, use_rep=0):
    '''

    Parameters
    ----------
    vidname0 : name of reference video (should have smallest Ly x Lx)
        can set to 'none' to calculate the smallest video
    other_vidnames : list of names of videos that have Us to align
    plot : option to plot a few steps of the morphing process and final U images
    use_rep : use representative image rather than the average image to get warping

    Returns
    -------
    warpedU : list of U's (flat), first index is U of reference cropped to same size as the other U's
    warp_info : list of dictionaries with information useful for reproducing warping outside of the function
    crop_vals : final xl, xr, yl, yr values used to crop from original to warped size
    
    '''
    
    warpedU = []
    warp_info = []
    V_orig = []
    
    
    # now get data for the reference image
    vidname0, proc0 = reference_files[0], reference_files[1]
    vidname1, proc1 = other_vidnames[0][0], other_vidnames[0][1]
    vid0 = np.load(proc0,  allow_pickle=True).item()
    vid1 = np.load(proc1,  allow_pickle=True).item()

    Ly0 = vid0['Ly'][0]
    Lx0 = vid0['Lx'][0]
    vid0_avg = z_score_im(vid0['avgframe'][0], Ly0, Lx0, return_im=1)
    vid0_V = vid0['motSVD'][0]
    vid0_repim = get_rep_image(vidname0, vid0_avg, vid0_V, Ly0, Lx0, cutoff = 0.0002, plot=0)
    vid0_U = z_score_U(vid0['motMask'][0], Ly0, Lx0, return_im=0)
    warpedU.append(vid0_U) #first idx will be U from the reference
    V_orig.append(vid0_V)
    del vid0
    
    crop_data = np.zeros((2,4)) #space for xl,xr,yl,yr
    idx = 1
    # now do the warping for the other videos
    for k in range(1):
        #load these individually so don't load in too much data at once
        Ly1 = vid1['Ly'][0]
        Lx1 = vid1['Lx'][0]
        
        vid1_avg = z_score_im(vid1['avgframe'][0], Ly1, Lx1)
        vid1_V = vid1['motSVD'][0]
        V_orig.append(vid1_V)
        vid1_repim = get_rep_image(vidname1, vid1_avg, vid1_V, Ly1, Lx1, cutoff = 0.0002, plot=0)
        vid1_U = z_score_U(vid1['motMask'][0], Ly1, Lx1,return_im=0)
        del vid1
        
        # make sure sizes match, and if they don't, resize
        if Ly0 != Ly1 or Lx0 != Lx1:
            print('{} has size {} x {} instead of {} x {}'.format(vidname1,Ly1,Lx1,Ly0,Lx0))
            vid1_repim = skimage.registration.resize(vid1_repim,(Ly0,Lx0,-1),anti_aliasing=True)
            vid1_U = resize_U(vid1_U,return_im=0)
        
        # now calculate matrices for transformation (rigid, crop, then nonrigid)
        rigid_tform, vid1_avg_rigid = get_rigid_warp_mat(vid0_repim, vid1_repim)
        print(vid1_avg_rigid.shape)
        vid1_avg_rigid_crop,Lx_crop,Ly_crop,xl,xr,yl,yr = crop_image(vid1_avg_rigid, Ly1, Lx1)
        vid0_avg_crop = vid0_repim[yl:yr+1,xl:xr+1]
        warp_mat = get_nonrigid_warp_mat(vid0_avg_crop, vid1_avg_rigid_crop,plot=plot)
        crop_data[idx,:] = np.array([xl,xr,yl,yr], dtype=int)
        
        # adjust the reference image to this crop
        vid0_U_crop = np.reshape(vid0_U,(Ly0,Lx0,500))
        vid0_U_crop = vid0_U_crop[yl:yr+1,xl:xr+1,:]
        vid0_U_crop = z_score_U(vid0_U_crop, Ly_crop, Lx_crop, return_im=0)
        vid0_U_crop /= (vid0_U_crop**2).sum(axis=0)
        vid0_U_crop = np.reshape(vid0_U_crop,(Ly_crop,Lx_crop,500))
        if len(other_vidnames) == 1:
            warpedU[0] = vid0_U_crop # replace 1st idx with cropped one
            crop_vals = crop_data[idx,:]
        
        # warp U's using matrices calculated above
        vid1_U_warped = warp_U(vid1_U, Ly0, Lx0, rigid_tform, crop_data[idx,:], warp_mat)
        vid1_U_warped = z_score_U(vid1_U_warped, Ly_crop, Lx_crop, return_im=0)
        vid1_U_warped /= (vid1_U_warped**2).sum(axis=0)
        vid1_U_warped = np.reshape(vid1_U_warped,(Ly_crop,Lx_crop,500))
        warpedU.append(vid1_U_warped)
        
        this_warp = {
            'vidname': vidname1, 'Ly': Ly1, 'Lx': Lx1, 'rigid_transform': rigid_tform, 
            'Ly_crop': Ly_crop, 'Lx_crop': Lx_crop, 'xl': xl, 'xr': xr, 'yl': yl, 'yr': yr, 
            'warp_mat': warp_mat
            }
        warp_info.append([this_warp])
        
        idx += 1
        
        # plot the warped and unwarped U's for comparison
        if plot:
            plt.figure(figsize=(12,9))
            U0_im = np.reshape(vid0_U,(Ly0,Lx0,-1))
            U1_im = np.reshape(vid1_U,(Ly1,Lx1,-1))
            for i in range(0,12,4):
                ax=plt.subplot(3,4,i+1)
                ax.imshow(U0_im[:,:,i])#, vmin=-2, vmax=2)
                ax.set_title('mask0')
                ax.axis('off')
                ax=plt.subplot(3,4,i+2)
                ax.imshow(vid0_U_crop[:,:,i])#, vmin=-2, vmax=2)
                ax.set_title('mask0 cropped')
                ax.axis('off')
                ax=plt.subplot(3,4,i+3)
                ax.imshow(vid1_U_warped[:,:,i])#, vmin=-2, vmax=2)
                ax.set_title('mask1 warped')
                ax.axis('off')
                ax=plt.subplot(3,4,i+4)
                ax.imshow(U1_im[:,:,i])#, vmin=-2, vmax=2)
                ax.set_title('mask1')
                ax.axis('off')
            plt.suptitle('{} motion masks warped to {} axes'.format(vidname1,vidname0))
            plt.show()
    
    return warpedU, warp_info, crop_data, V_orig

'''
CALCULATE WARPING
'''

def get_nonrigid_warp_mat(im0, im1, plot=0, num_warp=5, num_iter=10, tol=0.0001, prefilter=False):
    '''
    Parameters
    ----------
    im0 : z-scored 2d reference image, usually an avgframe from a video
    im1 : z-scored 2d image to align to the reference im0, usually rigid-transformed avgframe of another vid
    plot : default is 0; whether to plot the warping results
    other parameters are parameters for skimage.registration.optical_flow_tvl1. 
            most relevant to adjust are attachment and tightness.

    Returns
    -------
    warp_mat : 2d warping matrix for transforming im1 to same coord axis as im0

    '''
    
    Ly, Lx = im0.shape
    im0_z = z_score_im(im0, Ly, Lx)
    
    lowest_sse = float('inf')
    for att in np.arange(8,15,1):
        for tight in np.arange(0.2,0.8,0.1):
            v,u = skimage.registration.optical_flow_tvl1(im0, im1, attachment=att, tightness=tight, num_warp=num_warp,num_iter=num_iter,tol=tol,prefilter=prefilter)
            row_coords, col_coords = np.meshgrid(np.arange(Ly), np.arange(Lx), 
                                                 indexing = 'ij')
            this_warp_mat = np.array([row_coords + v, col_coords + u])
            
            this_im1w = skimage.transform.warp(im1.copy(),this_warp_mat,mode='constant')#, preserve_range=True)
            this_im1w = z_score_im(this_im1w, Ly, Lx)
            this_sse = np.sum((this_im1w-im0_z)**2)
            if this_sse < lowest_sse:
                lowest_sse = this_sse
                attachment = att
                tightness = tight
                warp_mat = this_warp_mat
                im1w = this_im1w

    # evaluate accuracy of warping
    im_overlap = np.zeros([Ly,Lx,3])
    im_overlap[:,:,0] = im0_z
    im_overlap[:,:,1] = im1w
    im_overlap[:,:,2] = im1w
    plt.imshow(im_overlap)
    plt.axis('off')
    plt.title('overlaid average images post-warping')
    plt.show()
    print('attachment: {}, tightness: {}'.format(attachment, tightness))
    print('sum of squared errors: {}'.format(lowest_sse))
    
            
    if plot:
        plt.figure(figsize=(12,3))
        ax1 = plt.subplot(131)
        ax1.imshow(im0_z,vmin=-2,vmax=2)
        ax1.set_title('align-to image cropped')
        ax1.axis('off')
        ax2 = plt.subplot(132)
        ax2.imshow(im1w,vmin=-2,vmax=2)
        ax2.set_title('reference image post-rigid and -nonrigid transform')
        ax2.axis('off')
        ax3 = plt.subplot(133)
        ax3.imshow(z_score_im(im1,Ly,Lx),vmin=-2,vmax=2)
        ax3.set_title('reference image post-rigid transform')
        ax3.axis('off')
    
    return warp_mat


def get_nonrigid_warp_mat_input(im0, im1, plot=0, attachment=8,tightness=0.5,num_warp=5, num_iter=10, tol=0.0001, prefilter=False):
    '''
    Parameters
    ----------
    im0 : z-scored 2d reference image, usually an avgframe from a video
    im1 : z-scored 2d image to align to the reference im0, usually rigid-transformed avgframe of another vid
    plot : default is 0; whether to plot the warping results
    other parameters are parameters for skimage.registration.optical_flow_tvl1. 
            most relevant to adjust are attachment and tightness.

    Returns
    -------
    warp_mat : 2d warping matrix for transforming im1 to same coord axis as im0
    '''

    Ly, Lx = im0.shape
    v,u = skimage.registration.optical_flow_tvl1(im0, im1, attachment=attachment, tightness=tightness, num_warp=num_warp,num_iter=num_iter,tol=tol,prefilter=prefilter)
    row_coords, col_coords = np.meshgrid(np.arange(Ly), np.arange(Lx), 
                                         indexing = 'ij')
    warp_mat = np.array([row_coords + v, col_coords + u])
    
    # evaluate accuracy of warping
    im0 = z_score_im(im0, Ly, Lx)
    im1w = skimage.transform.warp(im1,warp_mat,mode='constant', preserve_range=True)
    im1w = z_score_im(im1w, Ly, Lx)
    sse = np.sum((im1w-im0)**2)
    im_overlap = np.zeros([Ly,Lx,3])
    im_overlap[:,:,0] = im0
    im_overlap[:,:,1] = im1w
    im_overlap[:,:,2] = im1w
    plt.imshow(im_overlap)
    plt.axis('off')
    plt.title('overlaid average images post-warping')
    plt.show()
    print('Sum of squared error: {}'.format(sse))
    
            
    if plot:
        plt.figure(figsize=(8,3))
        ax1 = plt.subplot(131)
        ax1.imshow(im0,vmin=-2,vmax=2)
        ax1.set_title('im0 cropped')
        ax1.axis('off')
        ax2 = plt.subplot(132)
        ax2.imshow(im1w, vmin=-2,vmax=2)
        ax2.set_title('im1 post-rigid and -nonrigid transform')
        ax2.axis('off')
        ax3 = plt.subplot(133)
        ax3.imshow(z_score_im(im1,Ly,Lx),vmin=-2,vmax=2)
        ax3.set_title('im1 post-rigid transform')
        ax3.axis('off')
    
    return warp_mat


def get_rigid_warp_mat(im_align, im_ref, degshift=1, scaleshift=0.05, plot=0):
    '''
    Parameters
    ----------
    im0 : z-scored 2d reference image, usually an avgframe from a video
    im1 : z-scored 2d image to align to the reference im0, usually avgframe of another vid
    degshift : number of degrees to rotate by each time
    scaleshift : proportion of image to try scaling by for a better scale fit
    plot : default is 0; whether to plot transform calculations

    Returns
    -------
    tform : AffineTransform object; use this in the warp function for the rigid 
        transformation (or rather, use its inverse)
    im1_new : This is image1 transformed using tform
    '''
    im_ref_orig = im_ref.copy()
    scales = []
    tforms = []
    Ly_align, Lx_align = im_align.shape
    Ly_ref, Lx_ref = im_ref.shape
    scale = find_scalingfactor(im_align, im_ref, scaleshift=scaleshift)
    scales.append(scale)
    if scale>1.0:
        im_align = cv2.resize(im_align, (int(Lx_align*scale), int(Ly_align*scale)))
        Ly_align, Lx_align = im_align.shape
    else:
        im_ref = cv2.resize(im_ref, (int(Lx_ref/scale), int(Ly_ref/scale)))
        Ly_ref, Lx_ref = im_ref.shape
    yp1 = max(0, Ly_align - Ly_ref)
    xp1 = max(0, Lx_align - Lx_ref)
    im_ref = np.pad(im_ref, ((yp1//2,yp1//2+yp1%2), (xp1//2,xp1//2+xp1%2)))
    yp0 = max(0, Ly_ref - Ly_align)
    xp0 = max(0, Lx_ref - Lx_align)
    im_align = np.pad(im_align, ((yp0//2,yp0//2+yp0%2), (xp0//2,xp0//2+xp0%2)))
    print(im_align.shape, im_ref.shape)    

    num_rotations = int(360/degshift)
    mag = np.zeros(num_rotations)
    shifts = []
    fim_align = np.fft.fft2(im_align)
    angles = np.arange(-15, 16, degshift)
    for i,ang in enumerate(angles):
        im_r = skimage.transform.rotate(im_ref, angle=ang)
        fim_r = np.fft.fft2(im_r)
        im_product = fim_align * fim_r.conj()
        xcorr = np.fft.fftshift(np.fft.ifft2(im_product)).real
        #xcorr = gaussian_filter(xcorr.real, 5)

        maxima = np.unravel_index(np.argmax(xcorr), xcorr.shape) # this is in y,x
        mag[i] = xcorr[maxima]
        shifts.append(np.array(maxima, dtype=np.float64)) #this is still in y,x

    midpoints = np.array([np.floor(axis_size / 2) for axis_size in im_align.shape])

    max_idx = np.argmax(mag.real)
    angle = angles[max_idx]
    shift = shifts[max_idx] - midpoints
    shift = np.flip(shift) #this is in x,y

    print(f"value for CCW rotation (degrees): {angle}")
    print(f"value for translation (y,x): {shift}")

    #now apply transformations together
    tform = skimage.transform.AffineTransform(translation=-1*shift, rotation=(angle*(pi/180)))
    #im_ref = skimage.transform.warp(im_ref, tform)
    matrix = tform.params @ np.array([[scale, 0, -xp1//2], [0, scale, -yp1//2], [0, 0, 1.]])
    tform = skimage.transform.AffineTransform(matrix = matrix)
    im_ref_new = skimage.transform.warp(im_ref_orig, tform)#, preserve_range=True)

    # get indices in original
    Y, X = np.meshgrid(np.arange(0, Ly_ref, 1, int), np.arange(0, Lx_ref, 1, int))
    yout = skimage.transform.warp(Y, tform, order=0, preserve_range=True)
    xout = skimage.transform.warp(X, tform, order=0, preserve_range=True)
    crop_inds = np.array([xout.min(), xout.max(), yout.min(), yout.max()], int)
    
    if plot:
        plot_transformed_img(im_align, im_ref_orig, im_ref,shift=shift,angle=angle,scale=scale)
    
    return tform, im_ref_new#, crop_inds #, im_align, im_ref


def find_scalingfactor(im0, im1, scaleshift=.05):
    '''
    Parameters
    ----------
    im0 : z-scored 2d reference image, usually an avgframe from a video
    im1 : z-scored 2d image to align to the reference im0, usually avgframe of another vid
    scaleshift : step size for change in scale for iterations; default is .05.

    Returns
    -------
    scalingfactor : value that represents optimum scaling factor

    '''
    Ly,Lx = im0.shape
    scales = np.arange(0.65,1.4,scaleshift)
    mag = np.zeros(len(scales))
    for i in range(len(scales)):
        im1_fullscale = skimage.transform.rescale(im1,scale=scales[i],mode='constant')
        im1_s = np.zeros((Ly,Lx))
        fullLy, fullLx = im1_fullscale.shape
        if scales[i] < 1: #if img is smaller
            xpad = int((Lx - fullLx) / 2) 
            ypad = int((Ly - fullLy) / 2)
            im1_s[ypad:ypad+fullLy,xpad:xpad+fullLx] = im1_fullscale #pad image to same size
        elif scales[i] > 1: #if img is larger
            xtrim = int((fullLx-Lx)/2)
            ytrim = int((fullLy-Ly)/2)
            im1_s = im1_fullscale[ytrim:ytrim+Ly,xtrim:xtrim+Lx] #trim image to same size
        else:
            im1_s = im1_fullscale

        im_product = np.fft.fft2(im0) * np.fft.fft2(im1_s).conj()
        xcorr = np.fft.fftshift(np.fft.ifft2(im_product)).real
        xcorr = gaussian_filter(xcorr, 3)
        #maxima = np.unravel_index(np.argmax(xcorr), xcorr.shape) # this is in y,x
        #mag[i] = xcorr[maxima]
        mag[i] = xcorr[int(Ly/2),int(Lx/2)]

    max_idx = np.argmax(mag.real)
    scalingfactor = scales[max_idx]
    print(f"value for scaling: {scalingfactor}")

    return scalingfactor


def plot_transformed_img(im0,im1,im1_trans,shift='none',angle='0', scale='1'):
    '''

    Parameters
    ----------
    image0 : Reference image
    image1 : Offset image
    image1_trans : Offset image transformed to reference image axes
    shift : optional value that image was shifted by. The default is 'none'.
    angle : optional value of angle image was rotated. The default is '0'.
    scale : optional value that image was scaled by. The default is '1'.

    Returns
    -------
    None.
    '''
    
    plt.figure(figsize=(16, 4))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)
    
    ax1.imshow(im0)
    ax1.set_axis_off()
    ax1.set_title('Reference image')
    
    ax2.imshow(im1)
    ax2.set_axis_off()
    ax2.set_title('Offset image')
    
    
    ax3.imshow(im1_trans)
    ax3.set_axis_off()
    ax3.set_title("Transformed image")
    
    plt.show()
    
    print("Detected pixel offset (x, y): {}".format(shift))
    print("Detected angle offset (degrees): {}".format(angle))
    print("Detected scaling factor: {}".format(scale))

#%%
        
'''
CHANGING IMAGE SIZES
'''

def crop_image(im, Ly, Lx, plot=0):
    ''' this function could probably be improved, but currently crops off some of the edge padding '''
    #pdb.set_trace()
    colsizes = np.where(np.count_nonzero(im,axis=0)==0)[0]
    rowsizes = np.where(np.count_nonzero(im,axis=1)==0)[0]
    xlen = colsizes.shape[0]
    ylen = rowsizes.shape[0]
    if xlen > ylen: # do x first
        x_crop, Lx_crop, xl, xr = crop_x(im,Lx,Ly)
        im_cr, Ly_crop, yl, yr = crop_y(x_crop,Lx_crop,Ly,plot=plot)
    else: # do y first
        y_crop, Ly_crop, yl, yr = crop_y(im,Lx,Ly)
        im_cr, Lx_crop, xl, xr = crop_x(y_crop,Ly_crop,Lx,plot=plot)
  
    return im_cr, Lx_crop, Ly_crop, xl, xr, yl, yr


def crop_x(im,Lx,Ly,plot=0):
    ''' this crops out fully-zero edges (columns) '''
    colsizes = np.count_nonzero(im,axis=0)
    x_nonzero = np.where(colsizes>0)[0]
    
    xl = int(np.amin(x_nonzero))
    xr = int(np.amax(x_nonzero))

    x_crop = im[:,xl:xr+1]
    Lx_crop = x_crop.shape[1]

    print('crop left by {} pixels, crop right by {} pixels'.format(xl,xr-Lx))
    print('new x size: {} pixels'.format(Lx_crop))

    return x_crop, Lx_crop, xl, xr


def crop_y(im,Lx,Ly,plot=0):
    ''' this crops out fully-zero edges (rows) '''
    rowsizes = np.count_nonzero(im,axis=1)
    y_nonzero = np.where(rowsizes>0)[0]
    
    yl = int(np.amin(y_nonzero))
    yr = int(np.amax(y_nonzero))

    y_crop = im[yl:yr+1,:]
    Ly_crop = y_crop.shape[0]

    print('crop top by {} pixels, crop bottom by {} pixels'.format(yl,Ly-yr))
    print('new y size: {} pixels'.format(Ly_crop))

    return y_crop, Ly_crop, yl, yr


def resize_U(U1,Ly1,Lx1,Ly0,Lx0,return_im=1):
    ''' for U's that needed to be adjusted for pixel size, resize each PC '''
    if len(U1.shape) != 3:
        U1 = np.reshape(U1,(Ly1,Lx1,-1))
    comps = int(U1.shape[2])
    U1_resized = np.zeros((Ly0,Lx0,comps))
    for i in range(comps):
        U1_resized[:,:,i] = skimage.transform.resize(U1[:,:,i],(Ly0,Lx0),anti_aliasing=True)
    if not return_im:
        U1_resized = np.reshape(U1_resized,(Ly0*Lx0,-1))
    
    return U1_resized


def find_smallest_vid(vidnames):
    '''

    Parameters
    ----------
    vidnames : list of videos.

    Returns
    -------
    list with name of the smallest video (in terms of pixels), or a list of the smallest 2
    vidminLx : smallest Lx
    vidminLy : smallest Ly

    '''

    _, Ly, Lx = grab_videos_cv2(vidnames)
    
    minLx = np.min(Lx).astype(int)
    minLy = np.min(Ly).astype(int)

    #get indices of videos with the smallest Lx and Ly
    vidminLx = [i for i,value in enumerate(Lx) if value == minLx]
    vidminLy = [i for i,value in enumerate(Ly) if value == minLy]
    
    #see if any of the indices match for Lx or Ly
    vididx = set(vidminLx).intersection(vidminLy)

    if len(vididx) > 0: #if 1+ videos has both the smallest Lx and Ly
        refvid_idx = list(vididx)[0] #just take the first one
        print('same video {} has smallest Lx {} and Ly {}'.format(vidnames[refvid_idx],minLx,minLy))
        return [vidnames[refvid_idx]], minLx, minLy
    else: #if different videos have smallest Lx and Ly
        print('{} has smallest Lx'.format(vidnames[vidminLx[0]],minLx))
        print('{} has smallest Ly'.format(vidnames[vidminLy[0]],minLy))
        return [vidnames[vidminLx],vidnames[vidminLy]], minLx, minLy


#%%
'''
CALCULATE REPRESENTATIVE IMAGE
'''

def get_rep_image(vidname, avgframe, V, Ly, Lx, cutoff = 0.0002, plot=0):
    ''' calculate representative image of the video '''
    ''' cutoff scale may need to be adjusted '''
    ''' V is time x PCs '''
    
    V_z = np.transpose(V,(1,0)) #PCsx time now
    V_z *= np.sign(scipy.stats.skew(V_z, axis=0))
    sums = np.sum(np.abs(V_z),axis = 0)
    sums = scipy.stats.zscore(sums)
    
    trest = sums < np.percentile(sums, 0.1)
    #while sum(trest) < 100:
    #    cutoff += 0.0001
    #    trest = (np.abs(V_z)<cutoff).sum(axis=0)==V_z.shape[0]
    #    print(cutoff, sum(trest))
    times = np.where(trest==True)[0]
    
    # let's get these resting images
    imall = imall_init(len(times),[Ly],[Lx]) # let's assume max is 3 clusters
    utils.get_skipping_frames(imall, [[vidname]], times, np.array([0, V.shape[1]]))
    
    _,rep_image = best_rep_combo(avgframe[np.newaxis,:,:], imall[0],plot=plot)
    
    return rep_image


def best_rep_combo(imall0, imall1, plot=0):
    ''' return the 2 images that have highest correlation '''
    
    magnitude = np.zeros([imall0.shape[0],imall1.shape[0]])
    for i in range(imall0.shape[0]):
        for j in range(imall1.shape[0]):
            M0_cent = imall0[i,:,:]
            M1_cent = imall1[j,:,:]
            
            im_product = np.fft.fft2(M0_cent) * np.fft.fft2(M1_cent).conj()
            xcorr = np.fft.fftshift(np.fft.ifft2(im_product))
            maxima = np.unravel_index(np.argmax(xcorr), xcorr.shape) # this is in y,x
            magnitude[i,j] = xcorr.real[maxima]
            
    i,j = np.unravel_index(np.argmax(magnitude),magnitude.shape)
    
    if plot:
        plt.figure(figsize=(8,3))
        plt.subplot(121)
        plt.imshow(imall0[i,:,:])
        plt.title('mean image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(M1_cent)
        plt.title('most correlated image')
        plt.axis('off')
        plt.show()
    
    return imall0[i,:,:], imall1[j,:,:]
   

def center_baseline(V, sigma=100, window=500):
    ''' centers V so the baseline is at 0 '''
    Flow = filters.gaussian_filter(V.T, [0.,sigma])
    Flow = filters.minimum_filter1d(Flow, window)
    Flow = filters.maximum_filter1d(Flow, window)
    V_centered = (V.T - Flow).T
    #V_centered = (V.T - Flow.mean(axis=1)[:,np.newaxis]).T
    return V_centered


def get_cluster_timepoints_list(X, n_clusters='none', plot=0):
    ''' this does k-means clustering on X; n_clusters currently can be user-inputted '''
    ''' centers (output) is the time of the 'centroid' image '''
    
    cluster_times = []
    
    if isinstance(n_clusters,str):
        plt.scatter(X[:,0], X[:,1], marker='.', s=20, lw=0, alpha=0.5)
        plt.show()
        n_clusters = input("How many clusters? Enter an integer: ")
        n_clusters = int(n_clusters)
        plt.close()
    
    clusterer = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)
    cluster_centers = clusterer.cluster_centers_
    for i in range(n_clusters):
        cluster_times.append(np.where(cluster_labels == i)[0])

    centers = np.zeros(len(cluster_centers))
    idx=0
    for x,y in cluster_centers:
        centers[idx] = np.argmin(np.sqrt((x-X[:,0])**2 + (y-X[:,1])**2))
        centers[idx] = centers[idx]
        idx+=1
    centers= centers.astype(int)
    
    if plot:
        plt.figure(figsize=(8,8))
        colors = matplotlib.cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        plt.scatter(X[:,0], X[:,1], marker='.', s=20, lw=0, alpha=0.5,
                        c=colors, edgecolor='k')
            
        # Labeling the clusters
        # Draw white circles at cluster centers
        plt.scatter(X[centers,0], X[centers,1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')
        
        for i in range(len(centers)):
            plt.scatter(X[centers[i],0], X[centers[i],1], marker='$%d$'%i,
                        alpha=1, s=50, edgecolor='k')
        
        plt.title('k-means clustering (n={})'.format(n_clusters))
        plt.show()

    return cluster_times, cluster_labels, centers


def z_score_im(im,Ly,Lx,return_im=1):
    ''' return im refers to returning the image, rather than the flattened version'''
    if len(im.shape) == 2:
        im = np.reshape(im,(Ly*Lx)) #flatten image
    im = scipy.stats.zscore(im)
    if return_im:
        im = np.reshape(im,(Ly,Lx))

    return im


def z_score_U(U,Ly,Lx,return_im=0):
    if len(U.shape) == 3:
        U = np.reshape(U,(Ly*Lx,-1)) #flatten to 2d
    U = scipy.stats.zscore(U, axis=0)
    if return_im:
        U = np.reshape(U,(Ly,Lx,-1))

    return U

