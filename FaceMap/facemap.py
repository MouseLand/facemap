import pims
import numpy as np
from FaceMap import gui, utils, pupil
import time
import os

def run(filenames, parent=None):
    print('processing videos')
    # grab files
    if parent is not None:
        video = parent.video
        cumframes = parent.cumframes
        nframes = parent.nframes
        sbin = parent.sbin
        frame_shape = video[0].frame_shape
        rois = parent.ROIs
        for r in rois:
            if r.rind==2:
                r.yrange_bin = np.arange(np.floor(r.yrange[0]/sbin), np.floor(r.yrange[-1]/sbin)).astype(int)
                r.xrange_bin = np.arange(np.floor(r.xrange[0]/sbin), np.floor(r.xrange[-1]/sbin)).astype(int)
    else:
        video=[]
        nframes = 0
        iframes = []
        cumframes = [0]
        for file in filenames:
            video.append(pims.Video(file))
            nframes += len(video[-1])
            iframes.append(len(video[-1]))
            cumframes.append(cumframes[-1] + len(v[-1]))
        iframes = np.array(iframes).astype(int)
        cumframes = np.array(cumframes).astype(int)
        frame_shape = video[0].frame_shape
        sbin = 4
        rois = None

    Ly = frame_shape[0]
    Lx = frame_shape[1]
    isRGB = False
    if len(frame_shape) > 2:
        isRGB = True

    ncomps = 500
    tic = time.time()
    # compute average frame and average motion across videos (binned by sbin)
    avgframe, avgmotion = subsampled_mean(video, cumframes, sbin)
    print('computed subsampled mean at %1.2fs'%(time.time() - tic))

    # compute SVD from frames subsampled across videos and return spatial components
    U = compute_SVD(video, cumframes, avgmotion, ncomps, sbin, rois)
    print('computed subsampled SVD at %1.2fs'%(time.time() - tic))

    # project U onto all movie frames
    # and compute pupil (if selected)
    V, pup = process_ROIs(video, cumframes, avgmotion, U, sbin, tic, rois)

    print('computed projection at %1.2fs'%(time.time() - tic))

    # reshape components
    Lyb = int(np.floor(Ly / sbin))
    Lxb = int(np.floor(Lx / sbin))
    for nr in range(len(U)):
        U[nr] = np.reshape(U[nr], (Lyb, Lxb, -1))
    avgframe  = np.reshape(avgframe, (Lyb, Lxb))
    avgmotion = np.reshape(avgmotion, (Lyb, Lxb))

    # save output to file (can load in gui)
    save_ROIs(filenames, sbin, U, V, pup, avgframe, avgmotion, rois)

    return V, pup

def save_ROIs(filenames, sbin, U, V, pup, avgframe, avgmotion, rois=None):
    proc = {'motMask': U, 'motSVD': V, 'pupil': pup,
            'avgframe': avgframe, 'avgmotion': avgmotion,
            'filenames': filenames}
    basename, filename = os.path.split(filenames[0])
    filename, ext = os.path.splitext(filename)
    savename = os.path.join(basename, ("%s_proc.npy"%filename))
    print(savename)
    np.save(savename, proc)

def get_frames(video, cframes, cumframes):
    nframes = cumframes[-1]
    cframes = np.maximum(0, np.minimum(nframes-1, cframes))
    cframes = np.arange(cframes[0], cframes[-1]+1).astype(int)
    ivids = (cframes[np.newaxis,:] >= cumframes[1:,np.newaxis]).sum(axis=0)
    for n in np.unique(ivids):
        im = np.array(video[n][cframes[ivids==n] - cumframes[n]])
        if np.ndim(im)<4:
            im = im[np.newaxis, :, :, :]
        if n==np.unique(ivids)[0]:
            img = im
        else:
            img = np.concatenate((img, im), axis=0)
    if img.ndim > 3:
        img = img[:,:,:,0]
    img = np.transpose(img, (1,2,0)).astype(np.float32)
    return img

def subsampled_mean(video, cumframes, sbin=3):
    # grab up to 2000 frames to average over for mean
    # v is a list of videos loaded with pims
    # cumframes are the cumulative frames across videos
    frame_shape = video[0].frame_shape
    Ly = frame_shape[0]
    Lx = frame_shape[1]
    nframes = cumframes[-1]
    nf = min(2000, nframes)
    # load in chunks of up to 200 frames (for speed)
    nt0 = min(200, np.diff(cumframes).min())
    nsegs = int(np.floor(nf / nt0))
    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)

    Lyb = int(np.floor(Ly / sbin))
    Lxb = int(np.floor(Lx / sbin))
    avgframe = np.zeros((Lyb * Lxb), np.float32)
    avgmotion = np.zeros((Lyb * Lxb), np.float32)
    ns = 0
    for n in range(nsegs):
        t = tf[n]
        im = get_frames(video, np.arange(t,t+nt0), cumframes)
        # bin
        imbin = spatial_bin(im, sbin, Lyb, Lxb)
        # add to averages
        avgframe += imbin.mean(axis=-1)
        imbin = np.abs(np.diff(imbin, axis=-1))
        avgmotion += imbin.mean(axis=-1)
        ns+=1

    avgframe /= float(ns)
    avgmotion /= float(ns)

    return avgframe, avgmotion

def compute_SVD(video, cumframes, avgmotion, ncomps=500, sbin=3, rois=None):
    # compute the SVD over frames in chunks, combine the chunks and take a mega-SVD
    # number of components kept from SVD is ncomps
    # the pixels are binned in spatial bins of size sbin
    # v is a list of videos loaded with pims
    # cumframes are the cumulative frames across videos
    frame_shape = video[0].frame_shape
    Ly = frame_shape[0]
    Lx = frame_shape[1]

    sbin = max(1, sbin)
    nframes = cumframes[-1]

    # load in chunks of up to 1000 frames (for speed)
    nt0 = min(1000, nframes)
    nsegs = int(min(np.floor(25000 / nt0), np.floor(nframes / nt0)))
    nc = int(250) # <- how many PCs to keep in each chunk
    nc = min(nc, nt0-1)
    if nsegs==1:
        nc = min(ncomps, nt0-1)
    # what times to sample
    tf = np.floor(np.linspace(0, nframes-nt0-1, nsegs)).astype(int)

    Lyb = int(np.floor(Ly / sbin))
    Lxb = int(np.floor(Lx / sbin))
    U = [np.zeros((Lyb*Lxb, nsegs*nc), np.float32)]
    nroi = 0
    mind = []
    if rois is not None:
        for i,r in enumerate(rois):
            if r.rind==2:
                nroi += 1
                mind.append(i)
                nyb = r.yrange_bin.size
                nxb = r.xrange_bin.size
                U.append(np.zeros((nyb*nxb, nsegs*nc), np.float32))
    ns = 0
    for n in range(nsegs):
        tic=time.time()
        t = tf[n]
        im = get_frames(video, np.arange(t,t+nt0), cumframes)
        # bin
        imbin = spatial_bin(im, sbin, Lyb, Lxb)
        # compute motion energy
        imbin = np.abs(np.diff(imbin, axis=-1))
        imbin -= avgmotion[:,np.newaxis]
        usv  = utils.svdecon(imbin, k=nc)
        U[0][:, n*nc:(n+1)*nc] = usv[0]
        if nroi > 0:
            imbin = np.reshape((Lyb, Lxb, -1))
            for i,m in enumerate(mind):
                lilbin = imbin[np.ix_(rois[m].yrange_bin, rois[m].xrange_bin)]
                lilbin = np.reshape(lilbin, (-1, lilbin.shape[-1]))
                usv  = utils.svdecon(lilbin, k=nc)
                U[i][:, n*nc:(n+1)*nc] = usv[0]

        ns+=1
    for nr in range(len(U)):
        U[nr] = U[nr][:, :(ns+1)*nc]

    # take SVD of concatenated spatial PCs
    if ns > 1:
        for nr in range(len(U)):
            usv = utils.svdecon(U[nr], k = min(ncomps, U[nr].shape[1]-1))
            U[nr] = usv[0]

    return U

def spatial_bin(im, sbin, Lyb, Lxb):
    imbin = im.astype(np.float32)
    if sbin > 1:
        imbin = (np.reshape(im[:Lyb*sbin, :Lxb*sbin, :], (Lyb,sbin,Lxb,sbin,-1))).mean(axis=1).mean(axis=2)
    imbin = np.reshape(imbin, (Lyb*Lxb, -1))
    return imbin

def process_ROIs(video, cumframes, avgmotion, U, sbin=3, tic=None, rois=None):
    # project U onto each frame in the video and compute the motion energy
    # also compute pupil on single frames on non binned data
    # the pixels are binned in spatial bins of size sbin
    # video is a list of videos loaded with pims
    # cumframes are the cumulative frames across videos
    if tic is None:
        tic=time.time()
    nframes = cumframes[-1]
    ncomps = U[0].shape[1]
    V = [np.zeros((nframes, ncomps), np.float32)]
    pup = []

    mind=[]
    pind=[]
    if rois is not None:
        for i,r in enumerate(rois):
            if r.rind==1:
                pind.append(i)
                pup.append({'area': np.zeros((nframes,)), 'com': np.zeros((nframes,2))})
            if r.rind==2:
                mind.append(i)
                V.append(np.zeros((nframes, ncomps), np.float32))

    # compute in chunks of 2000
    nt0 = 2000
    nsegs = int(np.ceil(nframes / nt0))
    frame_shape = video[0].frame_shape
    Ly = frame_shape[0]
    Lx = frame_shape[1]
    Lyb = int(np.floor(Ly / sbin))
    Lxb = int(np.floor(Lx / sbin))
    for n in range(nsegs):
        t = n*nt0
        im = get_frames(video, np.arange(t,t+nt0), cumframes)

        # compute pupil
        if len(pind)>0:
            k=0
            for p in pind:
                com, area = pupil.process(im[np.ix_(rois[p].yrange, rois[p].xrange)],
                              rois[p].saturation, rois[p].pupil_sigma)
                pup[k]['com'][t:t+nt0,:] = com
                pup[k]['area'][t:t+nt0] = area
                k+=1
        # bin
        imbin = spatial_bin(im, sbin, Lyb, Lxb)
        if n>0:
            imbin = np.concatenate((imend[:,np.newaxis], imbin), axis=-1)
        imend = imbin[:,-1]
        # compute motion energy
        imbin = np.abs(np.diff(imbin, axis=-1))
        imbin -= avgmotion[:,np.newaxis]
        imbin = np.reshape(imbin, (Lyb, Lxb, -1))
        # compute svd projections onto all motion SVDs
        for nr in range(len(U)):
            if nr==0:
                vproj = np.reshape(imbin, (Lyb*Lxb, -1)).T @ U[nr]
            else:
                lilbin = imbin[np.ix_(rois[mind[nr-1]].yrange_bin, rois[mind[nr-1]].xrange_bin)]
                lilbin = np.reshape(lilbin, (-1, lilbin.shape[-1]))
                vproj = lilbin.T @ U[nr]
            # first time block will have one less subtracted frame
            if n==0:
                vproj = np.concatenate((vproj[0,:][np.newaxis, :], vproj), axis=0)
            V[nr][t:t+nt0, :] = vproj

        if n%5==0:
            print('segment %d / %d, time %1.2f'%(n+1, nsegs, time.time() - tic))

    return V, pup
