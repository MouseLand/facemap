import pims
import numpy as np
from FaceMap import gui, utils
import pyqtgraph as pg
import time
import os

def run(filenames, parent=None):
    print('processing videos')
    # grab files
    if parent is not None:
        video = parent.video
        iframes = parent.iframes
        nframes = parent.nframes
        frame_shape = video[0].frame_shape
    else:
        video=[]
        nframes = 0
        iframes = []
        for file in filenames:
            video.append(pims.Video(file))
            nframes += len(video[-1])
            iframes.append(len(video[-1]))
        iframes = np.array(iframes).astype(int)
        frame_shape = video[0].frame_shape

    Ly = frame_shape[0]
    Lx = frame_shape[1]
    isRGB = False
    if len(frame_shape) > 2:
        isRGB = True

    sbin = 4
    ncomps = 500
    tic = time.time()
    # compute average frame and average motion across videos
    avgframe, avgmotion = subsampled_mean(video, Ly, Lx, iframes, sbin)
    print('computed subsampled mean at %1.2fs'%(time.time() - tic))

    # compute SVD from frames subsampled across videos and return spatial components
    U = compute_SVD(video, Ly, Lx, iframes, avgmotion, ncomps, sbin)
    print('computed subsampled SVD at %1.2fs'%(time.time() - tic))

    # project U onto all movie frames
    V = project_masks(video, Ly, Lx, iframes, avgmotion, U, sbin, tic)
    print('computed projection at %1.2fs'%(time.time() - tic))

    # save output to file (can load in gui)
    save_ROIs(filenames, Ly, Lx, sbin, U, V, avgframe, avgmotion)

    return V

def save_ROIs(filenames, Ly, Lx, sbin, U, V, avgframe, avgmotion):
    Lyb = int(np.floor(Ly / sbin))
    Lxb = int(np.floor(Lx / sbin))
    U = np.reshape(U, (Lyb, Lxb, -1))
    avgframe  = np.reshape(avgframe, (Lyb, Lxb))
    avgmotion = np.reshape(avgmotion, (Lyb, Lxb))
    proc = {'motMask': U, 'motSVD': V,
            'avgframe': avgframe, 'avgmotion': avgmotion,
            'filenames': filenames}
    basename, filename = os.path.split(filenames[0])
    savename = os.path.join(basename, ("%s_proc.npy"%filename))
    print(savename)
    np.save(savename, proc)


def subsampled_mean(video, Ly, Lx, iframes, sbin=3):
    # grab up to 2000 frames to average over for mean
    # v is a list of videos loaded with pims
    # iframes are the frames in each video
    nframes = iframes.sum()
    nvids = len(video)
    nf = min(2000, nframes)
    # load in chunks of up to 200 frames (for speed)
    nt0 = min(200, iframes.min())
    nsegs = int(np.floor(nf / nt0))
    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)
    nfr = np.cumsum(np.concatenate(([0], iframes)))

    Lyb = int(np.floor(Ly / sbin))
    Lxb = int(np.floor(Lx / sbin))
    avgframe = np.zeros((Lyb * Lxb), np.float32)
    avgmotion = np.zeros((Lyb * Lxb), np.float32)
    ns = 0
    for n in range(nsegs):
        t = tf[n]
        # which video is segment "n" in
        ivid = np.logical_and(t < nfr[1:], t >= nfr[:-1]).nonzero()[0][0]
        tvid = t - nfr[ivid] # relative time in video
        if tvid > (iframes[ivid] - nt0):
            ivid += 1
            tvid = 0
            if ivid > nvids - 1:
                break
        im = np.array(video[ivid][t:t+nt0])
        if im.ndim > 3:
            im = im[:,:,:,0]
        im = np.transpose(im, (1,2,0)).astype(np.float32)
        imbin = spatial_bin(im, sbin, Lyb, Lxb)
        # add to averages
        avgframe += imbin.mean(axis=-1)
        imbin = np.abs(np.diff(imbin, axis=-1))
        avgmotion += np.abs(np.diff(imbin, axis=-1)).mean(axis=-1)
        ns+=1

    avgframe /= float(ns)
    avgmotion /= float(ns)

    return avgframe, avgmotion

def compute_SVD(video, Ly, Lx, iframes, avgmotion, ncomps=500, sbin=3):
    # compute the SVD over frames in chunks, combine the chunks and take a mega-SVD
    # number of components kept from SVD is ncomps
    # the pixels are binned in spatial bins of size sbin
    # v is a list of videos loaded with pims
    # iframes are the frames in each video
    sbin = max(1, sbin)
    nframes = iframes.sum()
    nvids = len(video)
    nf = min(2000, nframes)
    # load in chunks of up to 1000 frames (for speed)
    nt0 = min(1000, iframes.min())
    nsegs = int(min(np.floor(25000 / nt0), np.floor(nframes / nt0)))
    nc = int(250) # <- how many PCs to keep in each chunk
    nc = min(nc, nt0-1)
    if nsegs==1:
        nc = min(ncomps, nt0-1)
    # what times to sample
    tf = np.floor(np.linspace(0, nframes-nt0-1, nsegs)).astype(int)
    nfr = np.cumsum(np.concatenate(([0], iframes)))

    ns = 0
    Lyb = int(np.floor(Ly / sbin))
    Lxb = int(np.floor(Lx / sbin))
    U = np.zeros((Lyb*Lxb, nsegs*nc), np.float32)
    imchunk = np.zeros((Ly, Lx, nt0), np.float32)
    for n in range(nsegs):
        t = tf[n]
        # which video is segment "n" in
        ivid = np.logical_and(t < nfr[1:], t >= nfr[:-1]).nonzero()[0][0]
        tvid = t - nfr[ivid] # relative time in video
        if tvid > (iframes[ivid] - nt0):
            ivid += 1
            tvid = 0
            if ivid >= nvids - 1:
                break

        tic=time.time()
        im = np.array(video[ivid][t:t+nt0])
        if im.ndim > 3:
            im = im[:,:,:,0]
        # compute motion energy
        im = np.transpose(im, (1,2,0)).astype(np.float32)
        imbin = spatial_bin(im, sbin, Lyb, Lxb)
        imbin = np.abs(np.diff(imbin, axis=-1))
        imbin -= avgmotion[:,np.newaxis]
        usv  = utils.svdecon(imbin, k=nc)
        U[:, n*nc:(n+1)*nc] = usv[0]
        ns+=1
    U = U[:, :(ns+1)*nc]

    # take SVD of concatenated spatial PCs
    if ns > 1:
        usv = utils.svdecon(U, k = min(ncomps, U.shape[1]-1))
        u = usv[0]
    else:
        u = U
    return u

def spatial_bin(im, sbin, Lyb, Lxb):
    imbin = im.astype(np.float32)
    if sbin > 1:
        imbin = (np.reshape(im[:Lyb*sbin, :Lxb*sbin, :], (Lyb,sbin,Lxb,sbin,-1))).mean(axis=1).mean(axis=2)
    imbin = np.reshape(imbin, (Lyb*Lxb, -1))
    return imbin

def project_masks(video, Ly, Lx, iframes, avgmotion, U, sbin=3, tic=None):
    # project U onto each frame in the video and compute the motion energy
    nframes = iframes.sum()
    ncomps = U.shape[1]
    V = np.zeros((nframes, ncomps), np.float32)
    nvids = len(video)
    nfr = np.cumsum(np.concatenate(([0], iframes)))

    if tic is None:
        tic=time.time()

    # loop over videos in list
    # compute in chunks of 2000
    nt0 = 500
    Lyb = int(np.floor(Ly / sbin))
    Lxb = int(np.floor(Lx / sbin))
    for ivid in range(nvids):
        ifr   = nfr[ivid]
        nsegs = int(np.ceil(iframes[ivid] / nt0))
        tvid  = np.floor(np.linspace(0, iframes[ivid], nsegs+1)).astype(int)
        for n in range(nsegs):
            im = np.array(video[ivid][tvid[n]:tvid[n+1]])
            if im.ndim > 3:
                im = im[:,:,:,0]
            # compute motion energy
            im = np.transpose(im, (1,2,0)).astype(np.float32)
            imbin = spatial_bin(im, sbin, Lyb, Lxb)
            if n>0:
                imbin = np.concatenate((imend[:,np.newaxis], imbin), axis=-1)
            imend = imbin[:,-1]
            imbin = np.abs(np.diff(imbin, axis=-1))
            imbin -= avgmotion[:,np.newaxis]

            vproj = imbin.T @ U
            if n==0:
                vproj = np.concatenate((vproj[0,:][np.newaxis, :], vproj), axis=0)
            V[(ifr+tvid[n]) : (ifr+tvid[n+1]), :] = vproj

            if n%5==0:
                print('video %d, segment %d / %d, time %1.2f'%(ivid+1, n+1, nsegs, time.time() - tic))


                # parent.p1.clear()
                # parent.p2.clear()
                # for c in range(min(10,ncomps)):
                #     parent.p1.plot(V[:, c] * sign(skew(V[:, c])),  pen=(255-20*c, 0, 0+20*c))
                #     parent.p2.plot(zscore(V[:, c]) * sign(skew(V[:, c])),  pen=(255-20*c, 0, 0+20*c))
                #
                # parent.p1.setRange(xRange=(0,nframes),
                #                   padding=0.0)
                # parent.p1.setLimits(xMin=0,xMax=nframes)
                # parent.cframe = ifr+tvid[n+1] - 1
                # parent.scatter1 = pg.ScatterPlotItem()
                # parent.p1.addItem(parent.scatter1)
                # parent.scatter1.setData([parent.cframe, parent.cframe],
                #                       [V[parent.cframe, 0], V[parent.cframe, 1]],
                #                       size=10,brush=pg.mkBrush(255,0,0))
                #
                # parent.p2.setRange(xRange=(0,nframes),
                #                   padding=0.0)
                # parent.p2.setLimits(xMin=0,xMax=nframes)
                # parent.cframe = ifr+tvid[n+1] - 1
                # parent.scatter2 = pg.ScatterPlotItem()
                # parent.p2.addItem(parent.scatter1)
                # parent.scatter2.setData([parent.cframe, parent.cframe],
                #                       [V[parent.cframe, 0], V[parent.cframe, 1]],
                #                       size=10,brush=pg.mkBrush(255,0,0))
                #
                #
                # parent.jump_to_frame()
                # parent.win.show()
                # parent.show()
    return V
