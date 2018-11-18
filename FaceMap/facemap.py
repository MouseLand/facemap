import pims
import numpy as np
from FaceMap import gui

def run(filenames, parent=None):
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

    # compute average frame and average motion across videos
    avgframe, avgmotion = subsampled_mean(video, Ly, Lx, iframes)

    # compute SVD from frames subsampled across videos and return spatial components
    U = compute_SVD(video, Ly, Lx, iframes, avgmotion, ncomps=500, sbin=3)

    # project U onto all movie frames
    V = project_masks(video, Ly, Lx, iframes, avgmotion, U, sbin=3)

def subsampled_mean(video, Ly, Lx, iframes):
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

    avgframe = np.zeros((Ly, Lx), np.float32)
    avgmotion = np.zeros((Ly, Lx), np.float32)
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
        imchunk = im.astype(np.float32)

        # add to averages
        avgframe += imchunk.mean(axis=0)
        avgmotion += np.abs(np.diff(imchunk, axis=0)).mean(axis=0)
        ns+=1

    avgframe /= ns
    avgmotion /= ns

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
    print(nsegs)
    nc = int(ncomps / 2) # <- how many PCs to keep in each chunk
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
    print(U.shape)
    avgmotion = (np.reshape(avgmotion[:Lyb*sbin, :Lxb*sbin], (Lyb,sbin,Lxb,sbin))).mean(axis=-1).mean(axis=1).flatten()
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
        im = np.transpose(im, (1,2,0))
        # compute motion energy
        imbin = (np.reshape(im[:Lyb*sbin, :Lxb*sbin, :], (Lyb,sbin,Lxb,sbin,-1))).mean(axis=1).mean(axis=2)
        imbin = imbin.astype(np.float32)
        imbin = np.reshape(np.abs(np.diff(imbin)), (Lyb*Lxb, -1))
        imbin -= avgmotion[:,np.newaxis]
        usv  = utils.svdecon(imbin, k=nc)
        U[:, n*nc:(n+1)*nc] = usv[0]
        ns+=1
    U = U[:, :(ns+1)*nc]

    # take SVD of concatenated spatial PCs
    if U.shape[1] > ncomps:
        usv = utils.svdecon(U, k = min(ncomps, U.shape[1]))
        u = usv[0]
    else:
        u = U
    return u

def project_masks(video, Ly, Lx, iframes, avgmotion, U, sbin=3):
    # project U onto each frame in the video and compute the motion energy
    nframes = iframes.sum()
    V = np.zeros((U.shape[1], nframes), np.float32)
    nvids = len(video)
    nfr = np.cumsum(np.concatenate(([0], iframes)))

    # loop over videos in list
    # compute in chunks of 2000
    nt0 = 2000
    Lyb = int(np.floor(Ly / sbin))
    Lxb = int(np.floor(Lx / sbin))
    avgmotion = (np.reshape(avgmotion[:Lyb*sbin, :Lxb*sbin], (Lyb,sbin,Lxb,sbin))).mean(axis=-1).mean(axis=1).flatten()
    for ivid in range(nvids):
        ifr   = nfr[ivid]
        nsegs = int(np.ceil(iframes[ivid] / nt0))
        tvid  = np.floor(np.linspace(0, iframes[ivid], nsegs+1)).astype(int)
        for n in range(nsegs):
            im = np.array(v[ivid][tvid[n]:tvid[n+1]])
            if isRGB:
                im = im[:,:,:,0]
            im = np.transpose(im, (1,2,0))
            imbin = (np.reshape(im[:Lyb*sbin, :Lxb*sbin, :], (Lyb,sbin,Lxb,sbin,-1))).mean(axis=1).mean(axis=2)
            imbin = np.reshape(imbin, (Lyb*Lxb,-1)).astype(np.float32)
            if n==0:
                imbin = np.concatenate((imbin[:,0][:,np.newaxis], imbin), axis=-1)
            else:
                imbin = np.concatenate((imend[:,np.newaxis], imbin), axis=-1)
            imbin = np.abs(np.diff(imbin))
            imbin -= avgmotion[:,np.newaxis]

            vproj = U.T @ imbin
            V[:, (ifr+tvid[n]) : (ifr+tvid[n+1])] = vproj
            imend = imbin[:, -1]
    return V
