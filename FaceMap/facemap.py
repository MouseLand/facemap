import pims
import numpy as np
from FaceMap import gui, utils, pupil, running
import time
import os
import pdb

def run(filenames, parent=None):
    print('processing videos')
    # grab files
    if parent is not None:
        video = parent.video
        cumframes = parent.cumframes
        nframes = parent.nframes
        sbin = parent.sbin
        rois = parent.ROIs
        nroi = 0
        for r in rois:
            if r.rind==1:
                r.yrange_bin = np.arange(np.floor(r.yrange[0]/sbin), np.floor((r.yrange[-1])/sbin)).astype(int)
                r.xrange_bin = np.arange(np.floor(r.xrange[0]/sbin), np.floor((r.xrange[-1])/sbin)).astype(int)
                nroi+=1
        Ly = parent.Ly
        Lx = parent.Lx
        fullSVD = parent.checkBox.isChecked()
    else:
        video=[]
        nframes = 0
        iframes = []
        cumframes = [0]
        for file in filenames:
            v=[]
            for f in file:
                v.append(pims.Video(f))
            video.append(v)
            nframes += len(video[-1][0])
            iframes.append(len(video[-1][0]))
            cumframes.append(cumframes[-1] + len(video[-1][0]))
        iframes = np.array(iframes).astype(int)
        cumframes = np.array(cumframes).astype(int)
        frame_shape = video[0].frame_shape
        sbin = 4
        rois = None
        nroi = 0
        fullSVD = True

    isRGB = True
    #if len(frame_shape) > 2:
    #    isRGB = True

    tic = time.time()
    # compute average frame and average motion across videos (binned by sbin)
    avgframe, avgmotion = subsampled_mean(video, cumframes, Ly, Lx, sbin)
    print('computed subsampled mean at %1.2fs'%(time.time() - tic))

    ncomps = 500

    if fullSVD or nroi>0:
        # compute SVD from frames subsampled across videos and return spatial components
        U = compute_SVD(video, cumframes, Ly, Lx, avgmotion, ncomps, sbin, rois, fullSVD)
        print('computed subsampled SVD at %1.2fs'%(time.time() - tic))
    else:
        U = []

    # project U onto all movie frames
    # and compute pupil (if selected)
    V, pup, run = process_ROIs(video, cumframes, Ly, Lx, avgmotion, U, sbin, tic, rois, fullSVD)

    print('computed projection at %1.2fs'%(time.time() - tic))

    # reshape components
    #Lyb = int(np.floor(Ly / sbin))
    #Lxb = int(np.floor(Lx / sbin))
    #for nr in range(len(U)):
    #    U[nr] = np.reshape(U[nr], (Lyb, Lxb, -1))
    #avgframe  = np.reshape(avgframe, (Lyb, Lxb))
    #avgmotion = np.reshape(avgmotion, (Lyb, Lxb))

    # save output to file (can load in gui)
    save_ROIs(filenames, sbin, U, V, pup, run, avgframe, avgmotion, rois)

    return V, pup, run

def save_ROIs(filenames, sbin, U, V, pup, run, avgframe, avgmotion, rois=None):
    proc = {'motMask': U, 'motSVD': V, 'pupil': pup, 'running': run,
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
    imall = []
    for ii in range(len(video[0])):
        for n in np.unique(ivids):
            im = np.array(video[n][ii][cframes[ivids==n] - cumframes[n]])
            if np.ndim(im)<4:
                im = im[np.newaxis, :, :, :]
            if n==np.unique(ivids)[0]:
                img = im
            else:
                img = np.concatenate((img, im), axis=0)
        if img.ndim > 3:
            img = img[:,:,:,0]
        img = np.transpose(img, (1,2,0)).astype(np.float32)
        imall.append(img)
    return imall

def binned_inds(Ly, Lx, sbin):
    Lyb = np.zeros((len(Ly),), np.int32)
    Lxb = np.zeros((len(Ly),), np.int32)
    ir = []
    ix=0
    for n in range(len(Ly)):
        Lyb[n] = int(np.floor(Ly[n] / sbin))
        Lxb[n] = int(np.floor(Lx[n] / sbin))
        ir.append(np.arange(ix, ix + Lyb[n]*Lxb[n], 1, int))
        ix += Lyb[n]*Lxb[n]
    return Lyb, Lxb, ir

def subsampled_mean(video, cumframes, Ly, Lx, sbin=3):
    # grab up to 2000 frames to average over for mean
    # v is a list of videos loaded with pims
    # cumframes are the cumulative frames across videos
    # Ly, Lx are the sizes of the videos
    # sbin is the spatial binning
    nframes = cumframes[-1]
    nf = min(2000, nframes)
    # load in chunks of up to 200 frames (for speed)
    nt0 = min(200, np.diff(cumframes).min())
    nsegs = int(np.floor(nf / nt0))
    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)

    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)

    avgframe  = np.zeros(((Lyb * Lxb).sum(),), np.float32)
    avgmotion = np.zeros(((Lyb * Lxb).sum(),), np.float32)
    ns = 0
    for n in range(nsegs):
        t = tf[n]
        img = get_frames(video, np.arange(t,t+nt0), cumframes)
        # bin
        for n,im in enumerate(img):
            imbin = spatial_bin(im, sbin, Lyb[n], Lxb[n])
            # add to averages
            avgframe[ir[n]] += imbin.mean(axis=-1)
            imbin = np.abs(np.diff(imbin, axis=-1))
            avgmotion[ir[n]] += imbin.mean(axis=-1)
        ns+=1

    avgframe /= float(ns)
    avgmotion /= float(ns)
    avgframe0 = []
    avgmotion0 = []
    for n in range(len(Ly)):
        avgframe0.append(avgframe[ir[n]])
        avgmotion0.append(avgmotion[ir[n]])
    return avgframe0, avgmotion0

def compute_SVD(video, cumframes, Ly, Lx, avgmotion, ncomps=500, sbin=3, rois=None, fullSVD=True):
    # compute the SVD over frames in chunks, combine the chunks and take a mega-SVD
    # number of components kept from SVD is ncomps
    # the pixels are binned in spatial bins of size sbin
    # v is a list of videos loaded with pims
    # cumframes are the cumulative frames across videos
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

    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)

    if fullSVD:
        U = [np.zeros(((Lyb*Lxb).sum(), nsegs*nc), np.float32)]
    else:
        U = [np.zeros((0,1), np.float32)]
    nroi = 0
    motind = []
    ivid=[]
    if rois is not None:
        for i,r in enumerate(rois):
            ivid.append(r.ivid)
            if r.rind==1:
                nroi += 1
                motind.append(i)
                nyb = r.yrange_bin.size
                nxb = r.xrange_bin.size
                U.append(np.zeros((nyb*nxb, nsegs*min(nc,nyb*nxb)), np.float32))
    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind)

    ns = 0
    for n in range(nsegs):
        tic=time.time()
        t = tf[n]
        img = get_frames(video, np.arange(t,t+nt0), cumframes)
        if fullSVD:
            imall = np.zeros(((Lyb*Lxb).sum(), nt0-1), np.float32)
        for ii,im in enumerate(img):
            imbin = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
            # compute motion energy
            imbin = np.abs(np.diff(imbin, axis=-1))
            imbin -= avgmotion[ii][:,np.newaxis]

            if fullSVD:
                imall[ir[ii]] = imbin
            if nroi > 0:
                wmot = (ivid[motind]==ii).nonzero()[0]
                if wmot.size > 0:
                    wroi = motind[wmot]
                    for i in range(wroi.size):
                        lilbin = imbin[np.ix_(rois[wroi[i]].yrange_bin, rois[wroi[i]].xrange_bin, np.arange(0,imbin.shape[-1],1,int))]
                        lilbin = np.reshape(lilbin, (-1, lilbin.shape[-1]))
                        usv  = utils.svdecon(lilbin, k=nc)
                        ncb = min(nc, lilbin.shape[0])
                        U[wmot[i]+1][:, n*ncb:(n+1)*ncb]
        if fullSVD:
            usv  = utils.svdecon(imall, k=nc)
            U[0][:, n*nc:(n+1)*nc] = usv[0]
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

def process_ROIs(video, cumframes, Ly, Lx, avgmotion, U, sbin=3, tic=None, rois=None, fullSVD=True):
    # project U onto each frame in the video and compute the motion energy
    # also compute pupil on single frames on non binned data
    # the pixels are binned in spatial bins of size sbin
    # video is a list of videos loaded with pims
    # cumframes are the cumulative frames across videos
    if tic is None:
        tic=time.time()
    nframes = cumframes[-1]
        
    pup = []
    blink = []
    run = []

    motind=[]
    pupind=[]
    blind=[]
    runind = []
    ivid = []
    run = []
    if rois is not None:
        nroi=0 # number of motion ROIs
        for i,r in enumerate(rois):
            ivid.append(r.ivid)
            if r.rind==0:
                pupind.append(i)
                pup.append({'area': np.zeros((nframes,)), 'com': np.zeros((nframes,2))})
            elif r.rind==1:
                motind.append(i)
                nroi+=1
                V.append(np.zeros((nframes, U[nr].shape[1]), np.float32))
            elif r.rind==2:
                blind.append(i)
                blind.append(np.zeros((nframes,)))
            elif r.rind==3:
                runind.append(i)
                run.append(np.zeros((nframes,2)))
    ivid = np.array(ivid).astype(np.int32)

    if nroi>0:
        ncomps = U[1].shape[1]
    elif fullSVD:
        ncomps = U[0].shape[1]
    if fullSVD:
        V = [np.zeros((nframes, ncomps), np.float32)]
    else:
        V = [np.zeros((0,1), np.float32)]
        
    
    # compute in chunks of 2000
    nt0 = 2000
    nsegs = int(np.ceil(nframes / nt0))
    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    imend = []
    for ii in range(len(Ly)):
        imend.append([])
    for n in range(nsegs):
        t = n*nt0
        img = get_frames(video, np.arange(t,t+nt0), cumframes)

        # compute pupil
        if len(pupind)>0:
            k=0
            for p in pupind:
                com, area = pupil.process(img[ivid[p]][np.ix_(rois[p].yrange, rois[p].xrange, np.arange(0,im.shape[-1],1,int))],
                              rois[p].saturation, rois[p].pupil_sigma)
                pup[k]['com'][t:t+nt0,:] = com
                pup[k]['area'][t:t+nt0] = area
                k+=1

        # compute running
        if len(runind)>0:
            k=0
            for r in runind:
                imr = img[ivid[r]][np.ix_(rois[r].yrange, rois[r].xrange,
                                          np.arange(0,img[ivid[r]].shape[-1],1,int))]
                if n>0:
                    imr = np.concatenate((rend[k][:,np.newaxis], imr), axis=-1)
                else:
                    if k==0:
                        rend=[]
                    rend.append(imr)
                imr = np.transpose(imr, (2,0,1)).copy()
                dy, dx = running.process(imr)
                run[k][t:t+nt0,:] = np.concatenate((dy[:,np.newaxis], dx[:,np.newaxis]),axis=1)
                k+=1

        # bin and get motion
        if fullSVD:
            if n>0:
                imall = np.zeros(((Lyb*Lxb).sum(), img[0].shape[-1]), np.float32)
            else:
                imall = np.zeros(((Lyb*Lxb).sum(), img[0].shape[-1]-1), np.float32)
        for ii,im in enumerate(img):
            imbin = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
            if n>0:
                imbin = np.concatenate((imend[ii][:,np.newaxis], imbin), axis=-1)
            imend[ii] = imbin[:,-1]
            # compute motion energy
            imbin = np.abs(np.diff(imbin, axis=-1))
            imbin -= avgmotion[ii][:,np.newaxis]
            imall[ir[ii]] = imbin

            if nroi > 0:
                wmot = (ivid[motind]==ii).nonzero()[0]
                if wmot.size > 0:
                    wroi = motind[wmot]
                    for i in range(wroi.size):
                        lilbin = imbin[np.ix_(rois[wroi[i]].yrange_bin, rois[wroi[i]].xrange_bin, np.arange(0,imbin.shape[-1],1,int))]
                        lilbin = np.reshape(lilbin, (-1, lilbin.shape[-1]))
                        vproj = lilbin.T @ U[wmot[i]+1]
                        if n==0:
                            vproj = np.concatenate((vproj[0,:][np.newaxis, :], vproj), axis=0)
                        V[wmot[i]+1][t:t+nt0, :] = vproj

        if fullSVD:
            vproj = imall.T @ U[0]
            if n==0:
                vproj = np.concatenate((vproj[0,:][np.newaxis, :], vproj), axis=0)
            V[0][t:t+nt0, :] = vproj

        if n%5==0:
            print('segment %d / %d, time %1.2f'%(n+1, nsegs, time.time() - tic))

    return V, pup, run
