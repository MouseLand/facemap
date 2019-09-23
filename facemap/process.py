import pims
import numpy as np
from facemap import utils, pupil, running
from numba import vectorize,uint8,float32
import time
import os
import pdb
from scipy import io
from scipy.ndimage import gaussian_filter
import av


def get_frames_pims(imall, video, cframes, cumframes):
    nframes = cumframes[-1]
    cframes = np.maximum(0, np.minimum(nframes-1, cframes))
    cframes = np.arange(cframes[0], cframes[-1]+1).astype(int)
    ivids = (cframes[np.newaxis,:] >= cumframes[1:,np.newaxis]).sum(axis=0)
    for ii in range(len(video[0])):
        nk = 0
        for n in np.unique(ivids):
            cfr = cframes[ivids==n]
            nk += int(cfr[-1] - cfr[0] + 1)
            imall[ii] = np.array(video[n][ii][cfr[0]-cumframes[n]:cfr[-1]-cumframes[n]+1])[:,:,:,0].transpose((1,2,0))
    if nk < imall[0].shape[-1]:
        for ii,im in enumerate(imall):
            imall[ii] = im[:,:,:nk].copy()

def get_frames(imall, containers, cframes, cumframes):
    cframes = np.arange(cframes[0], min(cframes[-1], cumframes[-1])).astype(int)
    ivids = (cframes[np.newaxis,:] >= cumframes[1:,np.newaxis]).sum(axis=0)
    for ii in range(len(containers[0])):
        nk = 0
        for n in np.unique(ivids):
            cfr = cframes[ivids==n]
            k=0
            for frame in containers[n][ii].decode(video=0):
                imall[ii][:,:,nk] = frame.to_ndarray(format='rgb24')[:,:,0]
                k+=1
                nk+=1
                if k==cfr[-1]-cfr[0]+1:
                    break
    if nk < imall[0].shape[-1]:
        for ii,im in enumerate(imall):
            imall[ii] = im[:,:,:nk].copy()

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

@vectorize(['float32(uint8)'], nopython=True, target = 'parallel')
def ftype(x):
    return np.float32(x)

def spatial_bin(im, sbin, Lyb, Lxb):
    imbin = im.astype(np.float32)
    if sbin > 1:
        imbin = (np.reshape(im[:Lyb*sbin, :Lxb*sbin], (Lyb,sbin,Lxb,sbin,-1))).mean(axis=1).mean(axis=2)
    imbin = np.reshape(imbin, (Lyb*Lxb, -1))
    return imbin

def imall_init(nfr, Ly, Lx):
    imall = []
    for n in range(len(Ly)):
        imall.append(np.zeros((Ly[n],Lx[n],nfr), 'uint8'))
    return imall

def subsampled_mean(containers, cumframes, Ly, Lx, sbin=3):
    # grab up to 2000 frames to average over for mean
    # v is a list of containerss loaded with av
    # cumframes are the cumulative frames across videos
    # Ly, Lx are the sizes of the videos
    # sbin is the spatial binning
    nframes = cumframes[-1]
    nf = min(1000, nframes)
    # load in chunks of up to 100 frames (for speed)
    nt0 = min(100, np.diff(cumframes).min())
    nsegs = int(np.floor(nf / nt0))
    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)

    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    imall = imall_init(nt0, Ly, Lx)

    avgframe  = np.zeros(((Lyb * Lxb).sum(),), np.float32)
    avgmotion = np.zeros(((Lyb * Lxb).sum(),), np.float32)
    ns = 0
    for n in range(nsegs):
        t = tf[n]
        get_frames_pims(imall, containers, np.arange(t,t+nt0), cumframes)
        # bin
        for n,im in enumerate(imall):
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

def compute_SVD(containers, cumframes, Ly, Lx, avgmotion, ncomps=500, sbin=3, rois=None, fullSVD=True):
    # compute the SVD over frames in chunks, combine the chunks and take a mega-SVD
    # number of components kept from SVD is ncomps
    # the pixels are binned in spatial bins of size sbin
    # v is a list of containerss loaded with av
    # cumframes are the cumulative frames across videos
    sbin = max(1, sbin)
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

    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    img = imall_init(nt0, Ly, Lx)

    if fullSVD:
        U = [np.zeros(((Lyb*Lxb).sum(), nsegs*nc), np.float32)]
    else:
        U = [np.zeros((0,1), np.float32)]
    nroi = 0
    motind = []
    ivid=[]
    ni = []
    ni.append(0)
    if rois is not None:
        for i,r in enumerate(rois):
            ivid.append(r['ivid'])
            if r['rind']==1:
                nroi += 1
                motind.append(i)
                nyb = r['yrange_bin'].size
                nxb = r['xrange_bin'].size
                U.append(np.zeros((nyb*nxb, nsegs*min(nc,nyb*nxb)), np.float32))
                ni.append(0)
    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind)

    ns = 0
    for n in range(nsegs):
        tic=time.time()
        t = tf[n]
        get_frames_pims(img, containers, np.arange(t,t+nt0), cumframes)
        if fullSVD:
            imall = np.zeros(((Lyb*Lxb).sum(), nt0-1), np.float32)
        for ii,im in enumerate(img):
            usevid=False
            if fullSVD:
                usevid=True
            if nroi>0:
                wmot = (ivid[motind]==ii).nonzero()[0]
                if wmot.size>0:
                    usevid=True
            if usevid:
                imbin = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                # compute motion energy
                imbin = np.abs(np.diff(imbin, axis=-1))
                imbin -= avgmotion[ii][:,np.newaxis]
                if fullSVD:
                    imall[ir[ii]] = imbin
                if nroi>0 and wmot.size>0:
                    imbin = np.reshape(imbin, (Lyb[ii], Lxb[ii], -1))
                    wmot=np.array(wmot).astype(int)
                    wroi = motind[wmot]
                    for i in range(wroi.size):
                        lilbin = imbin[rois[wroi[i]]['yrange_bin'][0]:rois[wroi[i]]['yrange_bin'][-1]+1,
                                       rois[wroi[i]]['xrange_bin'][0]:rois[wroi[i]]['xrange_bin'][-1]+1]
                        lilbin = np.reshape(lilbin, (-1, lilbin.shape[-1]))
                        ncb = min(nc, lilbin.shape[0])
                        usv  = utils.svdecon(lilbin, k=ncb)
                        ncb = usv[0].shape[-1]
                        U[wmot[i]+1][:, ni[wmot[i]+1]:ni[wmot[i]+1]+ncb] = usv[0]
                        ni[wmot[i]+1] += ncb
        if n%5==0:
            print('SVD %d/%d chunks'%(n,nsegs))
        if fullSVD:
            ncb = min(nc, imall.shape[0])
            usv  = utils.svdecon(imall, k=ncb)
            ncb = usv[0].shape[-1]
            U[0][:, ni[0]:ni[0]+ncb] = usv[0]
            ni[0] += ncb
        ns+=1

    # take SVD of concatenated spatial PCs
    if ns > 1:
        for nr in range(len(U)):
            if nr==0 and fullSVD:
                U[nr] = U[nr][:, :ni[0]]
                usv = utils.svdecon(U[nr], k = min(ncomps, U[nr].shape[1]-1))
                U[nr] = usv[0]
            elif nr>0:
                U[nr] = U[nr][:, :ni[nr]]
                usv = utils.svdecon(U[nr], k = min(ncomps, U[nr].shape[1]-1))
                U[nr] = usv[0]
    return U

def process_ROIs(containers, cumframes, Ly, Lx, avgmotion, U, sbin=3, tic=None, rois=None, fullSVD=True):
    # project U onto each frame in the video and compute the motion energy
    # also compute pupil on single frames on non binned data
    # the pixels are binned in spatial bins of size sbin
    # containers is a list of videos loaded with av
    # cumframes are the cumulative frames across videos
    if tic is None:
        tic=time.time()
    nframes = cumframes[-1]

    pups = []
    pupreflector = []
    blinks = []
    runs = []

    motind=[]
    pupind=[]
    blind=[]
    runind = []
    ivid = []
    nroi=0 # number of motion ROIs

    if fullSVD:
        ncomps = U[0].shape[-1]
        V = [np.zeros((nframes, ncomps), np.float32)]
        M = [np.zeros((nframes), np.float32)]
    else:
        V = [np.zeros((0,1), np.float32)]
        M = [np.zeros((0,), np.float32)]
    if rois is not None:
        for i,r in enumerate(rois):
            ivid.append(r['ivid'])
            if r['rind']==0:
                pupind.append(i)
                pups.append({'area': np.zeros((nframes,)), 'com': np.zeros((nframes,2)),
                             'axdir': np.zeros((nframes,2,2)), 'axlen': np.zeros((nframes,2))})
                if 'reflector' in r:
                    pupreflector.append(utils.get_reflector(r['yrange'], r['xrange'], rROI=None, rdict=r['reflector']))
                else:
                    pupreflector.append(np.array([]))

            elif r['rind']==1:
                motind.append(i)
                nroi+=1
                V.append(np.zeros((nframes, U[nroi].shape[1]), np.float32))
                M.append(np.zeros((nframes,), np.float32))

            elif r['rind']==2:
                blind.append(i)
                blinks.append(np.zeros((nframes,)))
            elif r['rind']==3:
                runind.append(i)
                runs.append(np.zeros((nframes,2)))
    ivid = np.array(ivid).astype(np.int32)
    motind = np.array(motind).astype(np.int32)

    # compute in chunks of 500
    nt0 = 500
    nsegs = int(np.ceil(nframes / nt0))
    img = imall_init(nt0, Ly, Lx)
    # binned Ly and Lx and their relative inds in concatenated movies
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    imend = []
    for ii in range(len(Ly)):
        imend.append([])
    t=0
    nt1=0
    for n in range(nsegs):
        t += nt1
        get_frames(img, containers, [t, min(cumframes[-1],t+nt0)], cumframes)
        nt1 = img[0].shape[-1]
        # compute pupil
        if len(pupind)>0:
            k=0
            for p in pupind:
                imgp = img[ivid[p]][rois[p]['yrange'][0]:rois[p]['yrange'][-1]+1,
                                    rois[p]['xrange'][0]:rois[p]['xrange'][-1]+1]
                #imgp = gaussian_filter(imgp.astype(np.float32), [1,1], axis=(0,1))
                #imgp = gaussian_filter1d(imgp, 1, axis=1)
                imgp[~rois[p]['ellipse']] = 255
                com, area, axdir, axlen = pupil.process(imgp.astype(np.float32), rois[p]['saturation'],
                                                        rois[p]['pupil_sigma'], pupreflector[k])
                pups[k]['com'][t:t+nt1,:] = com
                pups[k]['area'][t:t+nt1] = area
                pups[k]['axdir'][t:t+nt1,:,:] = axdir
                pups[k]['axlen'][t:t+nt1,:] = axlen
                #print(area)
                k+=1

        if len(blind)>0:
            k=0
            for b in blind:
                imgp = img[ivid[b]][rois[b]['yrange'][0]:rois[b]['yrange'][-1]+1,
                                    rois[b]['xrange'][0]:rois[b]['xrange'][-1]+1]
                imgp[~rois[b]['ellipse']] = 255.0
                bl = np.maximum(0, (255 - imgp - (255-rois[b]['saturation']))).sum(axis=(0,1))
                blinks[k][t:t+nt0] = bl
                k+=1

        # compute running
        if len(runind)>0:
            k=0
            for r in runind:
                imr = img[ivid[r]][rois[r]['yrange'][0]:rois[r]['yrange'][-1]+1,
                                   rois[r]['xrange'][0]:rois[r]['xrange'][-1]+1]
                imr = np.transpose(imr, (2,0,1)).copy()
                # append last frame from previous set
                if n>0:
                    imr = np.concatenate((rend[k][np.newaxis,:,:],imr), axis=0)
                # save last frame
                if k==0:
                    rend=[]
                rend.append(imr[-1].copy())

                # compute phase correaltion between consecutive frames
                dy, dx = running.process(imr)
                if n>0:
                    runs[k][t:t+nt1] = np.concatenate((dy[:,np.newaxis], dx[:,np.newaxis]),axis=1)
                else:
                    runs[k][t+1:t+nt1] = np.concatenate((dy[:,np.newaxis], dx[:,np.newaxis]),axis=1)
                k+=1

        # bin and get motion
        if fullSVD:
            if n>0:
                imall = np.zeros(((Lyb*Lxb).sum(), img[0].shape[-1]), np.float32)
            else:
                imall = np.zeros(((Lyb*Lxb).sum(), img[0].shape[-1]-1), np.float32)
        if fullSVD or nroi > 0:
            for ii,im in enumerate(img):
                usevid=False
                if fullSVD:
                    usevid=True
                if nroi>0:
                    wmot = (ivid[motind]==ii).nonzero()[0]
                    if wmot.size>0:
                        usevid=True
                if usevid:
                    imbin = spatial_bin(im, sbin, Lyb[ii], Lxb[ii])
                    if n>0:
                        imbin = np.concatenate((imend[ii][:,np.newaxis], imbin), axis=-1)
                    imend[ii] = imbin[:,-1]
                    # compute motion energy
                    imbin = np.abs(np.diff(imbin, axis=-1))
                    if fullSVD:
                        M[t:t+imbin.shape[-1]] += imbin.sum(axis=(0,1))
                        imall[ir[ii]] = imbin - avgmotion[ii][:,np.newaxis]
                if nroi > 0 and wmot.size>0:
                    wmot=np.array(wmot).astype(int)
                    imbin = np.reshape(imbin, (Lyb[ii], Lxb[ii], -1))
                    avgmotion[ii] = np.reshape(avgmotion[ii], (Lyb[ii], Lxb[ii]))
                    wroi = motind[wmot]
                    for i in range(wroi.size):
                        lilbin = imbin[rois[wroi[i]]['yrange_bin'][0]:rois[wroi[i]]['yrange_bin'][-1]+1,
                                       rois[wroi[i]]['xrange_bin'][0]:rois[wroi[i]]['xrange_bin'][-1]+1]
                        M[wmot[i]+1][t:t+lilbin.shape[-1]] = lilbin.sum(axis=(0,1))
                        lilbin -= avgmotion[ii][rois[wroi[i]]['yrange_bin'][0]:rois[wroi[i]]['yrange_bin'][-1]+1,
                                       rois[wroi[i]]['xrange_bin'][0]:rois[wroi[i]]['xrange_bin'][-1]+1][...,np.newaxis]
                        lilbin = np.reshape(lilbin, (-1, lilbin.shape[-1]))
                        vproj = lilbin.T @ U[wmot[i]+1]
                        if n==0:
                            vproj = np.concatenate((vproj[0,:][np.newaxis, :], vproj), axis=0)
                        V[wmot[i]+1][t:t+vproj.shape[0], :] = vproj
            if fullSVD:
                vproj = imall.T @ U[0]
                if n==0:
                    vproj = np.concatenate((vproj[0,:][np.newaxis, :], vproj), axis=0)
                V[0][t:t+vproj.shape[0], :] = vproj

        if n%20==0:
            print('segment %d / %d, time %1.2f'%(n+1, nsegs, time.time() - tic))

    return V, M, pups, blinks, runs

def save(proc, savepath=None):
    # save ROIs and traces
    basename, filename = os.path.split(proc['filenames'][0][0])
    filename, ext = os.path.splitext(filename)
    if savepath is not None:
        basename = savepath
    savename = os.path.join(basename, ("%s_proc.npy"%filename))
    print(savename)
    np.save(savename, proc)
    if proc['save_mat']:
        savenamemat = os.path.join(basename, ("%s_proc.mat"%filename))
        print(savenamemat)
        io.savemat(savenamemat, {'proc': proc})
    return savename

def run(filenames, parent=None, proc=None, savepath=None):
    ''' uses filenames and processes fullSVD if no roi's specified '''
    ''' parent is from GUI '''
    ''' proc can be a saved ROI file from GUI '''
    ''' savepath is the folder in which to save _proc.npy '''
    print('processing videos')
    # grab files
    Lys = []
    Lxs = []
    rois=None
    if parent is not None:
        filenames = parent.filenames
        video = parent.video
        containers = []
        for fs in filenames:
            cs = []
            for f in fs:
                cs.append(av.open(f))
                cs[-1].streams.video[0].thread_type = 'AUTO'
            containers.append(cs)
        cumframes = parent.cumframes
        nframes = parent.nframes
        iframes = parent.iframes
        sbin = parent.sbin
        rois = utils.roi_to_dict(parent.ROIs, parent.rROI)
        Ly = parent.Ly
        Lx = parent.Lx
        fullSVD = parent.checkBox.isChecked()
        save_mat = parent.save_mat.isChecked()
        sy = parent.sy
        sx = parent.sx
    else:
        video=[]
        containers = []
        nframes = 0
        iframes = []
        cumframes = [0]
        k=0
        for fs in filenames:
            vs = []
            cs = []
            for f in fs:
                try:
                    vs.append(pims.Video(f))
                except:
                    vs.append(pims.PyAVReaderIndexed(f))
                cs.append(av.open(f))
                cs[-1].streams.video[0].thread_type = 'AUTO'
            video.append(vs)
            containers.append(cs)
            iframes.append(len(video[-1][0]))
            cumframes.append(cumframes[-1] + len(video[-1][0]))
            nframes += len(video[-1][0])
            if k==0:
                Ly = []
                Lx = []
                for vs in video[-1]:
                    fshape = vs.frame_shape
                    Ly.append(fshape[0])
                    Lx.append(fshape[1])
            k+=1
        iframes = np.array(iframes).astype(int)
        cumframes = np.array(cumframes).astype(int)
        if proc is None:
            sbin = 4
            fullSVD = True
            save_mat = False
            rois=None
        else:
            sbin = proc['sbin']
            fullSVD = proc['fullSVD']
            save_mat = proc['save_mat']
            rois = proc['rois']
            sy = proc['sy']
            sx = proc['sx']

    Lybin, Lxbin, iinds = binned_inds(Ly, Lx, sbin)
    LYbin,LXbin,sybin,sxbin = utils.video_placement(Lybin, Lxbin)

    nroi = 0
    if rois is not None:
        for r in rois:
            if r['rind']==1:
                r['yrange_bin'] = np.arange(np.floor(r['yrange'][0]/sbin),
                                            np.floor(r['yrange'][-1]/sbin)).astype(int)
                r['xrange_bin'] = np.arange(np.floor(r['xrange'][0]/sbin),
                                            np.floor(r['xrange'][-1])/sbin).astype(int)
                nroi+=1

    isRGB = True
    #if len(frame_shape) > 2:
    #    isRGB = True

    tic = time.time()
    # compute average frame and average motion across videos (binned by sbin)
    avgframe, avgmotion = subsampled_mean(video, cumframes, Ly, Lx, sbin)
    avgframe_reshape = utils.multivideo_reshape(np.hstack(avgframe)[:,np.newaxis],
                                          LYbin,LXbin,sybin,sxbin,Lybin,Lxbin,iinds)
    avgframe_reshape = np.squeeze(avgframe_reshape)
    avgmotion_reshape = utils.multivideo_reshape(np.hstack(avgmotion)[:,np.newaxis],
                                           LYbin,LXbin,sybin,sxbin,Lybin,Lxbin,iinds)
    avgmotion_reshape = np.squeeze(avgmotion_reshape)
    print('computed subsampled mean at %1.2fs'%(time.time() - tic))

    ncomps = 500
    if fullSVD or nroi>0:
        # compute SVD from frames subsampled across videos and return spatial components
        U = compute_SVD(video, cumframes, Ly, Lx, avgmotion, ncomps, sbin, rois, fullSVD)
        print('computed subsampled SVD at %1.2fs'%(time.time() - tic))
        U_reshape = U.copy()
        if fullSVD:
            U_reshape[0] = utils.multivideo_reshape(U_reshape[0], LYbin,LXbin,sybin,sxbin,Lybin,Lxbin,iinds)
        if nroi>0:
            k=1
            for r in rois:
                if r['rind']==1:
                    ly = r['yrange_bin'].size
                    lx = r['xrange_bin'].size
                    U_reshape[k] = np.reshape(U[k].copy(), (ly,lx,U[k].shape[-1]))
                    k+=1
    else:
        U = []
        U_reshape = []

    # project U onto all movie frames
    # and compute pupil (if selected)
    V, M, pups, blinks, runs = process_ROIs(containers, cumframes, Ly, Lx, avgmotion, U, sbin, tic, rois, fullSVD)

    # smooth pupil and blinks and running
    print('smoothing ...')
    for p in pups:
        if 'area' in p:
            p['area_smooth'],_ = pupil.smooth(p['area'].copy())
            p['com_smooth'] = p['com'].copy()
            p['com_smooth'][:,0],_ = pupil.smooth(p['com_smooth'][:,0].copy())
            p['com_smooth'][:,1],_ = pupil.smooth(p['com_smooth'][:,1].copy())
    for b in blinks:
        b,_ = pupil.smooth(b.copy())
    #for r in runs:
    #    r[:,0],_ = pupil.smooth(r[:,0].copy())
    #    r[:,1],_ = pupil.smooth(r[:,1].copy())

    #V_smooth = []
    #for m in V:
    #    ms,ireplace = pupil.smooth(m[:,0].copy())#, win=50)
    #    ireplace[np.logical_or(ms>ms.std()*4, ms<ms.std()*-4)] = True
    #    print(ireplace.sum())
    #    inds = ireplace.nonzero()[0][:,np.newaxis] + np.arange(-50,51,1,int)[np.newaxis,:]
    #    inds[inds<0] = 0
    #    inds[inds>=m.shape[0]] = m.shape[0]-1
    #    m[ireplace,:] = np.nanmedian(m[inds,:], axis=1)
        #V_smooth.append(m)

    print('computed projection at %1.2fs'%(time.time() - tic))

    proc = {
            'filenames': filenames, 'save_path': savepath, 'iframes': iframes, 'Ly': Ly, 'Lx': Lx,
            'sbin': sbin, 'fullSVD': fullSVD, 'save_mat': save_mat,
            'Lybin': Lybin, 'Lxbin': Lxbin,
            'sybin': sybin, 'sxbin': sxbin, 'LYbin': LYbin, 'LXbin': LXbin,
            'avgframe': avgframe, 'avgmotion': avgmotion,
            'avgframe_reshape': avgframe_reshape, 'avgmotion_reshape': avgmotion_reshape,
            'motion': M,
            'motSVD': V, 'motMask': U, 'motMask_reshape': U_reshape,
            'pupil': pups, 'running': runs, 'blink': blinks, 'rois': rois
            }

    # save processing
    savename = save(proc, savepath)
    return savename
