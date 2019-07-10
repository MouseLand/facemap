import numpy as np
from scipy.ndimage import gaussian_filter

def fit_gaussian(im, thres, do_xy, missing=None):
    ''' iterative fitting of pupil with gaussian @ sigma=thres '''
    ix,iy = im.nonzero()
    if missing is not None and len(missing) > 0:
        mx = missing[0]
        my = missing[1]
        miss = np.isin(ix*im.shape[1] + iy, mx*im.shape[1] + my)
        miss = miss.flatten()
    else:
        miss = np.zeros((ix.size,), np.bool)

    ix = ix[~miss].flatten()
    iy = iy[~miss].flatten()
    lam0 = im[ix,iy].copy()
    immed0 = np.median(lam0)
    lam = lam0.copy()
    ix0 = ix.copy()
    iy0 = iy.copy()
    lam /= lam.sum()
    mu = [(lam*ix).sum(), (lam*iy).sum()]
    xdist = ix - mu[0]
    ydist = iy - mu[1]
    xydist = np.concatenate((xdist[:,np.newaxis], ydist[:,np.newaxis]), axis=1)
    xy = xydist * lam[:,np.newaxis]**0.5
    sigxy = xy.T @ xy
    for k in range(5):
        # fill in NaN's
        dd = ((xydist @ np.linalg.inv(sigxy)) * xydist).sum(axis=1)
        dd=dd.flatten()
        glam = dd[:ix0.size] <= 2*thres**2
        lam0[~glam] = 0
        glam = dd[:ix0.size] <= thres**2
        immed0 = np.median(lam0[glam])
        lam = lam0.copy()
        #print(immed0, mu, sigxy)

        if missing is not None and len(missing) > 0:
            xdist = mx - mu[0]
            ydist = my - mu[1]
            mxy = np.concatenate((xdist[:,np.newaxis], ydist[:,np.newaxis]), axis=1)
            dd = ((mxy @ np.linalg.inv(sigxy)) * mxy).sum(axis=1)
            # within thres?
            #im[mx,my] = np.exp(-dd) * 2 * immed0
            ithr = dd <= 1.15 * thres**2
            im[mx[ithr], my[ithr]] = immed0 * 1 #* np.exp(-dd[ithr]) / np.exp(-dd[ithr]).mean()
            im[mx[~ithr], my[~ithr]] = 0
            ix = np.concatenate((ix0,mx), axis=0)
            iy = np.concatenate((iy0,my), axis=0)
            lamm = np.zeros((mx.size,),np.float32)
            lamm[ithr] = immed0 * 1.1
            #lamm = np.exp(-dd / 2) * immed0
            lam = np.concatenate((lam0, lamm), axis=0)

        lam /= lam.sum()
        mu = [(lam*ix).sum(), (lam*iy).sum()]
        xdist = ix - mu[0]
        ydist = iy - mu[1]
        xydist = np.concatenate((xdist[:,np.newaxis], ydist[:,np.newaxis]), axis=1)
        xy = xydist * lam[:,np.newaxis]**0.5
        sigxy = xy.T @ xy

    mu = np.array(mu)

    sv, u = np.linalg.eig(thres**2 * sigxy)
    sv = np.real(sv)
    # enforce some circularity on pupil
    # principal axis can only be 2x bigger than minor axis
    min_sv = sv.min()
    sv = min_sv * np.minimum(4, sv/min_sv)
    sv = sv[::-1]
    u = u[:,::-1]
    # compute pts surrounding ellipse
    if do_xy:
        n = 100 # Number of points around ellipse
        p = np.linspace(0, 2*np.pi, n)[:, np.newaxis]
        # Transformation
        xy = np.concatenate((np.cos(p), np.sin(p)),axis=1) * (sv**0.5) @ u
        xy += mu
    else:
        xy = []
    if missing is not None and len(missing) > 0:
        imout = im[mx,my]
    else:
        imout = []
    return mu, sv, xy, imout, u, sv

def process(img, saturation, pupil_sigma, pupreflector):
    ''' get pupil by fitting 2D gaussian
        (only uses pixels darker than saturation) '''

    # smooth in time by two bins
    cumsum = np.cumsum(img, axis=-1)
    img[:,:,1:-1] = (cumsum[:, :, 2:] - cumsum[:, :, :-2]) / float(2)

    nframes = img.shape[-1]
    com = np.nan*np.zeros((nframes,2))
    area = np.nan*np.zeros((nframes,))
    axdir = np.nan*np.zeros((nframes,2,2))
    axlen = np.nan*np.zeros((nframes,2))
    for n in range(nframes):
        try:
            # smooth in space by 1 pixel
            imgf = gaussian_filter(img[:,:,n], 1)
            imgf = 255.0 - imgf
            imgf = np.maximum(0, imgf - (255.0 - saturation))
            mu, sig, _, _, u, sv = fit_gaussian(imgf, pupil_sigma, False, missing=pupreflector)
        except:
            mu = np.nan*np.zeros((2,))
            sig = np.nan*np.zeros((2,))
            u   = np.nan*np.zeros((2,2))
            sv   = np.nan*np.zeros((2,))
        com[n,:] = mu
        area[n] = np.pi * (sig[0] * sig[1]) ** 0.5
        axlen[n,:] = sv
        axdir[n,:,:] = u
    return com, area, axdir, axlen

def smooth(area, win=30):
    ''' replace outliers in pupil area with smoothed pupil area'''
    ''' also replace nan's with smoothed pupil area '''
    ''' smoothed pupil area (marea) is a median filter with window 30 '''
    ''' win = 30  usually recordings are @ >=30Hz '''
    nt = area.size
    marea = np.zeros((win, nt))
    winhalf = int(win/2)
    for k in np.arange(-winhalf, winhalf, 1, int):
        if k < 0:
            marea[k+winhalf, :k] = area[-k:]
        elif k > 0:
            marea[k+winhalf, k:] = area[:-k]
        else:
            marea[k+winhalf, :] = area
    marea = np.nanmedian(marea, axis=0)
    ix = np.logical_or(np.isnan(area), np.isnan(marea)).nonzero()[0]
    ix2 = (np.logical_and(~np.isnan(area), ~np.isnan(marea))).nonzero()[0]
    if ix2.size>0:
        area[ix] = np.interp(ix, ix2, marea[ix2])
    else:
        area[ix] = 0
    marea[ix] = area[ix]

    # when do large deviations happen
    adiff = np.abs(area - marea)
    thres = area.std()/2
    ireplace = adiff > thres
    area[ireplace] = marea[ireplace]

    return area, ireplace
