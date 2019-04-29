import numpy as np
from scipy.ndimage import gaussian_filter1d

def fit_gaussian(im, thres, do_xy):
    ''' iterative fitting of pupil with gaussian @ sigma=thres '''
    ix,iy = im.nonzero()
    lam0 = im[ix, iy]
    lam = lam0.copy()
    for k in range(5):
        lam /= lam.sum()
        mu = [(lam*ix).sum(), (lam*iy).sum()]
        xdist = ix - mu[0]
        ydist = iy - mu[1]
        xydist = np.concatenate((xdist[:,np.newaxis], ydist[:,np.newaxis]), axis=1)
        xy = xydist * lam[:,np.newaxis]**0.5
        sigxy = xy.T @ xy
        lam = lam0.copy()
        dd = ((xydist @ np.linalg.inv(sigxy)) * xydist).sum(axis=1)
        lam[dd > 2*thres**2] = 0
    mu = np.array(mu)

    sv, u = np.linalg.eig(thres**2 * sigxy)
    sv = np.real(sv)
    # enforce some circularity on pupil
    # principal axis can only be 2x bigger than minor axis
    min_sv = sv.min()
    sv = min_sv * np.minimum(3, sv/min_sv)

    # compute pts surrounding ellipse
    if do_xy:
        n = 50 # Number of points around ellipse
        p = np.linspace(0, 2*np.pi, n)[:, np.newaxis]
        # Transformation
        xy = np.concatenate((np.cos(p), np.sin(p)),axis=1) * (sv[::-1]**0.5) @ u[:,::-1]
        xy += mu
    else:
        xy = []
    return mu, sv, xy

def process(img, saturation, pupil_sigma):
    ''' get pupil by fitting 2D gaussian
        (only uses pixels darker than saturation) '''

    # smooth in time by two bins
    cumsum = np.cumsum(img, axis=-1)
    img[:,:,1:-1] = (cumsum[:, :, 2:] - cumsum[:, :, :-2]) / float(2)

    # smooth in space by 1 pixel
    img = gaussian_filter1d(gaussian_filter1d(img, 1, axis=0), 1, axis=1)

    img -= img.min(axis=1).min(axis=0)[np.newaxis, np.newaxis, :]
    img = 255.0 - img
    img = np.maximum(0, img - (255.0 - saturation))
    nframes = img.shape[-1]
    com = np.nan*np.zeros((nframes,2))
    area = np.nan*np.zeros((nframes,))
    for n in range(nframes):
        try:
            mu, sig, _ = fit_gaussian(img[:,:,n], pupil_sigma, False)
        except:
            mu = np.nan*np.zeros((2,))
            sig = np.nan*np.zeros((2,))
        com[n,:] = mu
        area[n] = np.pi * (sig[0] * sig[1]) ** 0.5
    return com, area

def smooth(area):
    ''' replace outliers in pupil area with smoothed pupil area'''
    ''' also replace nan's with smoothed pupil area '''
    ''' smoothed pupil area (marea) is a median filter with window 30 '''
    win = 30 # usually recordings are @ >=30Hz
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
    ix2 = (~np.logical_or(np.isnan(area), np.isnan(marea))).nonzero()[0]
    area[ix] = np.interp(ix, ix2, marea[ix2])
    marea[ix] = area[ix]

    # when do large deviations happen
    adiff = np.abs(area - marea)
    thres = area.std()/2
    ireplace = adiff > thres
    area[ireplace] = marea[ireplace]

    return area
