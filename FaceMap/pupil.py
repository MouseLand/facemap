import numpy as np
from scipy.ndimage import gaussian_filter1d

def fit_gaussian(im, thres, do_xy):
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
    sv = min_sv * np.minimum(2, sv/min_sv)

    # compute pts surrounding ellipse
    if do_xy:
        n = 50 # Number of points around ellipse
        p = np.linspace(0, 2*np.pi, n)[:, np.newaxis]
        # Transformation
        xy = np.concatenate((np.cos(p), np.sin(p)),axis=1) * (sv**0.5) @ u
        xy += mu
    else:
        xy = []
    return mu, sv, xy

def process(img, saturation, pupil_sigma):
    # smooth in time by two bins
    cumsum = np.cumsum(img, axis=-1)
    img[:,:,1:-1] = (cumsum[:, :, 2:] - cumsum[:, :, :-2]) / float(2)
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
