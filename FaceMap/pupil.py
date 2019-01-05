import numpy as np

def fit_gaussian(im, thres):
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
    n = 50 # Number of points around ellipse
    p = np.linspace(0, 2*np.pi, n)[:, np.newaxis]
    # Transformation
    xy = np.concatenate((np.cos(p), np.sin(p)),axis=1) * (sv**0.5) @ u
    xy += mu

    return mu, sigxy, xy
