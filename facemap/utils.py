import numpy as np
from scipy.sparse.linalg import eigsh


def multivideo_reshape(X, LY, LX, sy, sx, Ly, Lx, iinds):
    ''' reshape X matrix pixels x n matrix into LY x LX - embed each video at sy, sx'''
    ''' iinds are indices of each video in concatenated array'''
    X_reshape = np.zeros((LY,LX,X.shape[-1]), np.float32)
    for i in range(len(Ly)):
        X_reshape[sy[i]:sy[i]+Ly[i],
                  sx[i]:sx[i]+Lx[i]] = np.reshape(X[iinds[i]], (Ly[i],Lx[i],X.shape[-1]))
    return X_reshape

def roi_to_dict(ROIs, rROI=None):
    rois = []
    for i,r in enumerate(ROIs):
        rois.append({'rind': r.rind, 'rtype': r.rtype, 'iROI': r.iROI, 'ivid': r.ivid, 'color': r.color,
                    'yrange': r.yrange, 'xrange': r.xrange, 'saturation': r.saturation})
        if hasattr(r, 'pupil_sigma'):
            rois[i]['pupil_sigma'] = r.pupil_sigma
        if hasattr(r, 'ellipse'):
            rois[i]['ellipse'] = r.ellipse
        if rROI is not None:
            if len(rROI[i]) > 0:
                rois[i]['reflector'] = []
                for rr in rROI[i]:
                    rdict = {'yrange': rr.yrange, 'xrange': rr.xrange, 'ellipse': rr.ellipse}
                    rois[i]['reflector'].append(rdict)

    return rois


def get_reflector(yrange, xrange, rROI=None, rdict=None):
    reflectors = np.zeros((yrange.size, xrange.size), np.bool)
    if rROI is not None and len(rROI)>0:
        for r in rROI:
            ellipse, ryrange, rxrange = r.ellipse.copy(), r.yrange.copy(), r.xrange.copy()
            ix = np.logical_and(rxrange >= 0, rxrange < xrange.size)
            ellipse = ellipse[:,ix]
            rxrange = rxrange[ix]
            iy = np.logical_and(ryrange >= 0, ryrange < yrange.size)
            ellipse = ellipse[iy,:]
            ryrange = ryrange[iy]
            reflectors[np.ix_(ryrange, rxrange)] = np.logical_or(reflectors[np.ix_(ryrange, rxrange)], ellipse)
    elif rdict is not None and len(rdict)>0:
        for r in rdict:
            ellipse, ryrange, rxrange = r['ellipse'].copy(), r['yrange'].copy(), r['xrange'].copy()
            ix = np.logical_and(rxrange >= 0, rxrange < xrange.size)
            ellipse = ellipse[:,ix]
            rxrange = rxrange[ix]
            iy = np.logical_and(ryrange >= 0, ryrange < yrange.size)
            ellipse = ellipse[iy,:]
            ryrange = ryrange[iy]
            reflectors[np.ix_(ryrange, rxrange)] = np.logical_or(reflectors[np.ix_(ryrange, rxrange)], ellipse)
    return reflectors.nonzero()

def video_placement(Ly, Lx):
    ''' Ly and Lx are lists of video sizes '''
    npix = Ly * Lx
    picked = np.zeros((Ly.size,), np.bool)
    ly = 0
    lx = 0
    sy = np.zeros(Ly.shape, int)
    sx = np.zeros(Lx.shape, int)
    if Ly.size==2:
        gridy = 1
        gridx = 2
    elif Ly.size==3:
        gridy = 1
        gridx = 2
    else:
        gridy = int(np.round(Ly.size**0.5 * 0.75))
        gridx = int(np.ceil(Ly.size / gridy))
    LY = 0
    LX = 0
    iy = 0
    ix = 0
    while (~picked).sum() > 0:
        # place biggest movie first
        npix0 = npix.copy()
        npix0[picked] = 0
        imax = np.argmax(npix0)
        picked[imax] = 1
        if iy==0:
            ly = 0
            rowmax=0
        if ix==0:
            lx = 0
        sy[imax] = ly
        sx[imax] = lx

        ly+=Ly[imax]
        rowmax = max(rowmax, Lx[imax])
        if iy==gridy-1 or (~picked).sum()==0:
            lx+=rowmax
        LY = max(LY, ly)
        iy+=1
        if iy >= gridy:
            iy = 0
            ix += 1
    LX = lx
    return LY, LX, sy, sx


def svdecon(X, k=100):
    NN, NT = X.shape
    if NN>NT:
        COV = (X.T @ X)/NT
    else:
        COV = (X @ X.T)/NN
    if k==0:
        k = np.minimum(COV.shape) - 1
    Sv, U = eigsh(COV, k = k)
    U, Sv = np.real(U), np.abs(Sv)
    U, Sv = U[:, ::-1], Sv[::-1]**.5
    if NN>NT:
        V = U
        U = X @ V
        U = U/(U**2).sum(axis=0)**.5
    else:
        V = (U.T @ X).T
        V = V/(V**2).sum(axis=0)**.5
    return U, Sv, V
