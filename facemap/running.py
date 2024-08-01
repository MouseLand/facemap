"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
# outputs the dx, dy offsets between frames by registering frame N to frame
# N-1. If the movement is larger than half the frame size, outputs NaN.
# ops.yrange, xrange are ranges to use for rectangular section of movie
import numpy as np
from numba import complex64, vectorize
from numpy.fft import fft2, ifft2, ifftshift

eps0 = 1e-20


def spatial_taper(sig, Ly, Lx):
    """spatial taper  on edges with gaussian of std sig"""
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    mY = y.max() - 2 * sig
    mX = x.max() - 2 * sig
    maskY = 1.0 / (1.0 + np.exp((yy - mY) / sig))
    maskX = 1.0 / (1.0 + np.exp((xx - mX) / sig))
    maskMul = maskY * maskX
    return maskMul


def gaussian_fft(sig, Ly, Lx):
    """gaussian filter in the fft domain with std sig and size Ly,Lx"""
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    hgx = np.exp(-np.square(xx / sig) / 2)
    hgy = np.exp(-np.square(yy / sig) / 2)
    hgg = hgy * hgx
    hgg /= hgg.sum()
    fhg = np.real(fft2(ifftshift(hgg)))
    # smoothing filter in Fourier domain
    return fhg


@vectorize(
    ["complex64(uint8, complex64)", "complex64(float32, complex64)"],
    nopython=True,
    target="parallel",
)
def multiplytype(x, y):
    return np.complex64(x) * y


@vectorize([complex64(complex64, complex64)], nopython=True, target="parallel")
def apply_dotnorm(Y, cfRefImg):
    x = Y / (eps0 + np.abs(Y))
    x = x * cfRefImg
    return x


@vectorize([complex64(complex64, complex64)], nopython=True, target="parallel")
def phase_norm(x, y):
    x = x / (eps0 + np.abs(x))
    x = x * y
    return x


def my_clip(X, lcorr):
    x00 = X[:, : lcorr + 1, : lcorr + 1]
    x11 = X[:, -lcorr:, -lcorr:]
    x01 = X[:, : lcorr + 1, -lcorr:]
    x10 = X[:, -lcorr:, : lcorr + 1]
    return x00, x01, x10, x11


def spatial_smooth(data, N):
    """spatially smooth data using cumsum over axis=1,2 with window N"""
    pad = np.zeros((data.shape[0], int(N / 2), data.shape[2]))
    dsmooth = np.concatenate((pad, data, pad), axis=1)
    pad = np.zeros((dsmooth.shape[0], dsmooth.shape[1], int(N / 2)))
    dsmooth = np.concatenate((pad, dsmooth, pad), axis=2)
    # in X
    cumsum = np.cumsum(dsmooth, axis=1)
    dsmooth = (cumsum[:, N:, :] - cumsum[:, :-N, :]) / float(N)
    # in Y
    cumsum = np.cumsum(dsmooth, axis=2)
    dsmooth = (cumsum[:, :, N:] - cumsum[:, :, :-N]) / float(N)
    return dsmooth


def process(data):
    """computes movement using phase correlation"""
    nt, Ly, Lx = data.shape
    ly = int(np.floor(Ly * 0.4))
    lx = int(np.floor(Lx * 0.4))
    lcorr = min(ly, lx)

    # taper edges
    maskMul = spatial_taper((Ly * Lx) ** 0.5 * 0.01, Ly, Lx).astype(np.complex64)

    # spatial filter
    fhg = gaussian_fft(2, Ly, Lx)

    # shifts and corrmax
    ymax = np.zeros((nt - 1,), np.int32)
    xmax = np.zeros((nt - 1,), np.int32)
    # cmax = np.zeros((nimg,), np.float32)

    # maskOffset = X[1:].mean() * (1. - maskMul)

    # taper and fft data
    X = data.astype(np.float32)
    # data -= data.mean(axis=-1, dtype=np.uint8).mean(axis=-1, dtype=np.uint8)[:,np.newaxis,np.newaxis]
    X -= X.mean(axis=-1).mean(axis=-1)[:, np.newaxis, np.newaxis]
    X = multiplytype(X, maskMul)
    # X -= X.mean(axis=-1).mean(axis=-1)[:,np.newaxis,np.newaxis]
    for t in range(nt):
        fft2(X[t])
    # phase correlation with previous frame
    X = phase_norm(X, fhg.astype(np.complex64))
    Xout = X[1:] * np.conj(X[:-1])
    for t in range(nt - 1):
        ifft2(Xout[t])
    x00, x01, x10, x11 = my_clip(Xout, lcorr)
    cc = np.real(np.block([[x11, x10], [x01, x00]]))
    # cc = spatial_smooth(cc, 2)
    for t in range(nt - 1):
        ymax[t], xmax[t] = np.unravel_index(
            np.argmax(cc[t], axis=None), (2 * lcorr + 1, 2 * lcorr + 1)
        )
        # cmax[t] = cc[t, ymax[t], xmax[t]]
    ymax, xmax = ymax - lcorr, xmax - lcorr
    return ymax, xmax
