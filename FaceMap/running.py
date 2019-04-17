# outputs the dx, dy offsets between frames by registering frame N to frame
# N-1. If the movement is larger than half the frame size, outputs NaN.
# ops.yrange, xrange are ranges to use for rectangular section of movie

import numpy as np
import numpy.fft as fft

try:
    import mkl_fft
    HAS_MKL=True
except ImportError:
    HAS_MKL=False


def fft2(data, s=None):
    if s==None:
        s=(data.shape[-2], data.shape[-1])
    if HAS_MKL:
        data = mkl_fft.fft2(data,shape=s,axes=(-2,-1))
    else:
        data = fft.fft2(data, s, axes=(-2,-1))
    return data

def ifft2(data, s=None):
    if s==None:
        s=(data.shape[-2], data.shape[-1])
    if HAS_MKL:
        data = mkl_fft.ifft2(data, shape=s, axes=(-2,-1))
    else:
        data = fft.ifft2(data, s, axes=(-2,-1))
    return data

def spatial_taper(sig, Ly, Lx):
    ''' spatial taper  on edges with gaussian of std sig '''
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    mY = y.max() - 2*sig
    mX = x.max() - 2*sig
    maskY = 1./(1.+np.exp((yy-mY)/sig))
    maskX = 1./(1.+np.exp((xx-mX)/sig))
    maskMul = maskY * maskX
    return maskMul

def process(data):
    ''' computes movement using phase correlation  '''
    nt, Ly, Lx = data.shape
    lx = int(np.ceil(Lx / 4))
    ly = int(np.ceil(Ly / 4))
    xc = int(np.floor(Lx/2))
    yc = int(np.floor(Ly/2))
    yr = np.arange(yc - ly, yc + ly, 1, int)
    xr = np.arange(xc - lx, xc + lx, 1, int)

    data -= data.mean(axis=0).mean(axis=0)
    data  = data.astype(np.float32)

    maskMul = spatial_taper((Ly*Lx)**0.5*0.1, Ly, Lx)
    data *= maskMul
    data = fft2(data)

    eps0 = 1e-20
    data /= (eps0 + np.abs(data))

    cc = data[:-1] * np.conj(data[1:])

    cc = np.real(ifft2(cc))
    cc = cc[np.ix_(np.arange(0, nt-1, 1, int), yr, xr)]
    cc = np.reshape(fft.fftshift(cc, axes=(1,2)), (nt-1, -1))

    ix = np.argmax(cc, axis=1)
    ymax, xmax = np.unravel_index(ix, (2*ly+1, 2*lx+1))
    ymax, xmax = ymax+ly, xmax+lx
    return ymax, xmax
