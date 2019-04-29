# outputs the dx, dy offsets between frames by registering frame N to frame
# N-1. If the movement is larger than half the frame size, outputs NaN.
# ops.yrange, xrange are ranges to use for rectangular section of movie
from scipy.fftpack import next_fast_len
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

def gaussian_fft(sig, Ly, Lx):
    ''' gaussian filter in the fft domain with std sig and size Ly,Lx '''
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    hgx = np.exp(-np.square(xx/sig) / 2)
    hgy = np.exp(-np.square(yy/sig) / 2)
    hgg = hgy * hgx
    hgg /= hgg.sum()
    fhg = np.real(fft.fft2(fft.ifftshift(hgg))); # smoothing filter in Fourier domain
    return fhg

def process(data):
    ''' computes movement using phase correlation  '''
    nt, Ly, Lx = data.shape
    ly = int(np.ceil(Ly * 0.4))
    lx = int(np.ceil(Lx * 0.4))

    data -= data.mean(axis=-1).mean(axis=-1)[:,np.newaxis,np.newaxis]
    data  = data.astype(np.float32)

    # taper edges
    maskMul = spatial_taper((Ly*Lx)**0.5*0.01, Ly, Lx)
    data *= maskMul

    # spatial filter
    lyy, lxx = next_fast_len(Ly), next_fast_len(Lx)
    fhg = gaussian_fft(2, lyy, lxx)

    data = fft2(data, s=(lyy,lxx))

    eps0 = 1e-20
    data /= (eps0 + np.abs(data))

    cc = data[:-1] * np.conj(data[1:]) * fhg

    cc = np.real(ifft2(cc))
    cc = fft.fftshift(cc, axes=(1,2))

    yc = int(np.floor(cc.shape[1]/2))
    xc = int(np.floor(cc.shape[2]/2))
    yr = np.arange(yc - ly, yc + ly + 1, 1, int)
    xr = np.arange(xc - lx, xc + lx + 1, 1, int)

    cc = cc[np.ix_(np.arange(0, nt-1, 1, int), yr, xr)]
    cc = np.reshape(cc, (nt-1, -1))

    ix = np.argmax(cc, axis=1)
    ymax, xmax = np.unravel_index(ix, (2*ly+1, 2*lx+1))
    ymax, xmax = ymax-ly, xmax-lx

    return ymax, xmax
