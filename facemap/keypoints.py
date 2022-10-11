import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d, uniform_filter1d, median_filter
from scipy.interpolate import interp1d
from scipy.linalg import eigh
import math, os
from scipy import signal
import pandas as pd


def get_confidence_threshold(conf, baseline_window=200):
    """ get threshold on confidence for keypoints
    
    confidence is timepoints by keypoints 
    
    """
    conf_baseline = conf - gaussian_filter1d(conf, baseline_window, axis=0)
    threshold = -6 * conf_baseline.std(axis=0)
    return conf_baseline, threshold

def nanmedian_filter(x, win=7):
    """ nanmedian filter array along last axis"""
    nt = x.shape[-1]
    # pad so that x will be divisible by win
    pad = (win - (nt + 2*(win//2)) % win) % win
    xpad = np.pad(x, (win//2, win//2+win+pad), mode='edge')
    xmed = np.zeros_like(x)
    for k in range(win):
        xm = np.nanmedian(xpad[k:k-win].reshape(-1, win), axis=-1)
        xmed[...,k::win] = xm[:len(np.arange(k,nt,win))]
    return xmed

def filter_outliers(x, y, filter_window = 15, baseline_window = 50, max_spike = 25, max_diff = 25):

    # remove frames with large jumps
    x_diff = np.abs(np.append(np.zeros(1,), np.diff(x)))
    y_diff = np.abs(np.append(np.zeros(1,), np.diff(y)))
    replace_inds = np.logical_or(x_diff > max_diff, y_diff > max_diff)
    x[replace_inds] = np.nan
    y[replace_inds] = np.nan
    
    # remove frames with large deviations from baseline
    x_baseline = nanmedian_filter(x, baseline_window)
    y_baseline = nanmedian_filter(y, baseline_window)
    replace_inds = np.logical_or(np.abs(x - x_baseline) > max_spike, np.abs(y - y_baseline) > max_spike)
    x[replace_inds] = np.nan
    y[replace_inds] = np.nan
    replace_inds = np.isnan(x)
    
    # filter x and y
    x_filt = nanmedian_filter(x, filter_window)
    y_filt = nanmedian_filter(y, filter_window)
    #x_filt = x_baseline
    #y_filt = y_baseline
    
    # this in theory shouldn't add more frames
    replace_inds = np.logical_or(replace_inds, np.isnan(x_filt))
    ireplace = np.nonzero(replace_inds)[0]

    # replace outlier frames with median
    if len(ireplace) > 0:
        # good indices
        iinterp = np.nonzero(np.logical_and(~replace_inds, ~np.isnan(x_filt)))[0]
        x[replace_inds] = np.interp(ireplace, iinterp, x_filt[iinterp])
        y[replace_inds] = np.interp(ireplace, iinterp, y_filt[iinterp])

    if 0:
        # replace overall outlier deflections from baseline
        x_baseline = x.mean() #nanmedian_filter(x, baseline_window)
        y_baseline = y.mean() #nanmedian_filter(y, baseline_window)
        max_spike = x.std() * 5, y.std() * 5
        replace_inds = np.logical_or(np.abs(x - x_baseline) > max_spike[0], 
                                    np.abs(y - y_baseline) > max_spike[1])
        x[replace_inds] = x.mean()#_baseline[replace_inds]
        y[replace_inds] = y.mean()#_baseline[replace_inds]
    
    return x,y 

def keypoint_labels_per_cam(cam_type=0):
    if cam_type==0:
        keypoints_labels = ['eye(back)', 'eye(bottom)', 'eye(front)', 'eye(top)', 
                            'whisker(c1)', 'whisker(d2)', 'whisker(d1)',
                            'nose(bottom)', 'nose(r)', 'nose(tip)', 'nose(top)', 'paw']
    else:
        keypoints_labels = ['eye(back)', 'eye(bottom)', 'eye(front)', 'eye(top)', 'whisker(c1)', 'whisker(d2)', 'whisker(d1)',
                            'nose(bottom)', 'nose(tip)', 'nose(top)', 'mouth', 'lowerlip', 'paw']
    return keypoints_labels

def load_keypoints(kp_file, outlier_filter=True, median_filter=False, 
                        keypoint_labels=[], confidence_threshold=False):

    use_all = False
    if keypoint_labels is None:
        cam_type = int(os.path.split(kp_file)[1][3])
        keypoint_labels = keypoint_labels_per_cam(cam_type)
    elif len(keypoint_labels)==0:
        use_all = True

    df = pd.read_hdf(kp_file)
    if use_all:
        inds = np.arange(0, len(df.columns), 3)
        keypoint_labels = df.columns.get_level_values('bodyparts')[::3]
    else:
        if 'whisker(c2)' in list(df.columns.get_level_values('bodyparts')):
            keypoint_labels[5] = 'whisker(c2)'
        inds = np.array([(df.columns.get_level_values('bodyparts') == label).nonzero()[0][0] for label in keypoint_labels])
        if 'whisker(c2)' in list(df.columns.get_level_values('bodyparts')):
            keypoint_labels[5] = 'whisker(d2)'
    xy = np.stack((df.values[:,inds], df.values[:,inds+1]), axis=-1)
    conf = df.values[:,inds+2]
    
    if not isinstance(confidence_threshold, float) and confidence_threshold is True:
        conf, confidence_threshold = get_confidence_threshold(conf)
    if not isinstance(confidence_threshold, (bool, int)):
        xy[conf < confidence_threshold] = np.nan
        print(np.isnan(xy).sum())
    
    if outlier_filter:
        for i in range(xy.shape[1]):
            x, y = xy[:,i,0], xy[:,i,1]
            x, y = filter_outliers(x, y)
            xy[:,i] = np.vstack((x,y)).T
    

    return xy, list(keypoint_labels)

def keypoints_features(xy):
    xy_vel = (np.diff(xy, axis=0)**2).sum(axis=-1)**0.5
    xy_vel = np.vstack((xy_vel[[0]], xy_vel))
    xy_rad = ((xy - xy.mean(axis=0))**2).sum(axis=-1)**0.5
    xy_dists = compute_dists(xy)
    return xy, xy_vel, xy_rad, xy_dists

def compute_dists(xy):
    xy_dists = ((xy[:,:,np.newaxis] - xy[:,np.newaxis])**2).mean(axis=-1)
    upper_triangular = np.triu_indices(xy_dists.shape[-1], 1)
    xy_dists = xy_dists[:, upper_triangular[0], upper_triangular[1]]
    return xy_dists

def gabor_wavelet(sigma, f, ph, n_pts=201, is_torch=False):
    x = np.linspace(0, 2*np.pi, n_pts+1)[:-1].astype('float32')
    cos = np.cos
    sin = np.sin
    exp = np.exp
    xc = x - x.mean()
    cosine = cos(ph + f * xc)
    gaussian = exp(-(xc**2) / (2*sigma**2))
    G = gaussian * cosine
    G /= (G**2).sum()**0.5
    return G

def get_gabor_transform(data, freqs=np.geomspace(1, 10, 5)):
    """ data is time points by features """
    n_time, n_features = data.shape
    n_widths = len(freqs)
    gabor_transform = np.zeros((n_time, 2*n_widths, n_features), 'float32')    
    for k,f in enumerate(freqs):
        #plt.plot(gabor_wavelet(2,f,0))
        gw0 = gabor_wavelet(1,f,0)
        gw1 = gabor_wavelet(1,f,np.pi/2)
        for j in range(n_features):
            filt0 = np.convolve(zscore(data[:,j]), gw0, mode='same')
            filt1 = np.convolve(zscore(data[:,j]), gw1, mode='same')  
            gabor_transform[:,2*k,j] = filt0
            gabor_transform[:,2*k+1,j] = (filt0**2 + filt1**2)**0.5
    return gabor_transform

def get_wavelet_transform(data, wavelet_func=['gaus1','mexh'], widths=np.geomspace(1,60,5)):
    """ data is time points by features """
    import pywt
    n_wavelets = len(wavelet_func)
    n_widths = len(widths)
    n_time, n_features = data.shape
    wavelet_transform = np.zeros((n_time, n_widths, n_wavelets, n_features), 'float32')
    for k, wave_type in enumerate(wavelet_func):
        for i in range(data.shape[1]):
            coef = pywt.cwt(data[:,i] - data[:,i].mean(), widths, wave_type)[0].T
            #coef -= coef.mean(axis=0)
            #pca = PCA(n_components=2).fit_transform(coef)
            wavelet_transform[:, :, k, i] = coef
    return wavelet_transform

def compute_wavelet_transforms(data, n_comps=3):
    wt = get_wavelet_transform(data)
    gt = get_gabor_transform(data)
    
    wgt_pcs = np.zeros((gt.shape[0], n_comps*3, gt.shape[-1]), 'float32')
    for j in range(gt.shape[-1]):
        wt_j = wt[:,:,:,j]
        wt_j /= wt_j[:,0].std(axis=0)
        #wt_j = wt_j.reshape(wt_j.shape[0], -1)
        gt_j = gt[:,:,j]
        gt_j /= gt_j[:,0].std(axis=0)
        wgt_pcs[:,:3,j] = PCA(n_components=n_comps).fit_transform(gt_j)
        #gt_j @ 
        wgt_pcs[:,3:6,j] = PCA(n_components=n_comps).fit_transform(wt_j[:,:,0])
        wgt_pcs[:,6:,j] = PCA(n_components=n_comps).fit_transform(wt_j[:,:,1])
        #wgt_pcs[:,:,j] = PCA(n_components=9).fit_transform(np.hstack((gt_j, wt_j[:,:,0], wt_j[:,:,1])))
    return wgt_pcs
  
def find_representative_points(points, winsize=100, n_repcheck=500):
    """ find times with minimal movement for resting face for alignment """
    points_meancenter = points - np.nanmedian(points,axis=0) # mean-center
    points_meancenter /= np.nanstd(points, axis=0)
    points_unif = uniform_filter1d(points_meancenter,size=winsize, axis=0) # uniform filter
    #points_collapse = np.nanmean(points_unif,axis=(2,1)) # mean over all keypoints
    points_smooth = np.abs(gaussian_filter1d(np.diff(points_unif, axis=0),sigma=3)) # diff smooth abs
    # moving average
    movavg = gaussian_filter1d(points_smooth, winsize, axis=0)
    movavg = movavg.sum(axis=(1,2))
    rep_times = np.argsort(movavg)[:n_repcheck]
    return np.median(points[rep_times], axis=0), rep_times



    
