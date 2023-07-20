"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os

import numpy as np
import torch

from .. import keypoints
from ..utils import bin1d, compute_varexp
from .prediction_utils import resample_data, rrr_prediction


def get_normalized_keypoints(keypoints_path, exclude_keypoints=None, running=None):
    """
    Load keypoints and normalize them
    Parameters
    ----------
    keypoints_path : str
        path to keypoints file
    Returns
    -------
    keypoints_normalized : 2D-array
        normalized keypoints of shape [n_keypoints x 2, time]
    """
    # Load keypoints
    if os.path.splitext(keypoints_path)[-1] == ".h5":
        xy, keypoint_labels = keypoints.load_keypoints(keypoints_path)
    else:
        kp = np.load(keypoints_path, allow_pickle=True).item()
        xy, keypoint_labels = kp["xy"], kp["keypoint_labels"]
    if exclude_keypoints is not None:
        xy0 = np.zeros((xy.shape[0], 0, 2))
        keypoint_labels0 = []
        for k, key in enumerate(keypoint_labels):
            if exclude_keypoints not in key:
                xy0 = np.concatenate((xy0, xy[:, [k]]), axis=1)
                keypoint_labels0.append(key)
        xy, keypoint_labels = xy0, keypoint_labels0
    print("predicting neural activity using...")
    print(keypoint_labels)
    print("xy shape", xy.shape)

    # Normalize keypoints (input data x)
    x = xy.reshape(xy.shape[0], -1).copy()
    if running is not None:
        x = np.concatenate((x, running[:, np.newaxis]), axis=-1)
        print("and running")
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    print("x shape", x.shape)
    return x


def rrr_varexp_kps(
    kp_path,
    tcam,
    tneural,
    Y,
    U,
    spks,
    delay=-1,
    tbin=4,
    rank=32,
    device=torch.device("cuda"),
):
    """predict neural PCs with keypoint wavelets Y and compute varexp for PCs and spks"""
    varexp_neurons = np.nan * np.zeros((len(spks), 2))
    xy, keypoint_labels = keypoints.load_keypoints(kp_path, keypoint_labels=None)
    xy_dists = keypoints.compute_dists(xy)
    X = keypoints.compute_wavelet_transforms(xy_dists)
    X = X.reshape(X.shape[0], -1)

    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    X_ds = resample_data(X, tcam, tneural, crop="linspace")
    if delay < 0:
        Ys = np.vstack((Y[-delay:], np.tile(Y[[-1], :], (-delay, 1))))
    else:
        X_ds = np.vstack((X_ds[delay:], np.tile(X_ds[[-1], :], (delay, 1))))
        Ys = Y

    Y_pred_test, varexp, itest, A, B = rrr_prediction(
        X_ds, Ys, rank=Y.shape[-1], lam=1e-3, tbin=tbin, device=device
    )[:5]
    # return Y_pred_test at specified rank
    Y_pred_test = X_ds[itest] @ B[:, :rank] @ A[:, :rank].T

    itest -= delay
    # single neuron prediction
    spks_pred_test = Y_pred_test @ U.T
    spks_test = spks[:, itest].T
    varexp_neurons[:, 0] = compute_varexp(spks_test, spks_pred_test)
    spks_test_bin = bin1d(spks_test, tbin)
    spks_pred_test_bin = bin1d(spks_pred_test, tbin)
    varexp_neurons[:, 1] = compute_varexp(spks_test_bin, spks_pred_test_bin)

    return varexp, varexp_neurons, spks_pred_test, itest
