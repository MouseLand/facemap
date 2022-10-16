import os
import time

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from torch.nn.functional import conv1d

from facemap import keypoints
from facemap.utils import bin1d

from .model import KeypointsNetwork


def split_traintest(n_t, frac=0.25, pad=3):
    """this returns deterministic split of train and test in time chunks

    Parameters
    ----------

    n_t : int
        number of timepoints to split

    frac : float (optional, default 0.25)
        fraction of points to put in test set

    pad : int (optional, default 3)
        number of timepoints to exclude from test set before and after training segment, 
        in addition to 5 timepoints auto-included

    Returns
    --------

    itrain: 2D int array
        times in train set, arranged in chunks

    itest: 2D int array
        times in test set, arranged in chunks

    """
    # usu want 10 segs, but might not have enough frames for that
    n_segs = int(min(10, n_t / 4))
    n_len = int(np.floor(n_t / n_segs))
    inds_train = np.linspace(0, n_t - n_len - 5, n_segs).astype(int)
    l_train = int(np.floor(n_len * (1 - frac)))
    inds_test = inds_train + l_train + pad
    l_test = (
        np.diff(np.stack((inds_train, inds_train + l_train)).T.flatten()).min()
        - pad
    )
    itrain = inds_train[:, np.newaxis] + np.arange(0, l_train, 1, int)
    itest = inds_test[:, np.newaxis] + np.arange(0, l_test, 1, int)
    return itrain, itest


def resample_frames(data, torig, tout):
    """
    Resample data from times torig at times tout.
    data is (n_samples, n_features). The data is filtered using a gaussian filter before resampling.

    Parameters
    ----------
    data : 2D array, input data (n_samples, n_features)

    torig : 1D-array, original times

    tout : 1D-array, times to resample to

    Returns
    --------

    dout : ND-array
        data resampled at tout

    """
    fs = torig.size / tout.size  # relative sampling rate
    data = gaussian_filter1d(data, np.ceil(fs / 4), axis=0)
    f = interp1d(torig, data, kind="linear", axis=0, fill_value="extrapolate")
    dout = f(tout)
    return dout


def resample_data(data, tcam, tneural, crop="linspace"):
    """
    Resample data from camera times tcam at times tneural

    sometimes there are fewer camera timestamps than frames, so data is cropped

    data is (n_samples, n_features). The data is filtered using a gaussian filter before resampling.

    Parameters
    ----------
    data : 2D array, input data (n_samples, n_features)

    tcam : 1D-array, original times

    tneural : 1D-array, times to resample to

    Returns
    --------

    data_resampled : ND-array
        data resampled at tout

    """
    ttot = len(data)
    tc = len(tcam)
    if crop == "end":
        d = data[:tc]
    elif crop == "start":
        d = data[ttot - tc :]
    elif crop == "linspace":
        d = data[np.linspace(0, ttot - 1, tc).astype(int)]
    else:
        d = data[(ttot - tc) // 2 : (ttot - tc) // 2 + tc]
    data_resampled = resample_frames(d, tcam, tneural)
    return data_resampled


def ridge_regression(X, Y, lam=0):
    """predict Y from X using regularized linear regression

    *** subtract mean from X and Y before predicting

    Prediction:
    >>> Y_pred = X @ A

    Parameters
    ----------

    X : 2D array, input data (n_samples, n_features)

    Y : 2D array, data to predict (n_samples, n_predictors)

    Returns
    --------

    A : 2D array - prediction matrix 1 (n_predictors, rank)
    """
    CXX = (X.T @ X + lam * np.eye(X.shape[1])) / X.shape[0]
    CXY = (X.T @ Y) / X.shape[0]
    A = np.linalg.solve(CXX, CXY)
    return A


def reduced_rank_regression(X, Y, rank=None, lam=0):
    """predict Y from X using regularized reduced rank regression

    *** subtract mean from X and Y before predicting

    if rank is None, returns A and B of full-rank (minus one) prediction

    Prediction:
    >>> Y_pred = X @ B @ A.T

    Parameters
    ----------

    X : 2D array, input data (n_samples, n_features)

    Y : 2D array, data to predict (n_samples, n_predictors)

    rank : int (optional, default None)
        rank to compute reduced rank regression for

    lam : float (optional, default 0)
        regularizer

    Returns
    --------

    A : 2D array - prediction matrix 1 (n_predictors, rank)

    B : 2D array - prediction matrix 2 (n_features, rank)

    """
    min_dim = min(Y.shape[1], min(X.shape[0], X.shape[1])) - 1
    if rank is None:
        rank = min_dim
    else:
        rank = min(min_dim, rank)

    # make covariance matrices
    CXX = (X.T @ X + lam * np.eye(X.shape[1])) / X.shape[0]
    CYX = (Y.T @ X) / X.shape[0]

    # compute inverse square root of matrix
    s, u = eigh(CXX)
    # u = model.components_.T
    # s = model.singular_values_**2
    CXXMH = (u * (s + lam) ** -0.5) @ u.T

    # project into prediction space
    M = CYX @ CXXMH

    # do svd of prediction projection
    model = PCA(n_components=rank).fit(M)
    c = model.components_.T
    s = model.singular_values_
    A = M @ c
    B = CXXMH @ c

    return A, B


def rrr_prediction(
    X, Y, rank=None, lam=0, itrain=None, itest=None, tbin=None
):
    """predict Y from X using regularized reduced rank regression for all ranks up to "rank"

    *** subtract mean from X and Y before predicting

    if rank is None, returns A and B of full-rank (minus one) prediction

    Prediction:
    >>> Y_pred_test = X_test @ B @ A.T

    Parameters
    ----------

    X : 2D array, input data (n_samples, n_features)

    Y : 2D array, data to predict (n_samples, n_predictors)

    rank : int (optional, default None)
        rank up to which to compute reduced rank regression for

    lam : float (optional, default 0)
        regularizer

    itrain: 1D int array (optional, default None)
        times in train set

    itest: 1D int array (optional, default None)
        times in test set

    tbin: int (optional, default None)
        also compute variance explained in bins of tbin

    Returns
    --------

    Y_pred_test : 2D array - prediction of Y with max rank (len(itest), n_features)

    varexp : 1D array - variance explained across all features (rank,)

    itest: 1D int array
        times in test set

    A : 2D array - prediction matrix 1 (n_predictors, rank)

    B : 2D array - prediction matrix 2 (n_features, rank)

    varexpf : 1D array - variance explained per feature (rank, n_features)

    corrf : 1D array - correlation with Y per feature (rank, n_features)

    """
    n_t, n_feats = Y.shape
    if itrain is None and itest is None:
        itrain, itest = split_traintest(n_t)
    itrain, itest = itrain.flatten(), itest.flatten()
    A, B = reduced_rank_regression(X[itrain], Y[itrain], rank=rank, lam=lam)
    rank = A.shape[1]
    corrf = np.zeros((rank, n_feats))
    varexpf = np.zeros((rank, n_feats))
    varexp = np.zeros((rank, 2)) if tbin is not None else np.zeros((rank, 1))
    Y_pred_test = np.zeros((len(itest), n_feats))
    for r in range(rank):
        Y_pred_test = X[itest] @ B[:, : r + 1] @ A[:, : r + 1].T
        Y_test_var = (Y[itest] ** 2).mean(axis=0)
        corrf[r] = (Y[itest] * Y_pred_test).mean(axis=0) / (
            Y_test_var**0.5 * Y_pred_test.std(axis=0)
        )
        residual = ((Y[itest] - Y_pred_test) ** 2).mean(axis=0)
        varexpf[r] = 1 - residual / Y_test_var
        varexp[r, 0] = 1 - residual.mean() / Y_test_var.mean()
        if tbin is not None and tbin > 1:
            varexp[r, 1] = compute_varexp(
                bin1d(Y[itest], tbin).flatten(), bin1d(Y_pred_test, tbin).flatten()
            )

    return Y_pred_test, varexp.squeeze(), itest, A, B, varexpf, corrf


def rrr_varexp_svds(
    svd_path, tcam, tneural, Y, U, spks, delay=-1, running=None, 
    tbin=4, rank=32
):
    """predict neural PCs Y and compute varexp for PCs and spks"""
    varexp = np.nan * np.zeros((127, 2, 2))
    varexp_neurons = np.nan * np.zeros((len(spks), 2, 2))
    svds = np.load(svd_path, allow_pickle=True).item()
    spks_pred_test0 = [] 
    Y_pred_test0 = []
    for k, key in enumerate(["motSVD", "movSVD"]):
        X = svds[key][0].copy()
        if running is not None: 
            X = np.concatenate((X, running[:,np.newaxis]), axis=-1)

        X -= X.mean(axis=0)
        X /= X[:, 0].std(axis=0)

        X_ds = resample_data(X, tcam, tneural, crop="linspace")
        if delay < 0:
            Ys = np.vstack((Y[-delay:], np.tile(Y[[-1], :], (-delay, 1))))
        else:
            X_ds = np.vstack((X_ds[delay:], np.tile(X_ds[[-1], :], (delay, 1))))
            Ys = Y
        Y_pred_test, ve_test, itest, A, B = rrr_prediction(
            X_ds, Ys, rank=Y.shape[-1], lam=1e-6, tbin=tbin
        )[:5]
        varexp[:, :, k] = ve_test
        # return Y_pred_test at specified rank
        Y_pred_test = X_ds[itest] @ B[:, :rank] @ A[:, :rank].T

        itest -= delay
        # single neuron prediction
        spks_pred_test = Y_pred_test @ U.T
        spks_test = spks[:, itest].T
        varexp_neurons[:, 0, k] = compute_varexp(spks_test, spks_pred_test)
        spks_test_bin = bin1d(spks_test, tbin)
        spks_pred_test_bin = bin1d(spks_pred_test, tbin)
        varexp_neurons[:, 1, k] = compute_varexp(spks_test_bin, spks_pred_test_bin)
        spks_pred_test0.append(spks_pred_test)
        Y_pred_test0.append(Y_pred_test)
    return varexp, varexp_neurons, np.array(Y_pred_test0), np.array(spks_pred_test0), itest


def rrr_varexp_kps(kp_path, tcam, tneural, Y, U, spks, delay=-1, tbin=4, rank=32):
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
        X_ds, Ys, rank=Y.shape[-1], lam=1e-3, tbin=tbin
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


def compute_varexp(y_true, y_pred):
    """variance explained of y_true by y_pred across axis=0"""
    y_var = ((y_true - y_true.mean(axis=0)) ** 2).mean(axis=0)
    residual = ((y_true - y_pred) ** 2).mean(axis=0)
    varexp = 1 - residual / y_var
    return varexp


def get_neural_pcs(neural_activity, n_comps=128):
    """
    Perform PCA on neural activity data
    Parameters
    ----------
    neural_activity : 2D-array
        neural activity data of shape [neurons x time]
    n_comps : int
        number of principal components to keep
    Returns
    -------
    U : 2D-array
        neural PCs of shape [neurons x n_comps]
    S : 1D-array
        singular values of shape [n_comps]
    V : 2D-array
        neural PCs of shape [n_comps x time]
    """
    neural_activity -= neural_activity.mean(axis=1)[:, np.newaxis]
    std = ((neural_activity**2).mean(axis=1) ** 0.5)[:, np.newaxis]
    std = np.where(std == 0, 1, std)  # don't scale when std==0
    neural_activity /= std
    model = PCA(n_components=n_comps, copy=False).fit(neural_activity)
    U, S = model.components_, model.singular_values_
    V = U.T * S
    return U, S, V


def get_keypoints_to_neural_prediction(
    keypoints_path,
    y,
    behavior_timestamps,
    neural_timestamps,
    running=None,
    exclude_keypoints=None,
    delay=-1,
    verbose=False,
    device=torch.device("cuda"),
):
    """
    Get variance explained of neural PCs prediction by keypoints
    (or variance explained of neural spikes prediction by keypoints)
    Parameters
    ----------
    keypoints_path : str
        path to keypoints file (either .npy or .h5)
    y : 2D-array
        neural PCs or spike responses to predict
    behavior_timestamps : 1D array
        timestamps of behavior data for each frame
    neural_timestamps : 1D array
        timestamps of neural data for each frame
    verbose : bool, optional
        print progress, by default False
    device : torch.device, optional
        device to use for prediction, by default torch.device("cuda")
    """
    # Load keypoints
    if os.path.splitext(keypoints_path)[-1] == ".h5":
        xy, keypoint_labels = keypoints.load_keypoints(keypoints_path)
    else:
        kp = np.load(keypoints_path, allow_pickle=True).item()
        xy, keypoint_labels = kp["xy"], kp["keypoint_labels"]
    if exclude_keypoints is not None:
        xy0=np.zeros((xy.shape[0],0,2))
        keypoint_labels0 = []
        for k,key in enumerate(keypoint_labels):
            if exclude_keypoints not in key:
                xy0 = np.concatenate((xy0, xy[:,[k]]), axis=1)
                keypoint_labels0.append(key)
        xy, keypoint_labels = xy0, keypoint_labels0
    print('predicting neural activity using...')
    print(keypoint_labels)    

    # Normalize keypoints (input data x)
    x = xy.reshape(xy.shape[0], -1).copy()
    if running is not None or (exclude_keypoints is not None and 'running' not in exclude_keypoints): 
        x = np.concatenate((x, running[:,np.newaxis]), axis=-1)
        print('and running')
    x = (x - x.mean(axis=0)) / x.std(axis=0)

    # Initialize model for keypoints to neural prediction
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = KeypointsNetwork(n_in=x.shape[-1], n_out=y.shape[-1]).to(device)

    # Train model and get predictions
    y_pred_test, varexp_testdata, test_indices = train_model(
        model,
        x,
        y,
        behavior_timestamps,
        neural_timestamps,
        delay=delay,
        verbose=verbose,
        device=device,
    )

    return x, keypoint_labels, model, y_pred_test, varexp_testdata, test_indices


def get_keypoints_to_neural_varexp(
    keypoints_path,
    behavior_timestamps,
    neural_timestamps,
    neural_pcs_V,
    neural_pcs_U,
    spks,
    delay=-1,
    tbin=4,
    running=None,
    verbose=False,
    exclude_keypoints=None,
    device=torch.device("cuda"),
):
    """
    Get variance explained of neural PCs prediction by keypoints
    (or variance explained of neural spikes prediction by keypoints)
    Parameters
    ----------
    keypoints_path : str
        path to keypoints file (either .npy or .h5)
    behavior_timestamps : 1D array
        timestamps of behavior data for each frame
    neural_timestamps : 1D array
        timestamps of neural data for each frame
    neural_pcs_V : 2D array
        Transformed neural PCs where each row is a PC and each column is a frame
    neural_pcs_U : 2D array
        Eigenvalues where each row is a PC and each column is a neuron
    spks : 2D array
        Neural activity where each row is a neuron and each column is a frame
    delay : int, optional
        number of frames to delay neural data, by default -1
    tbin : int, optional
        bin size for data, by default 4
    running : array, optional
        1D running trace to include in the prediction model
    verbose : bool, optional
        print progress, by default False
    varexp_per_keypoint : bool, optional
        return variance explained per keypoint, by default False and returns variance explained for all keypoints
    device : torch.device, optional
        device to use for prediction, by default torch.device("cuda")
    """
    (
        x,
        keypoint_labels,
        model,
        y_pred_test,
        varexp_testdata,
        test_indices,
    ) = get_keypoints_to_neural_prediction(
        keypoints_path,
        neural_pcs_V,
        behavior_timestamps,
        neural_timestamps,
        delay=delay,
        running=running,
        exclude_keypoints=exclude_keypoints,
        verbose=verbose,
        device=device,
    )

    n_t = x.shape[0]
    latents = np.zeros((n_t, model.core.features.latent[0].weight.shape[0]), "float32")
    batch_size = 10000
    n_batches = int(np.ceil(n_t / batch_size))
    with torch.no_grad():
        model.eval()
        for n in range(n_batches):
            x_batch = x[n * batch_size : min(n_t, (n + 1) * batch_size)].astype(
                "float32"
            )
            x_batch = torch.from_numpy(x_batch).to(device)
            y_pred, l_batch = model(x_batch.unsqueeze(0))
            latents[
                n * batch_size : min(n_t, (n + 1) * batch_size)
            ] = l_batch.cpu().numpy()

    y_pred_test = y_pred_test.reshape(-1, neural_pcs_V.shape[-1])
    varexp = np.zeros(2)
    varexp_neurons = np.zeros((len(spks), 2))
    varexp[0] = varexp_testdata
    Y_test_bin = bin1d(neural_pcs_V[test_indices.flatten()], tbin)
    Y_pred_test_bin = bin1d(y_pred_test, tbin)
    varexp[1] = (
        1 - ((Y_test_bin - Y_pred_test_bin) ** 2).mean() / ((Y_test_bin) ** 2).mean()
    )
    print(f"all kp, varexp {varexp[0]:.3f}; tbin=4: {varexp[1]:.3f}")
    spks_pred_test = y_pred_test @ neural_pcs_U.T
    spks_test = spks[:, test_indices.flatten()].T
    varexp_neurons[:, 0] = compute_varexp(spks_test, spks_pred_test)
    spks_test_bin = bin1d(spks_test, tbin)
    spks_pred_test_bin = bin1d(spks_pred_test, tbin)
    varexp_neurons[:, 1] = compute_varexp(spks_test_bin, spks_pred_test_bin)
    spks_pred_test0 = spks_pred_test.T.copy()
    y_pred_test0 = y_pred_test.copy()

    return (
            varexp,
            varexp_neurons,
            y_pred_test0,
            spks_pred_test0,
            test_indices,
            latents,
            model,
        )


def get_predicted_neural_activity(
    x, spks, neural_pcs_U, model, device=torch.device("cuda")
):
    """
    Get predicted neural activity from trained model
    """
    model.eval()
    with torch.no_grad():
        y_pred = (
            model(torch.from_numpy(x).float().to(device), torch.arange(x.shape[1]))
            .cpu()
            .numpy()
        )
    spks_pred_test = y_pred @ neural_pcs_U.T
    spks_test = spks.T
    varexp_neurons = compute_varexp(spks_test, spks_pred_test)
    return spks_pred_test, varexp_neurons


def peer_prediction(spks, xpos, ypos, dum=400, tbin=4):
    ineu1 = np.logical_xor((xpos % dum) < dum / 2, (ypos % dum) < dum / 2)
    # ineu1 = np.random.rand(len(spks)) > 0.5
    ineu2 = np.logical_not(ineu1)
    n_components = 128
    Vn = []
    for ineu in [ineu1, ineu2]:
        Vn.append(
            PCA(n_components=n_components, copy=False).fit_transform(spks[ineu].T)
        )
    varexp = np.zeros(2)
    varexp_neurons = np.zeros((spks.shape[0], 2))
    for k, ineu in enumerate([ineu1, ineu2]):
        V_pred_test, varexpk, itest = rrr_prediction(
            Vn[(k + 1) % 2],
            Vn[k % 2],
            rank=128,
            lam=1e-1,
            tbin=tbin,
        )[:3]
        varexp += varexpk[-1]
        U = spks[ineu] @ Vn[k]
        U /= (U**2).sum(axis=0) ** 0.5
        spks_pred_test = V_pred_test @ U.T
        spks_test = spks[ineu][:, itest].T
        varexp_neurons[ineu, 0] = compute_varexp(spks_test, spks_pred_test)
        spks_test_bin = bin1d(spks_test, tbin)
        spks_pred_test_bin = bin1d(spks_pred_test, tbin)
        varexp_neurons[ineu, 1] = compute_varexp(spks_test_bin, spks_pred_test_bin)
    # average variance explained for two halves
    varexp /= 2
    return varexp, varexp_neurons, itest


def split_batches(
    tcam, tneural, frac=0.25, pad=3, itrain=None, itest=None
):
    """this returns deterministic split of train and test in time chunks for neural and cam times

    Parameters
    ----------

    n_t : int
        number of timepoints to split

    tcam : 1D array
        times of camera frames

    tneural : 1D array
        times of neural frames

    frac : float (optional, default 0.25)
        fraction of points to put in test set

    pad : int (optional, default 3)
        number of timepoints to exclude from test set before and after training segment

    itrain: 2D int array
        times in train set, arranged in chunks

    itest: 2D int array
        times in test set, arranged in chunks


    Returns
    --------

    itrain: 1D int array
        times in train set, arranged in chunks

    itest: 1D int array
        times in test set, arranged in chunks

    itrain_cam: 2D int array
        times in cam frames in train set, arranged in chunks

    itest_cam: 2D int array
        times in cam frames in test set, arranged in chunks

    """

    if itrain is None or itest is None:
        itrain, itest = split_traintest(
            len(tneural), frac=frac, pad=pad
        )
    inds_train, inds_test = itrain[:, 0], itest[:, 0]
    l_train, l_test = itrain.shape[-1], itest.shape[-1]

    # find itrain and itest in cam inds
    f = interp1d(
        tcam,
        np.arange(0, len(tcam)),
        kind="nearest",
        axis=-1,
        fill_value="extrapolate",
        bounds_error=False,
    )

    inds_cam_train = f(tneural[inds_train]).astype("int")
    inds_cam_test = f(tneural[inds_test]).astype("int")

    l_cam_train = int(np.ceil(np.diff(tneural).mean() / np.diff(tcam).mean() * l_train))
    l_cam_test = int(np.ceil(np.diff(tneural).mean() / np.diff(tcam).mean() * l_test))

    # create itrain and itest in cam inds
    itrain_cam = inds_cam_train[:, np.newaxis] + np.arange(0, l_cam_train, 1, int)
    itest_cam = inds_cam_test[:, np.newaxis] + np.arange(0, l_cam_test, 1, int)

    itrain_cam = np.minimum(len(tcam) - 1, itrain_cam)
    itest_cam = np.minimum(len(tcam) - 1, itest_cam)

    # inds for downsampling itrain_cam and itest_cam
    itrain_sample = f(tneural[itrain.flatten()]).astype(int)
    itest_sample = f(tneural[itest.flatten()]).astype(int)

    # convert to indices in itrain_cam and itest_cam
    it = np.zeros(len(tcam), "bool")
    it[itrain_sample] = True
    itrain_sample = it[itrain_cam.flatten()].nonzero()[0]

    it = np.zeros(len(tcam), "bool")
    it[itest_sample] = True
    itest_sample = it[itest_cam.flatten()].nonzero()[0]

    return itrain, itest, itrain_cam, itest_cam, itrain_sample, itest_sample


def split_data(
    X,
    Y,
    tcam,
    tneural,
    frac=0.25,
    delay=-1,
    device=torch.device("cuda"),
):
    # ensure keypoints and timestamps are same length
    tc, ttot = len(tcam), len(X)
    inds = np.linspace(0, max(ttot, tc) - 1, min(ttot, tc)).astype(int)
    X = X[inds] if ttot > tc else X
    tcam = tcam[inds] if tc > ttot else tcam
    if delay < 0:
        Ys = np.vstack((Y[-delay:], np.tile(Y[[-1], :], (-delay, 1))))
        Xs = X
    elif delay > 0:
        Xs = np.vstack((X[delay:], np.tile(X[[-1], :], (delay, 1))))
        Ys = Y
    else:
        Xs = X
        Ys = Y
    splits = split_batches(tcam, tneural, frac=frac)
    itrain, itest, itrain_cam, itest_cam, itrain_sample, itest_sample = splits
    X_train = torch.from_numpy(Xs[itrain_cam]).float().to(device)
    Y_train = torch.from_numpy(Ys[itrain]).float().to(device)
    X_test = torch.from_numpy(Xs[itest_cam]).float().to(device)
    Y_test = torch.from_numpy(Ys[itest]).float().to(device).reshape(-1, Y.shape[-1])

    itrain_sample_b = torch.zeros(itrain_cam.size, dtype=bool, device=device)
    itrain_sample_b[itrain_sample] = True
    itest_sample_b = torch.zeros(itest_cam.size, dtype=bool, device=device)
    itest_sample_b[itest_sample] = True
    itrain_sample_b = itrain_sample_b.reshape(itrain_cam.shape)
    itest_sample_b = itest_sample_b.reshape(itest_cam.shape)

    itest -= delay

    return (
        X_train,
        X_test,
        Y_train,
        Y_test,
        itrain_sample_b,
        itest_sample_b,
        itrain_sample,
        itest_sample,
        itrain,
        itest,
    )


def train_model(
    model,
    X_dat,
    Y_dat,
    tcam_list,
    tneural_list,
    delay=-1,
    smoothing_penalty=0.5,
    n_iter=300,
    learning_rate=1e-3,
    annealing_steps=2,
    weight_decay=1e-4,
    device=torch.device("cuda"),
    verbose=False,
):
    """train behavior -> neural model using multiple animals"""

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    ### make input data a list if it's not already
    not_list = False
    if not isinstance(X_dat, list):
        not_list = True
        X_dat, Y_dat, tcam_list, tneural_list = (
            [X_dat],
            [Y_dat],
            [tcam_list],
            [tneural_list],
        )

    ### split data into train / test and concatenate
    arrs = [[], [], [], [], [], [], [], [], [], []]
    for i, (X, Y, tcam, tneural) in enumerate(
        zip(X_dat, Y_dat, tcam_list, tneural_list)
    ):
        dsplits = split_data(
            X, Y, tcam, tneural, delay=delay, device=device
        )
        for d, a in zip(dsplits, arrs):
            a.append(d)
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        itrain_sample_b,
        itest_sample_b,
        itrain_sample,
        itest_sample,
        itrain,
        itest,
    ) = arrs
    n_animals = len(X_train)

    tic = time.time()
    ### determine total number of batches across all animals to sample from
    n_batches = [0]
    n_batches.extend([X_train[i].shape[0] for i in range(n_animals)])
    n_batches = np.array(n_batches)
    c_batches = np.cumsum(n_batches)
    n_batches = n_batches.sum()

    anneal_epochs = n_iter - 50 * np.arange(1, annealing_steps + 1)

    ### optimize all parameters with SGD
    for epoch in range(n_iter):
        model.train()
        if epoch in anneal_epochs:
            if verbose:
                print("annealing learning rate")
            optimizer.param_groups[0]["lr"] /= 10.0
        np.random.seed(epoch)
        rperm = np.random.permutation(n_batches)
        train_loss = 0
        for nr in rperm:
            i = np.nonzero(nr >= c_batches)[0][-1]
            n = nr - c_batches[i]

            y_pred = model(
                X_train[i][n].unsqueeze(0), itrain_sample_b[i][n], animal_id=i
            )[0]
            loss = ((y_pred - Y_train[i][n].unsqueeze(0)) ** 2).mean()
            loss += (
                smoothing_penalty
                * (torch.diff(model.core.features[1].weight) ** 2).sum()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= n_batches

        # compute test loss and test variance explained
        if epoch % 20 == 0 or epoch == n_iter - 1:
            ve_all, y_pred_all = [], []
            model.eval()
            with torch.no_grad():
                pstr = f"epoch {epoch}, "
                for i in range(n_animals):
                    y_pred = model(X_test[i], itest_sample_b[i].flatten(), animal_id=i)[
                        0
                    ]
                    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
                    tl = ((y_pred - Y_test[i]) ** 2).mean()
                    ve = 1 - tl / ((Y_test[i] - Y_test[i].mean(axis=0)) ** 2).mean()
                    y_pred_all.append(y_pred.cpu().numpy())
                    ve_all.append(ve.item())
                    if n_animals == 1:
                        pstr += f"animal {i}, train loss {train_loss:.4f}, test loss {tl.item():.4f}, varexp {ve.item():.4f}, "
                    else:
                        pstr += f"varexp{i} {ve.item():.4f}, "
            pstr += f"time {time.time()-tic:.1f}s"
            if verbose:
                print(pstr)

    if not_list:
        return y_pred_all[0], ve_all[0], itest[0]
    else:
        return y_pred_all, ve_all, itest


def train_model_test(
    model,
    X,
    Y,
    tcam,
    tneural,
    sgd=False,
    lam=1e-3,
    n_iter=600,
    learning_rate=5e-4,
    fix_model=True,
    smoothing_penalty=1.0,
    weight_decay=1e-4,
    device=torch.device("cuda"),
):

    dsplits = split_data(X, Y, tcam, tneural, device=device)
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        itrain_sample_b,
        itest_sample_b,
        itrain_sample,
        itest_sample,
        itrain,
        itest,
    ) = dsplits

    tic = time.time()

    n_batches = X_train.shape[0]
    if sgd:
        model.train()
        if fix_model:
            for param in model.parameters():
                param.requires_grad = False
            model.test_classifier.weight.requires_grad = True
            model.test_classifier.bias.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        for epoch in range(n_iter):
            model.train()
            np.random.seed(epoch)
            rperm = np.random.permutation(n_batches)
            train_loss = 0
            for n in rperm:
                y_pred, latents = model(
                    X_train[n].unsqueeze(0), itrain_sample_b[n], test=True
                )
                loss = ((y_pred - Y_train[n].unsqueeze(0)) ** 2).mean()
                loss += (
                    smoothing_penalty
                    * (torch.diff(model.features[1].weight) ** 2).sum()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= n_batches
            if epoch % 20 == 0 or epoch == n_iter - 1:
                ve_all = []
                y_pred_all = []
                with torch.no_grad():
                    model.eval()
                    pstr = f"epoch {epoch}, "
                    y_pred = model(X_test, itest_sample_b.flatten(), test=True)[0]
                    tl = ((y_pred - Y_test) ** 2).mean()
                    ve = 1 - tl / (Y_test**2).mean()
                    # ve = ve.item()
                    y_pred = y_pred.cpu().numpy()
                    # y_pred_all.append(y_pred.cpu().numpy())
                    # ve_all.append(ve.item())
                    pstr += f"train loss {train_loss:.4f}, test loss {tl.item():.4f}, varexp {ve.item():.4f}"
                    print(pstr)

    else:
        itrain = itrain.reshape(n_batches, -1)
        l_train = itrain.shape[-1]
        with torch.no_grad():
            model.eval()
            for n in range(n_batches):
                y_pred, latents = model(
                    X_train[n].unsqueeze(0), itrain_sample_b[n], test=True
                )
                if n == 0:
                    n_latents = latents.shape[-1]
                    latents_train = np.ones((itrain.size, n_latents + 1), "float32")
                latents_train[
                    n * l_train : (n + 1) * l_train, :n_latents
                ] = latents.cpu().numpy()
            latents_test = np.ones((itest.size, n_latents + 1), "float32")
            latents_test[:, :n_latents] = (
                model(X_test, itest_sample_b.flatten(), test=True)[1]
                .cpu()
                .numpy()
                .reshape(-1, n_latents)
            )

            Y_train = Y_train.cpu().numpy()
            Y_test = Y_test.cpu().numpy().reshape(-1, Y_test.shape[-1])
            Y_train = Y_train.reshape(-1, Y_train.shape[-1])
            A = np.linalg.solve(
                latents_train.T @ latents_train + lam * np.eye(n_latents + 1),
                latents_train.T @ Y_train,
            )
            y_pred = latents_test @ A
            tl = ((y_pred - Y_test) ** 2).mean()
            ve = 1 - tl / (Y_test**2).mean()
            model.test_classifier.weight.data = (
                torch.from_numpy(A[:n_latents].T).float().to(device)
            )
            model.test_classifier.bias.data = torch.from_numpy(A[-1]).float().to(device)
            print(ve)

    return y_pred, ve, itest


def KLDiv_discrete(P, Q, binsize=200):
    # Q is the null distribution; P and Q are 2D distributions

    x_bins = np.append(np.arange(0, np.amax(P[:, 0]), binsize), np.amax(P[:, 0]))
    y_bins = np.append(np.arange(0, np.amax(P[:, 1]), binsize), np.amax(P[:, 1]))

    this_KL = 0
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            Qx = (
                np.sum(
                    (Q[:, 0] >= x_bins[i])
                    & (Q[:, 0] < x_bins[i + 1])
                    & (Q[:, 1] >= y_bins[j])
                    & (Q[:, 1] < y_bins[j + 1])
                )
            ) / len(Q)
            Px = (
                np.sum(
                    (P[:, 0] >= x_bins[i])
                    & (P[:, 0] < x_bins[i + 1])
                    & (P[:, 1] >= y_bins[j])
                    & (P[:, 1] < y_bins[j + 1])
                )
            ) / len(P)
            if (Px == 0) | (
                Qx == 0
            ):  # no points in test or null distrib -- can't have log(0), or /0
                continue

            this_KL += Px * np.log(Px / Qx)

    return this_KL


def causal_filter(X, swave, tlag, remove_start=False, device=torch.device("cuda")):
    """filter matrix X (n_channels, (n_batches,) n_time) with filters swave

    returns Xfilt (n_out, n_batches*n_time)
    """
    if X.ndim < 3:
        X = X.unsqueeze(1)
    NT = X.shape[-1]
    nt = swave.shape[1]
    # reshape X for input to be (n_channels*n_batches, 1, n_time)
    Xfilt = conv1d(
        X.reshape(-1, X.shape[-1]).unsqueeze(1), swave.unsqueeze(1), padding=nt + tlag
    )
    Xfilt = Xfilt[..., :NT]
    Xfilt = Xfilt[..., nt:] if remove_start else Xfilt
    Xfilt = Xfilt.reshape(X.shape[0], X.shape[1], swave.shape[0], -1)
    Xfilt = Xfilt.permute(0, 2, 1, 3)
    Xfilt = Xfilt.reshape(X.shape[0] * swave.shape[0], X.shape[1], -1)
    return Xfilt


def fit_causal_prediction(
    X_train, X_test, swave, lam=1e-3, tlag=1, device=torch.device("cuda")
):
    """predict X in the future with exponential filters"""
    # fit on train data
    Xfilt = causal_filter(X_train, swave, tlag)
    Xfilt = Xfilt.reshape(Xfilt.shape[0], -1)
    NT = X_train.shape[1] * X_train.shape[2]
    nff = Xfilt.shape[0]
    CC = (Xfilt @ Xfilt.T) / NT + lam * torch.eye(nff, device=device)
    CX = (Xfilt @ X_train.reshape(-1, NT).T) / NT
    B = torch.linalg.solve(CC, CX)

    # performance on test data
    Xfilt = causal_filter(X_test, swave, tlag, remove_start=True)
    Xfilt = Xfilt.reshape(Xfilt.shape[0], -1)
    ypred = B.T @ Xfilt
    nt = swave.shape[1]
    ve = compute_varexp(X_test[:, :, nt:].reshape(X_test.shape[0], -1).T, ypred.T)
    return ve, ypred, B


def future_prediction(X, Ball, swave, device=torch.device("cuda")):
    """create future prediction"""
    tlag = Ball.shape[-1]
    Xfilt = causal_filter(X, swave, tlag, remove_start=True)
    vef = np.zeros((X.shape[0], tlag))
    nt = swave.shape[1]
    Xpred = np.zeros((X.shape[0], X.shape[1], X.shape[2] - nt, tlag))
    for k in range(tlag):
        Xfilt0 = Xfilt[:, :, tlag - k :].reshape(Xfilt.shape[0], -1)
        B = torch.from_numpy(Ball[:, :, k]).to(device)
        ypred = B.T @ Xfilt0
        ve = compute_varexp(
            X[:, :, nt : -(tlag - k)].reshape(X.shape[0], -1).T, ypred.T
        )
        ypred = ypred.reshape(X.shape[0], X.shape[1], -1)
        vef[:, k] = ve.cpu().numpy()
        Xpred[:, :, : -(tlag - k), k] = ypred.cpu().numpy()
    return vef, Xpred


def predict_future(
    x, keypoint_labels=None, get_future=True, lam=1e-2, device=torch.device("cuda")
):
    """predict keypoints or latents in future

    x is (n_time, n_keypoints) and z-scored per keypoint

    """
    nt = 128
    sigs = torch.FloatTensor(2 ** np.arange(0, 8, 1)).unsqueeze(-1)
    swave = torch.exp(-torch.arange(nt) / sigs).to(device)
    swave = torch.flip(swave, [1])
    swave = swave / (swave**2).sum(1, keepdim=True) ** 0.5

    tlags = np.arange(1, 501, 1)
    tlags = np.append(tlags, np.arange(525, 2000, 25))

    X = torch.from_numpy(x.T).float().to(device)

    itrain, itest = split_traintest(len(x), frac=0.25, pad=nt)

    X_train = X[:, itrain]
    X_test = X[:, itest]

    n_kp = X_train.shape[0]
    n_tlags = len(tlags)
    vet = np.zeros((n_kp, n_tlags), "float32")
    Ball = np.zeros((swave.shape[0] * n_kp, n_kp, n_tlags), "float32")
    for k, tlag in enumerate(tlags):
        ve, ypred, B = fit_causal_prediction(X_train, X_test, swave, tlag=tlag, lam=lam)
        vet[:, k] = ve.cpu().numpy()
        Ball[:, :, k] = B.cpu().numpy()

    if get_future:
        vef, ypred = future_prediction(X_test, Ball[:, :, :500], swave)
    else:
        ypred = None

    if keypoint_labels is not None:
        # tile for X and Y
        kp_labels = np.tile(np.array(keypoint_labels)[:, np.newaxis], (1, 2)).flatten()

        areas = ["eye", "whisker", "nose"]
        vet_area = np.zeros((len(areas), vet.shape[1]))
        for j in range(len(areas)):
            ak = np.array(
                [k for k in range(len(kp_labels)) if areas[j] in kp_labels[k]]
            )
            vet_area[j] = vet[ak].mean(axis=0)
    else:
        vet_area = None

    return vet, vet_area, tlags, ypred, itest[:, nt:]
