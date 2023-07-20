"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os
import time

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from torch.nn.functional import conv1d

from facemap.utils import bin1d, compute_varexp, split_traintest

from .neural_model import KeypointsNetwork


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


def reduced_rank_regression(X, Y, rank=None, lam=0, device=torch.device("cuda")):
    """predict Y from X using regularized reduced rank regression

    *** subtract mean from X and Y before predicting

    if rank is None, returns A and B of full-rank (minus one) prediction

    Prediction:
    >>> Y_pred = X @ B @ A.T

    Parameters
    ----------

    X : 2D array, input data, float32 torch tensor (n_samples, n_features)

    Y : 2D array, data to predict, float32 torch tensor (n_samples, n_predictors)

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
    CXX = (X.T @ X + lam * torch.eye(X.shape[1], device=device)) / X.shape[0]
    CYX = (Y.T @ X) / X.shape[0]

    # compute inverse square root of matrix
    # s, u = eigh(CXX.cpu().numpy())
    u, s = torch.svd_lowrank(CXX, q=rank)[:2]
    CXXMH = (u * (s + lam) ** -0.5) @ u.T

    # project into prediction space
    M = CYX @ CXXMH
    # do svd of prediction projection
    # model = PCA(n_components=rank).fit(M)
    # c = model.components_.T
    # s = model.singular_values_
    s, c = torch.svd_lowrank(M, q=rank)[1:]
    A = M @ c
    B = CXXMH @ c
    return A, B


def rrr_prediction(
    X,
    Y,
    rank=None,
    lam=0,
    itrain=None,
    itest=None,
    tbin=None,
    device=torch.device("cuda"),
):
    """predict Y from X using regularized reduced rank regression for all ranks up to "rank"

    *** subtract mean from X and Y before predicting

    if rank is None, returns A and B of full-rank (minus one) prediction

    Prediction:
    >>> Y_pred_test = X_test @ B @ A.T

    Parameters
    ----------

    X : 2D array, input data, float32 (n_samples, n_features)

    Y : 2D array, data to predict, float32 (n_samples, n_predictors)

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
    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)
    A, B = reduced_rank_regression(
        X[itrain], Y[itrain], rank=rank, lam=lam, device=device
    )
    min_dim = min(Y.shape[1], min(X.shape[0], X.shape[1])) - 1
    if rank is None:
        rank = min_dim
    else:
        rank = min(min_dim, rank)
    corrf = np.zeros((rank, n_feats))
    varexpf = np.zeros((rank, n_feats))
    varexp = (
        np.zeros((rank, 2)) if (tbin is not None and tbin > 1) else np.zeros((rank, 1))
    )
    Y_pred_test = np.zeros((len(itest), n_feats))
    for r in range(rank):
        Y_pred_test = X[itest] @ B[:, : r + 1] @ A[:, : r + 1].T
        Y_test_var = (Y[itest] ** 2).mean(axis=0)
        corrf[r] = (
            (
                (Y[itest] * Y_pred_test).mean(axis=0)
                / (Y_test_var**0.5 * Y_pred_test.std(axis=0))
            )
            .cpu()
            .numpy()
        )
        residual = ((Y[itest] - Y_pred_test) ** 2).mean(axis=0)
        varexpf[r] = (1 - residual / Y_test_var).cpu().numpy()
        varexp[r, 0] = (1 - residual.mean() / Y_test_var.mean()).cpu().numpy()
        if tbin is not None and tbin > 1:
            varexp[r, 1] = (
                compute_varexp(
                    bin1d(Y[itest], tbin).flatten(), bin1d(Y_pred_test, tbin).flatten()
                )
                .cpu()
                .numpy()
            )
    return (
        Y_pred_test.cpu().numpy(),
        varexp.squeeze(),
        itest,
        A.cpu().numpy(),
        B.cpu().numpy(),
        varexpf.squeeze(),
        corrf.squeeze(),
    )


def rrr_prediction_resample(X, Y, tcam, tneural, delay=-1, rank=32, lam=1e-6):
    """predict neurons or neural PCs from behavior X at times tcam, tneural"""
    X_ds = resample_data(X, tcam, tneural, crop="linspace")
    if delay < 0:
        Ys = np.vstack((Y[-delay:], np.tile(Y[[-1], :], (-delay, 1))))
    else:
        X_ds = np.vstack((X_ds[delay:], np.tile(X_ds[[-1], :], (delay, 1))))
        Ys = Y
    Y_pred_test, ve_test, itest, A, B = rrr_prediction(
        X_ds.astype("float32"), Ys.astype("float32"), rank=rank, lam=lam
    )[:5]
    return Y_pred_test, ve_test, itest, A, B


def rrr_varexp(X0, Y, tcam, tneural, U=None, spks=None, delay=-1, rank=32, lam=1e-6):
    """predict neural PCs Y and compute varexp for PCs and spks"""
    varexp = np.nan * np.zeros((rank, len(X0)))
    varexp_neurons = (
        np.nan * np.zeros((len(spks), len(X0)))
        if spks is not None
        else np.nan * np.zeros((Y.shape[-1], len(X0)))
    )
    spks_pred_test0 = []
    Y_pred_test0 = []
    for k in range(len(X0)):
        X = X0[k]

        X -= X.mean(axis=0)
        X /= X[:, 0].std(axis=0)

        Y_pred_test, ve_test, itest, A, B = rrr_prediction_resample(
            X, Y, tcam, tneural, delay=delay, rank=rank, lam=lam
        )
        varexp[: len(ve_test), k] = ve_test

        itest -= delay
        # single neuron prediction
        spks_pred_test = Y_pred_test @ U.T if spks is not None else Y_pred_test
        spks_test = spks[:, itest].T if spks is not None else Y[itest]
        varexp_neurons[:, k] = compute_varexp(spks_test, spks_pred_test)
        spks_pred_test0.append(spks_pred_test)
        Y_pred_test0.append(Y_pred_test)
    return (
        varexp.squeeze(),
        varexp_neurons.squeeze(),
        np.array(Y_pred_test0).squeeze(),
        np.array(spks_pred_test0).squeeze(),
        itest,
    )


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
    Vt = model.transform(neural_activity)
    neural_pcs = U.T * S
    return neural_pcs, Vt


def get_pca_inverse_transform(pca_data, components):
    """
    Perform inverse PCA transform on data
    Parameters
    ----------
    pca_data : 2D-array
        neural PCs of shape [n_features x n_comps]
    components : 2D-array
        neural PCs of shape [n_samples x n_comps]
    Returns
    -------
    data : 2D-array
        data of shape [n_samples x n_features]
    """
    data = (pca_data @ components.T).T
    return data


def get_keypoints_to_neural_varexp(
    input_data,
    target_neural_data,
    behavior_timestamps,
    neural_timestamps,
    spks=None,
    U=None,
    delay=-1,
    compute_latents=False,
    verbose=False,
    n_iter=300,
    learning_rate=1e-3,
    weight_decay=1e-4,
    gui=None,
    GUIobject=None,
    device=torch.device("cuda"),
):
    """
    Get variance explained of neural PCs prediction by keypoints
    (or variance explained of neural spikes prediction by keypoints)
    Parameters
    ----------
    input_data : 2D-array
        input data of shape [n_features, time]
    target_neural_data : 2D array
        neural data of shape [neurons x time] which can represent neural PCs or neural spikes
    behavior_timestamps : 1D array
        timestamps of behavior data for each frame
    neural_timestamps : 1D array
        timestamps of neural data for each frame
    delay : int, optional
        number of frames to delay neural data, by default -1
    running : array, optional
        1D running trace to include in the prediction model
    verbose : bool, optional
        print progress, by default False
    varexp_per_keypoint : bool, optional
        return variance explained per keypoint, by default False and returns variance explained for all keypoints
    device : torch.device, optional
        device to use for prediction, by default torch.device("cuda")
    Returns
    -------
    varexp : float
        variance explained of neural PCs prediction by keypoints
    model : torch.nn.Module
        model used for prediction
    itest : 1D array
        indices of test data
    """
    x = input_data
    y = target_neural_data

    # Initialize model for keypoints to neural prediction
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = KeypointsNetwork(n_in=x.shape[-1], n_out=y.shape[-1]).to(device)

    # Train model and get predictions
    print("Inputs: ", x.shape)
    print("Neural: ", y.shape)
    print("Using lr, n_iter, weight_decay: ", learning_rate, n_iter, weight_decay)

    (
        y_pred_test,
        varexp,
        spks_pred_test,
        varexp_neurons,
        test_indices,
    ) = model.train_model(
        x,
        y,
        behavior_timestamps,
        neural_timestamps,
        U=U,
        spks=spks,
        delay=delay,
        n_iter=n_iter,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        verbose=verbose,
        gui=gui,
        GUIobject=GUIobject,
        device=device,
    )
    if compute_latents:
        latents = get_trained_model_predictions(
            x, model, behavior_timestamps, neural_timestamps, device=device
        )[-1]
    else:
        latents = None
    y_pred_test = y_pred_test.reshape(-1, y.shape[-1])
    print(f"all kp, varexp {varexp:.3f}")
    if spks is not None:
        print(f"neuron varexp = {varexp_neurons.mean():.3f}")
    else:
        varexp_neurons, spks_pred_test = None, None

    return (
        varexp,
        varexp_neurons,
        y_pred_test,
        spks_pred_test,
        test_indices,
        latents,
        model,
    )


def get_trained_model_predictions(
    keypoints,
    model,
    behavior_timestamps,
    neural_timestamps,
    device=torch.device("cuda"),
):
    """
    Get prediction from keypoints using a trained model
    Parameters
    ----------
    keypoints :
        array of shape (n_frames, n_keypoints, 2)
    model : torch.nn.Module
        an instance of KeypointsNetwork model already trained
    device : torch.device, optional
        device to use for prediction, by default torch.device("cuda")
    Returns
    -------
    prediction : ND-array
        prediction from the model
    """
    num_timepoints = keypoints.shape[0]
    pred_data = np.zeros((num_timepoints, model.readout.n_out), "float32")
    batch_size = 10000
    latents = np.zeros(
        (num_timepoints, model.core.features.latent[0].weight.shape[0]), "float32"
    )
    n_batches = int(np.ceil(num_timepoints / batch_size))

    with torch.no_grad():
        model.eval()
        for n in range(n_batches):
            x_batch = keypoints[
                n * batch_size : min(num_timepoints, (n + 1) * batch_size)
            ].astype("float32")
            x_batch = torch.from_numpy(x_batch).to(device)
            y_batch, l_batch = model(x_batch.unsqueeze(0))
            pred_data[
                n * batch_size : min(num_timepoints, (n + 1) * batch_size)
            ] = y_batch.cpu().numpy()
            latents[
                n * batch_size : min(num_timepoints, (n + 1) * batch_size)
            ] = l_batch.cpu().numpy()
    f = interp1d(
        behavior_timestamps,
        np.arange(0, len(behavior_timestamps)),
        kind="nearest",
        fill_value="extrapolate",
        bounds_error=False,
    )
    sample_inds = np.round(f(neural_timestamps)).astype(int)
    pred_data = pred_data[sample_inds]
    return pred_data, latents


def resample_data_to_neural_timestamps(data, behavior_timestamps, neural_timestamps):
    """
    Resample data to neural timestamps using linear interpolation
    Parameters
    ----------
    data : 2D array
    behavior_timestamps : 1D array
        timestamps of behavior data for each frame
    neural_timestamps : 1D array
        timestamps of neural data for each frame
    Returns
    -------
    data_resampled : 2D array
        Resampled data
    """
    f = interp1d(behavior_timestamps, np.arange(0, len(behavior_timestamps)))
    sample_inds = np.round(f(neural_timestamps)).astype(int)
    data_resampled = data[sample_inds]
    return data_resampled


def peer_prediction(spks, xpos, ypos, dum=400):
    ineu1 = np.logical_xor((xpos % dum) < dum / 2, (ypos % dum) < dum / 2)
    # ineu1 = np.random.rand(len(spks)) > 0.5
    ineu2 = np.logical_not(ineu1)
    n_components = 128
    Vn = []
    for ineu in [ineu1, ineu2]:
        Vn.append(
            PCA(n_components=n_components, copy=False).fit_transform(spks[ineu].T)
        )
    varexp = 0
    varexp_neurons = np.zeros((spks.shape[0]))
    for k, ineu in enumerate([ineu1, ineu2]):
        V_pred_test, varexpk, itest = rrr_prediction(
            Vn[(k + 1) % 2], Vn[k % 2], rank=128, lam=1e-1
        )[:3]
        varexp += varexpk[-1]
        U = spks[ineu] @ Vn[k]
        U /= (U**2).sum(axis=0) ** 0.5
        spks_pred_test = V_pred_test @ U.T
        spks_test = spks[ineu][:, itest].T
        varexp_neurons[ineu] = compute_varexp(spks_test, spks_pred_test)

    # average variance explained for two halves
    varexp /= 2
    return varexp, varexp_neurons, itest


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
    Xfilt = causal_filter(X_train, swave, tlag, device=device)
    Xfilt = Xfilt.reshape(Xfilt.shape[0], -1)
    NT = X_train.shape[1] * X_train.shape[2]
    nff = Xfilt.shape[0]
    CC = (Xfilt @ Xfilt.T) / NT + lam * torch.eye(nff, device=device)
    CX = (Xfilt @ X_train.reshape(-1, NT).T) / NT
    B = torch.linalg.solve(CC, CX)

    # performance on test data
    Xfilt = causal_filter(X_test, swave, tlag, remove_start=True, device=device)
    Xfilt = Xfilt.reshape(Xfilt.shape[0], -1)
    ypred = B.T @ Xfilt
    nt = swave.shape[1]
    ve = compute_varexp(X_test[:, :, nt:].reshape(X_test.shape[0], -1).T, ypred.T)
    return ve, ypred, B


def future_prediction(X, Ball, swave, device=torch.device("cuda")):
    """create future prediction"""
    tlag = Ball.shape[-1]
    Xfilt = causal_filter(X, swave, tlag, remove_start=True, device=device)
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
        ve, ypred, B = fit_causal_prediction(
            X_train, X_test, swave, tlag=tlag, lam=lam, device=device
        )
        vet[:, k] = ve.cpu().numpy()
        Ball[:, :, k] = B.cpu().numpy()

    if get_future:
        vef, ypred = future_prediction(X_test, Ball[:, :, :500], swave, device=device)
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
