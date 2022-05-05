import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA


def update_mainwindow_progressbar(MainWindow, GUIobject, s, prompt):
    if MainWindow is not None and GUIobject is not None:
        message = s.getvalue().split("\x1b[A\n\r")[0].split("\r")[-1]
        MainWindow.update_status_bar(
            prompt + message, update_progress=True, hide_progress=False
        )
        GUIobject.QApplication.processEvents()


def update_mainwindow_message(MainWindow, GUIobject, prompt, hide_progress=True):
    if MainWindow is not None and GUIobject is not None:
        MainWindow.update_status_bar(
            prompt, update_progress=False, hide_progress=hide_progress
        )
        GUIobject.QApplication.processEvents()


def bin1d(X, tbin):
    """bin over first axis of data with bin tbin"""
    size = list(X.shape)
    X = X[: size[0] // tbin * tbin].reshape((size[0] // tbin, tbin, -1)).mean(axis=1)
    size[0] = X.shape[0]
    return X.reshape(size)


def split_testtrain(n_t, frac=0.25):
    """this returns indices of testing data and training data"""
    n_segs = int(
        min(20, n_t / 4)
    )  # usu want 20 segs, but might not have enough frames for that
    n_len = int(n_t / n_segs)
    ninds = np.linspace(0, n_t - n_len, n_segs).astype(int)
    itest = (ninds[:, np.newaxis] + np.arange(0, n_len * frac, 1, int)).flatten()
    itrain = np.ones(n_t, np.bool)
    itrain[itest] = 0
    return itest, itrain


def get_frame(cframe, nframes, cumframes, containers):
    cframe = np.maximum(0, np.minimum(nframes - 1, cframe))
    cframe = int(cframe)
    try:
        ivid = (cumframes < cframe).nonzero()[0][-1]
    except:
        ivid = 0
    img = []
    for vs in containers[ivid]:
        frame_ind = cframe - cumframes[ivid]
        capture = vs
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_ind:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
        ret, frame = capture.read()
        if ret:
            img.append(frame)
        else:
            print("Error reading frame")
    return img


def load_images_from_video(video_path, selected_frame_ind):
    """
    Load images from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    for frame_ind in selected_frame_ind:
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != frame_ind:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print("Error reading frame")
    frames = np.array(frames)
    return frames


def rrr_prediction(X, Y, rank=None, lam=0, itrain=None, itest=None):
    """predict Y from X using regularized reduced rank regression

    returns prediction accuracy on test data + model params

    """
    n_t, n_feats = Y.shape
    if itrain is None and itest is None:
        itest, itrain = split_testtrain(n_t)
    A, B = reduced_rank_regression(X[itrain], Y[itrain], rank=rank, lam=lam)
    rank = A.shape[1]
    corrf = np.zeros((rank, n_feats))
    varexpf = np.zeros((rank, n_feats))
    varexp = np.zeros(rank)
    for r in range(rank):
        Y_pred_test = X[itest] @ B[:, : r + 1] @ A[:, : r + 1].T
        Y_test_var = (Y[itest] ** 2).mean(axis=0)
        corrf[r] = (Y[itest] * Y_pred_test).mean(axis=0) / (
            Y_test_var**0.5 * Y_pred_test.std(axis=0)
        )
        residual = ((Y[itest] - Y_pred_test) ** 2).mean(axis=0)
        varexpf[r] = 1 - residual / Y_test_var
        varexp[r] = 1 - residual.mean() / Y_test_var.mean()

    return A, B, varexp, varexpf, corrf


def rrr_ridge_prediction(X, Y, B, lam=0):
    """predict Y from X @ B using ridge regression

    B is obtained from rrr

    returns prediction accuracy on test data + model params

    """
    n_t, n_feats = Y.shape
    itest, itrain = split_testtrain(n_t)
    rank = B.shape[1]
    corrf = np.zeros((rank, n_feats))
    varexpf = np.zeros((rank, n_feats))
    varexp = np.zeros(rank)
    for r in range(rank):
        A = ridge_regression(X[itrain] @ B[:, : r + 1], Y[itrain], lam=lam)
        Y_pred_test = X[itest] @ B[:, : r + 1] @ A
        Y_test_var = (Y[itest] ** 2).mean(axis=0)
        corrf[r] = (Y[itest] * Y_pred_test).mean(axis=0) / (
            Y_test_var**0.5 * Y_pred_test.std(axis=0)
        )
        residual = ((Y[itest] - Y_pred_test) ** 2).mean(axis=0)
        varexpf[r] = 1 - residual / Y_test_var
        varexp[r] = 1 - residual.mean() / Y_test_var.mean()

    return A, varexp, varexpf, corrf


def ridge_regression(X, Y, lam=0):
    """predict Y from X using regularized reduced rank regression

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


def resample_frames(data, torig, tout):
    """
    Resample data at times torig at times tout.
    data is components x time. The data is filtered using a gaussian filter before resampling.
    Parameters
    ----------
    data : ND-array
        Data to resample
    torig : 1D-array
        Original times
    tout : 1D-array
        Times to resample to
    """
    fs = torig.size / tout.size  # relative sampling rate
    data = gaussian_filter1d(data, np.ceil(fs / 4), axis=1)
    f = interp1d(torig, data, kind="linear", axis=-1, fill_value="extrapolate")
    dout = f(tout)
    return dout


def resample_data(data, data_timestamps, target_timestamps):
    """
    Resample data to a new time base.
    Parameters
    ----------
    data : ND-array
        Data to be resampled.
    data_timestamps : 1D-array
        Timestamps of the data.
    target_timestamps : 1D-array
        Target timestamps for resampling the data.
    Returns
    -------
    data_resampled : ND-array
        Resampled data.
    """
    # Estimate the interpolation function for the data
    f = interp1d(
        data_timestamps.squeeze(),
        data,
        kind="linear",
        axis=-1,
        fill_value="extrapolate",
    )
    # Resample the data
    resampled_data = f(target_timestamps)
    return resampled_data.squeeze()


def resample_timestamps(init_timestamps, target_timestamps):
    """
    Resample timestamps to a new time base.
    Parameters
    ----------
    init_timestamps : 1D-array
        Timestamps of the data.
    target_timestamps : 1D-array
        Target timestamps for resampling the data.
    Returns
    -------
    resampled_timestamps : 1D-array
        Resampled timestamps.
    """
    # Estimate the interpolation function for the data
    f = interp1d(
        init_timestamps.squeeze(),
        np.arange(init_timestamps.size),
        kind="linear",
        axis=-1,
        fill_value="extrapolate",
    )
    # Resample the data
    resampled_timestamps = f(target_timestamps)
    return resampled_timestamps.squeeze()


def get_frames(imall, containers, cframes, cumframes):
    nframes = cumframes[-1]  # total number of frames
    cframes = np.maximum(0, np.minimum(nframes - 1, cframes))
    cframes = np.arange(cframes[0], cframes[-1] + 1).astype(int)
    # Check frames exist in which video (for multiple videos, one view)
    ivids = (cframes[np.newaxis, :] >= cumframes[1:, np.newaxis]).sum(axis=0)

    for ii in range(len(containers[0])):  # for each view in the list
        nk = 0
        for n in np.unique(ivids):
            cfr = cframes[ivids == n]
            start = cfr[0] - cumframes[n]
            end = cfr[-1] - cumframes[n] + 1
            nt0 = end - start
            capture = containers[n][ii]
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != start:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            fc = 0
            ret = True
            while fc < nt0 and ret:
                ret, frame = capture.read()
                if ret:
                    imall[ii][nk + fc] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    print("img load failed, replacing with prev..")
                    imall[ii][nk + fc] = imall[ii][nk + fc - 1]
                fc += 1
            nk += nt0

    if nk < imall[0].shape[0]:
        for ii, im in enumerate(imall):
            imall[ii] = im[:nk].copy()


def close_videos(containers):
    """Method is called to close all videos/containers open for reading
    using openCV.
    Parameters:-(Input) containers: a 2D list of pointers to videos captured by openCV
                (Output) N/A"""
    for i in range(len(containers)):  # for each video in the list
        for j in range(len((containers[0]))):  # for each cam/view
            cap = containers[i][j]
            cap.release()


def get_frame_details(filenames):
    """
    Uses cv2 to open video files and obtain their details
    Parameters:-(Input) filenames: a 2D list of video files
                (Output) cumframes: list of total frame size for each cam/view
                (Output) Ly: list of dimension x for each cam/view
                (Output) Lx: list of dimension y for each cam/view
                (Output) containers: a 2D list of pointers to videos that are open
    """
    cumframes = [0]
    containers = []
    for fs in filenames:  # for each video in the list
        Ly = []
        Lx = []
        cs = []
        for n, f in enumerate(fs):  # for each cam/view
            cap = cv2.VideoCapture(f)
            cs.append(cap)
            framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            Lx.append(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            Ly.append(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        containers.append(cs)
        cumframes.append(cumframes[-1] + framecount)
    cumframes = np.array(cumframes).astype(int)
    return cumframes, Ly, Lx, containers


def get_cap_features(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return fps, nframes


def get_skipping_frames(imall, filenames, cframes, cumframes):
    nframes = cumframes[-1]  # total number of frames
    cframes = np.maximum(0, np.minimum(nframes - 1, cframes))
    cframes = np.arange(cframes[0], cframes[-1] + 1).astype(int)
    ivids = (cframes[np.newaxis, :] >= cumframes[1:, np.newaxis]).sum(axis=0)
    i = 0
    for ii in range(len(filenames[0])):
        for n in np.unique(ivids):
            cfr = cframes[ivids == n]
            ifr = cfr - cumframes[n]
            capture = cv2.VideoCapture(filenames[n][ii])
            for iframe in ifr:
                capture.set(cv2.CAP_PROP_POS_FRAMES, iframe)
                ret, frame = capture.read()
                if ret:
                    imall[ii][i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    i += 1
                else:
                    break
            capture.release()


def multivideo_reshape(X, LY, LX, sy, sx, Ly, Lx, iinds):
    """reshape X matrix pixels x n matrix into LY x LX - embed each video at sy, sx"""
    """ iinds are indices of each video in concatenated array"""
    X_reshape = np.zeros((LY, LX, X.shape[-1]), np.float32)
    for i in range(len(Ly)):
        X_reshape[sy[i] : sy[i] + Ly[i], sx[i] : sx[i] + Lx[i]] = np.reshape(
            X[iinds[i]], (Ly[i], Lx[i], X.shape[-1])
        )
    return X_reshape


def roi_to_dict(ROIs, rROI=None):
    rois = []
    for i, r in enumerate(ROIs):
        rois.append(
            {
                "rind": r.rind,
                "rtype": r.rtype,
                "iROI": r.iROI,
                "ivid": r.ivid,
                "color": r.color,
                "yrange": r.yrange,
                "xrange": r.xrange,
                "saturation": r.saturation,
            }
        )
        if hasattr(r, "pupil_sigma"):
            rois[i]["pupil_sigma"] = r.pupil_sigma
        if hasattr(r, "ellipse"):
            rois[i]["ellipse"] = r.ellipse
        if rROI is not None:
            if len(rROI[i]) > 0:
                rois[i]["reflector"] = []
                for rr in rROI[i]:
                    rdict = {
                        "yrange": rr.yrange,
                        "xrange": rr.xrange,
                        "ellipse": rr.ellipse,
                    }
                    rois[i]["reflector"].append(rdict)

    return rois


def get_reflector(yrange, xrange, rROI=None, rdict=None):
    reflectors = np.zeros((yrange.size, xrange.size), np.bool)
    if rROI is not None and len(rROI) > 0:
        for r in rROI:
            ellipse, ryrange, rxrange = (
                r.ellipse.copy(),
                r.yrange.copy(),
                r.xrange.copy(),
            )
            ix = np.logical_and(rxrange >= 0, rxrange < xrange.size)
            ellipse = ellipse[:, ix]
            rxrange = rxrange[ix]
            iy = np.logical_and(ryrange >= 0, ryrange < yrange.size)
            ellipse = ellipse[iy, :]
            ryrange = ryrange[iy]
            reflectors[np.ix_(ryrange, rxrange)] = np.logical_or(
                reflectors[np.ix_(ryrange, rxrange)], ellipse
            )
    elif rdict is not None and len(rdict) > 0:
        for r in rdict:
            ellipse, ryrange, rxrange = (
                r["ellipse"].copy(),
                r["yrange"].copy(),
                r["xrange"].copy(),
            )
            ix = np.logical_and(rxrange >= 0, rxrange < xrange.size)
            ellipse = ellipse[:, ix]
            rxrange = rxrange[ix]
            iy = np.logical_and(ryrange >= 0, ryrange < yrange.size)
            ellipse = ellipse[iy, :]
            ryrange = ryrange[iy]
            reflectors[np.ix_(ryrange, rxrange)] = np.logical_or(
                reflectors[np.ix_(ryrange, rxrange)], ellipse
            )
    return reflectors.nonzero()


def video_placement(Ly, Lx):
    """Ly and Lx are lists of video sizes"""
    npix = Ly * Lx
    picked = np.zeros((Ly.size,), np.bool)
    ly = 0
    lx = 0
    sy = np.zeros(Ly.shape, int)
    sx = np.zeros(Lx.shape, int)
    if Ly.size == 2:
        gridy = 1
        gridx = 2
    elif Ly.size == 3:
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
        if iy == 0:
            ly = 0
            rowmax = 0
        if ix == 0:
            lx = 0
        sy[imax] = ly
        sx[imax] = lx

        ly += Ly[imax]
        rowmax = max(rowmax, Lx[imax])
        if iy == gridy - 1 or (~picked).sum() == 0:
            lx += rowmax
        LY = max(LY, ly)
        iy += 1
        if iy >= gridy:
            iy = 0
            ix += 1
    LX = lx
    return LY, LX, sy, sx


def svdecon(X, k=100):
    np.random.seed(0)  # Fix seed to get same output for eigsh
    """
    v0 = np.random.uniform(-1,1,size=min(X.shape)) 
    NN, NT = X.shape
    if NN>NT:
        COV = (X.T @ X)/NT
    else:
        COV = (X @ X.T)/NN
    if k==0:
        k = np.minimum(COV.shape) - 1
    Sv, U = eigsh(COV, k = k, v0=v0)
    U, Sv = np.real(U), np.abs(Sv)
    U, Sv = U[:, ::-1], Sv[::-1]**.5
    if NN>NT:
        V = U
        U = X @ V
        U = U/(U**2).sum(axis=0)**.5
    else:
        V = (U.T @ X).T
        V = V/(V**2).sum(axis=0)**.5
    """
    U, Sv, V = PCA(
        n_components=k, svd_solver="randomized", random_state=np.random.RandomState(0)
    )._fit(X)
    return U, Sv, V
