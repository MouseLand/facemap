"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import cv2
import h5py
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA


BODYPARTS = [
    "eye(back)",
    "eye(bottom)",
    "eye(front)",
    "eye(top)",
    "lowerlip",
    "mouth",
    "nose(bottom)",
    "nose(r)",
    "nose(tip)",
    "nose(top)",
    "nosebridge",
    "paw",
    "whisker(I)",  # "whisker(c1)",
    "whisker(III)",  # "whisker(d2)",
    "whisker(II)",  # "whisker(d1)",
]

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


def bin1d(X, bin_size, axis=0):
    """mean bin over axis of data with bin bin_size"""
    if bin_size > 0:
        size = list(X.shape)
        Xb = X.swapaxes(0, axis)
        Xb = (
            Xb[: size[axis] // bin_size * bin_size]
            .reshape((size[axis] // bin_size, bin_size, -1))
            .mean(axis=1)
        )
        Xb = Xb.swapaxes(axis, 0)
        size[axis] = Xb.shape[axis]
        Xb = Xb.reshape(size)
        return Xb
    else:
        return X


def compute_varexp(y_true, y_pred):
    """variance explained of y_true by y_pred across axis=0"""
    y_var = ((y_true - y_true.mean(axis=0)) ** 2).mean(axis=0)
    residual = ((y_true - y_pred) ** 2).mean(axis=0)
    varexp = 1 - residual / y_var
    return varexp


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
        np.diff(np.stack((inds_train, inds_train + l_train)).T.flatten()).min() - pad
    )
    itrain = inds_train[:, np.newaxis] + np.arange(0, l_train, 1, int)
    itest = inds_test[:, np.newaxis] + np.arange(0, l_test, 1, int)
    return itrain, itest


def split_batches(tcam, tneural, frac=0.25, pad=3, itrain=None, itest=None):
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
        itrain, itest = split_traintest(len(tneural), frac=frac, pad=pad)
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
    itrain=None,
    itest=None,
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
    splits = split_batches(tcam, tneural, frac=frac, itrain=itrain, itest=itest)
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


def load_keypoints(bodyparts, h5_path):
    """Load keypoints using h5py

    Args:
        bodyparts (list): List of bodyparts in the same order as in FacemapDataset
        h5_path (hdf filepath): Path to hdf file containing keypoints
    Returns:
        pose_data (np.array): Array of size 3 x key points x frames
    """
    pose_x_coord = []
    pose_y_coord = []
    pose_likelihood = []
    pose_data = h5py.File(h5_path, "r")["Facemap"]
    for bodypart in bodyparts:  # Load bodyparts in the same order as in FacemapDataset
        pose_x_coord.append(pose_data[bodypart]["x"][:])
        pose_y_coord.append(pose_data[bodypart]["y"][:])
        pose_likelihood.append(pose_data[bodypart]["likelihood"][:])

    pose_x_coord = np.array([pose_x_coord])  # size: key points x frames
    pose_y_coord = np.array([pose_y_coord])  # size: key points x frames
    pose_likelihood = np.array([pose_likelihood])  # size: key points x frames

    pose_data = np.concatenate(
        (pose_x_coord, pose_y_coord, pose_likelihood), axis=0
    )  # size: 3 x key points x frames

    return pose_data


def get_keypoints_for_neuralpred(
    kp_file,
    bodyparts=None,
    outlier_filter=True,
    keypoint_labels=[],
    confidence_threshold=False,
):
    """
    Load keypoints and normalize them
    Parameters
    ----------
    keypoints_path : str
        path to keypoints file
    Returns
    -------
    keypoints_normalized : 2D-array
        normalized keypoints of shape [n_frames, n_keypoints * 2]
    """
    if bodyparts is None:
        bodyparts = BODYPARTS

    pose_data = load_keypoints(bodyparts, kp_file)
    print("pose data shape", pose_data.shape)
    xy = np.stack((pose_data[0].T, pose_data[1].T), axis=-1)
    print("xy.shape", xy.shape)
    conf = pose_data[2]

    # TODO: Add function for confidence_threshold

    if outlier_filter:
        for i in range(xy.shape[1]):
            x, y = xy[:, i, 0], xy[:, i, 1]
            x, y = filter_outliers(x, y)
            xy[:, i] = np.vstack((x, y)).T

    print("outlier filter shape", xy.shape)
    # Normalize keypoints (input data x)
    x = xy.reshape(xy.shape[0], -1).copy()
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    print("x.shape", x.shape)
    return x


def nanmedian_filter(x, win=7):
    """nanmedian filter array along last axis"""
    nt = x.shape[-1]
    # pad so that x will be divisible by win
    pad = (win - (nt + 2 * (win // 2)) % win) % win
    xpad = np.pad(x, (win // 2, win // 2 + win + pad), mode="edge")
    xmed = np.zeros_like(x)
    for k in range(win):
        xm = np.nanmedian(xpad[k : k - win].reshape(-1, win), axis=-1)
        xmed[..., k::win] = xm[: len(np.arange(k, nt, win))]
    return xmed


def filter_outliers(
    x, y, filter_window=15, baseline_window=50, max_spike=25, max_diff=25
):
    # remove frames with large jumps
    x_diff = np.abs(
        np.append(
            np.zeros(
                1,
            ),
            np.diff(x),
        )
    )
    y_diff = np.abs(
        np.append(
            np.zeros(
                1,
            ),
            np.diff(y),
        )
    )
    replace_inds = np.logical_or(x_diff > max_diff, y_diff > max_diff)
    x[replace_inds] = np.nan
    y[replace_inds] = np.nan

    # remove frames with large deviations from baseline
    x_baseline = nanmedian_filter(x, baseline_window)
    y_baseline = nanmedian_filter(y, baseline_window)
    replace_inds = np.logical_or(
        np.abs(x - x_baseline) > max_spike, np.abs(y - y_baseline) > max_spike
    )
    x[replace_inds] = np.nan
    y[replace_inds] = np.nan
    replace_inds = np.isnan(x)

    # filter x and y
    x_filt = nanmedian_filter(x, filter_window)
    y_filt = nanmedian_filter(y, filter_window)
    # x_filt = x_baseline
    # y_filt = y_baseline

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
        x_baseline = x.mean()  # nanmedian_filter(x, baseline_window)
        y_baseline = y.mean()  # nanmedian_filter(y, baseline_window)
        max_spike = x.std() * 5, y.std() * 5
        replace_inds = np.logical_or(
            np.abs(x - x_baseline) > max_spike[0], np.abs(y - y_baseline) > max_spike[1]
        )
        x[replace_inds] = x.mean()  # _baseline[replace_inds]
        y[replace_inds] = y.mean()  # _baseline[replace_inds]

    return x, y


def gabor_wavelet(sigma, f, ph, n_pts=201, is_torch=False):
    x = np.linspace(0, 2 * np.pi, n_pts + 1)[:-1].astype("float32")
    cos = np.cos
    sin = np.sin
    exp = np.exp
    xc = x - x.mean()
    cosine = cos(ph + f * xc)
    gaussian = exp(-(xc**2) / (2 * sigma**2))
    G = gaussian * cosine
    G /= (G**2).sum() ** 0.5
    return G


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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[np.newaxis, ...]
            img.append(frame)
        else:
            print("Error reading frame")
    return img


def get_batch_frames(
    frame_indices, total_frames, cumframes, containers, video_idx, grayscale=False
):
    # frame_indices = np.maximum(0, np.minimum(total_frames - 1, frame_indices))
    """
    try:
        video_idx = (cumframes < frame_indices).nonzero()[0][-1]
    except:
        video_idx = 0
    """
    imgs = []
    # for vs in containers[video_idx]:
    capture = containers[0][video_idx]
    for idx in frame_indices:
        frame_ind = idx - cumframes[video_idx]
        # capture = vs
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_ind:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
        ret, frame = capture.read()
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[np.newaxis, ...]
        if ret:
            imgs.append(frame)
        else:
            print("Error reading frame")
    return np.array(imgs)


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
    # Set bounds of the resampled timestamps
    resampled_timestamps[resampled_timestamps < 0] = 0
    resampled_timestamps[resampled_timestamps > init_timestamps.size - 1] = (
        init_timestamps.size - 1
    )
    return resampled_timestamps.squeeze().astype(int)


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
    reflectors = np.zeros((yrange.size, xrange.size), bool)
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
    picked = np.zeros((Ly.size,), bool)
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
    )._fit(X)[:3]
    return U, Sv, V
