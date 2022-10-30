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
    )._fit(X)
    return U, Sv, V
