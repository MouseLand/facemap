"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import random
from platform import python_version

import cv2  # opencv
import matplotlib
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Import packages ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import torch  # pytorch
from scipy.ndimage import gaussian_filter

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Global variables~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
N_FACTOR = 2**4 // (2**2)
SIGMA = 3 * 4 / N_FACTOR
Lx = 64


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helper functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


#  Following Function adopted from cellpose:
#  https://github.com/MouseLand/cellpose/blob/35c16c94e285a4ec2fa17f148f06bbd414deb5b8/cellpose/transforms.py#L187
def normalize99(X, device=None):
    """
    Normalize image so 0.0 is 1st percentile and 1.0 is 99th percentile
     Parameters
    -------------
    img: ND-array
        image of size [Ly x Lx]
    Returns
    --------------
    X: ND-array
        normalized image of size [Ly x Lx]
    """
    if device is not None:
        x01 = torch.quantile(X, 0.01)
        x99 = torch.quantile(X, 0.99)
        X = (X - x01) / (x99 - x01)
    else:
        x01 = np.nanpercentile(X, 1)
        x99 = np.nanpercentile(X, 99)
        X = (X - x01) / (x99 - x01)
    return X


def get_rmse(predictions, gt):
    """
    Compute Euclidean distance between predictions and ground truth
    Parameters
    ----------
    predictions : ND-array of shape (n_samples, n_joints, 2)
        Predictions from network
    gt : ND-array of shape (n_samples, n_joints, 2)
        Ground truth
    """
    x1, y1 = predictions[:, :, 0], predictions[:, :, 1]
    x2, y2 = gt[:, :, 0], gt[:, :, 1]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def predict(net, im_input, smooth=False):
    lx = int(net.image_shape[0] / N_FACTOR)
    ly = int(net.image_shape[1] / N_FACTOR)
    batch_size = im_input.shape[0]
    num_keypoints = len(net.bodyparts)
    locx_mesh, locy_mesh = torch.meshgrid(
        torch.arange(batch_size), torch.arange(num_keypoints), indexing="ij"
    )
    locx_mesh = locx_mesh.to(net.device)
    locy_mesh = locy_mesh.to(net.device)

    # Predict
    with torch.no_grad():
        hm_pred, locx_pred, locy_pred = net(im_input)

        if smooth:
            hm_pred = gaussian_filter(hm_pred.cpu().numpy(), [0, 1, 1])

        hm_pred = hm_pred.reshape(batch_size, num_keypoints, lx * ly)
        locx_pred = locx_pred.reshape(batch_size, num_keypoints, lx * ly)
        locy_pred = locy_pred.reshape(batch_size, num_keypoints, lx * ly)

        # likelihood, imax = torch.max(hm_pred, -1)
        _, imax = torch.max(hm_pred, -1)
        likelihood = torch.sigmoid(hm_pred)
        likelihood, _ = torch.max(likelihood, -1)
        i_y = imax % lx
        i_x = torch.div(imax, lx, rounding_mode="trunc")
        x_pred = (locx_pred[locx_mesh, locy_mesh, imax] * (-2 * SIGMA)) + i_x
        y_pred = (locy_pred[locx_mesh, locy_mesh, imax] * (-2 * SIGMA)) + i_y

    x_pred *= N_FACTOR
    y_pred *= N_FACTOR

    return y_pred, x_pred, likelihood


def numpy_predict(net, im_input, smooth=False):
    lx = int(net.image_shape[0] / N_FACTOR)
    ly = int(net.image_shape[1] / N_FACTOR)
    batch_size = im_input.shape[0]
    num_keypoints = len(net.bodyparts)
    locx_mesh, locy_mesh = np.meshgrid(
        np.arange(batch_size), np.arange(num_keypoints), indexing="ij"
    )

    # Predict
    with torch.no_grad():
        hm_pred, locx_pred, locy_pred = net(im_input)

        if smooth:
            hm_pred = gaussian_filter(hm_pred.cpu().numpy(), [0, 1, 1])

        hm_pred = hm_pred.reshape(batch_size, num_keypoints, lx * ly).cpu().numpy()
        locx_pred = locx_pred.reshape(batch_size, num_keypoints, lx * ly).cpu().numpy()
        locy_pred = locy_pred.reshape(batch_size, num_keypoints, lx * ly).cpu().numpy()

        likelihood = np.maximum(hm_pred, -1)
        imax = np.argmax(hm_pred, -1)
        i_y = imax % lx
        i_x = imax // lx
        x_pred = (locx_pred[locx_mesh, locy_mesh, imax] * (-2 * SIGMA)) + i_x
        y_pred = (locy_pred[locx_mesh, locy_mesh, imax] * (-2 * SIGMA)) + i_y

    x_pred *= N_FACTOR
    y_pred *= N_FACTOR

    return y_pred, x_pred, likelihood


def randomly_adjust_contrast(img):
    """
    Randomly adjusts contrast of image
    img: ND-array of size nchan x Ly x Lx
    Assumes image values in range 0 to 1
    """
    brange = [-0.2, 0.2]
    bdiff = brange[1] - brange[0]
    crange = [0.7, 1.3]
    cdiff = crange[1] - crange[0]
    imax = img.max()
    if (bdiff < 0.01) and (cdiff < 0.01):
        return img
    bfactor = np.random.rand() * bdiff + brange[0]
    cfactor = np.random.rand() * cdiff + crange[0]
    mm = img.mean()
    jj = img + bfactor * imax
    jj = np.minimum(imax, (jj - mm) * cfactor + mm)
    jj = jj.clip(0, imax)
    return jj


def add_motion_blur(img, kernel_size=None, vertical=True, horizontal=True):
    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))

    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)

    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    if vertical:
        # Apply the vertical kernel.
        img = cv2.filter2D(img, -1, kernel_v)
    if horizontal:
        # Apply the horizontal kernel.
        img = cv2.filter2D(img, -1, kernel_h)

    return img


def plot_imgs_landmarks(
    imgs, keypoints, pred_keypoints=None, cmap="jet", s=10, figsize=(10, 10)
):
    """
    Plot images and keypoints in a grid.
    Parameters
    ----------
    imgs : LIST of ND-arrays, float
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]
    landmarks : ND-array of shape (N, bodyparts, 2)
            Array of landmarks.
    Returns
    -------
    fig : matplotlib figure
        Figure containing the images and landmarks.
    """
    n_imgs = len(imgs)
    n_cols = int(np.ceil(np.sqrt(n_imgs)))
    n_rows = int(np.ceil(n_imgs / n_cols))

    cmap = matplotlib.cm.get_cmap(cmap)
    colornorm = matplotlib.colors.Normalize(vmin=0, vmax=keypoints[0].shape[0])
    colors = cmap(colornorm(np.arange(keypoints[0].shape[0])))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_imgs == 1:
        axes = np.array([axes])
    for i, ax in enumerate(axes.flat):
        if i < n_imgs:
            if imgs[i].ndim == 2:
                ax.imshow(imgs[i], cmap="gray")
            else:
                ax.imshow(imgs[i].squeeze(), cmap="gray")
            ax.scatter(keypoints[i][:, 0], keypoints[i][:, 1], s=s, color=colors)
            if pred_keypoints is not None:
                ax.scatter(
                    pred_keypoints[i][:, 0],
                    pred_keypoints[i][:, 1],
                    marker="+",
                    s=s * 2,
                    color=colors,
                )
        if i == n_imgs:
            ax.axis("off")
    return fig


# Following used to check cropped sections of frames
try:
    from qtpy import QtWidgets, QtCore
    from qtpy.QtWidgets import QDialog, QPushButton
    import pyqtgraph as pg

    class test_popup(QDialog):
        
        def __init__(self, frame, gui, title="Test Popup"):
            super().__init__(gui)
            self.gui = gui
            self.frame = frame

            self.setWindowTitle(title)
            self.verticalLayout = QtWidgets.QVBoxLayout(self)

            # Add image and ROI bbox
            self.win = pg.GraphicsLayoutWidget()
            self.win.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
            ROI_win = self.win.addViewBox(invertY=True)
            self.img = pg.ImageItem(self.frame)
            ROI_win.addItem(self.img)
            self.win.show()
            self.verticalLayout.addWidget(self.win)

            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.clicked.connect(self.close)

            # Position buttons
            self.widget = QtWidgets.QWidget(self)
            self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
            self.horizontalLayout.setContentsMargins(-1, -1, -1, 0)
            self.horizontalLayout.setObjectName("horizontalLayout")
            self.horizontalLayout.addWidget(self.cancel_button)
            self.verticalLayout.addWidget(self.widget)

            self.show()
except ImportError:
    print("pip install facemap[gui]")