# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Import packages ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

print("numpy version: %s" % np.__version__)
import random
from platform import python_version

import cv2  # opencv
import pyqtgraph as pg
import torch  # pytorch
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QPushButton
from scipy.ndimage import gaussian_filter

print("python version:", python_version())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Global variables~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
N_FACTOR = 2**4 // (2**2)
SIGMA = 3 * 4 / N_FACTOR
Lx = 64
print("Global varaibles set:")
print("N_FACTOR:", N_FACTOR)
print("SIGMA:", SIGMA)

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
        x01 = np.percentile(X, 1)
        x99 = np.percentile(X, 99)
        X = (X - x01) / (x99 - x01)
    return X


def predict(net, im_input, batchsize=1, smooth=True):

    xmesh, ymesh = np.meshgrid(
        torch.arange(net.image_shape[0] / N_FACTOR),
        torch.arange(net.image_shape[1] / N_FACTOR),
    )
    ymesh = torch.from_numpy(ymesh).to(net.device)
    xmesh = torch.from_numpy(xmesh).to(net.device)

    # Predict
    with torch.no_grad():
        if im_input.ndim == 3:
            im_input = im_input[np.newaxis, ...]
        hm_pred, locx_pred, locy_pred = net(im_input)

        hm_pred = hm_pred.squeeze()
        locx_pred = locx_pred.squeeze()
        locy_pred = locy_pred.squeeze()

        if smooth:
            hm_smo = gaussian_filter(hm_pred.cpu().numpy(), [0, 1, 1])
            hm_smo = hm_smo.reshape(hm_smo.shape[0], hm_smo.shape[1], Lx * Lx)
            imax = torch.argmax(hm_smo, -1)
            likelihood = torch.diag(hm_smo[:, :, imax])
        else:
            hm_pred = hm_pred.reshape(hm_pred.shape[0], Lx * Lx)
            imax = torch.argmax(hm_pred, 1)
            likelihood = torch.diag(hm_pred[:, imax])

        # this part computes the position error on the training set
        locx_pred = locx_pred.reshape(locx_pred.shape[0], Lx * Lx)
        locy_pred = locy_pred.reshape(locy_pred.shape[0], Lx * Lx)

        nn = hm_pred.shape[0]
        x_pred = ymesh.flatten()[imax] - (2 * SIGMA) * locx_pred[torch.arange(nn), imax]
        y_pred = xmesh.flatten()[imax] - (2 * SIGMA) * locy_pred[torch.arange(nn), imax]

    return y_pred * N_FACTOR, x_pred * N_FACTOR, likelihood


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


# Following used to check cropped sections of frames
class test_popup(QDialog):
    def __init__(self, frame, gui):
        super().__init__(gui)
        self.gui = gui
        self.frame = frame

        self.setWindowTitle("Chosen ROI")
        self.verticalLayout = QtWidgets.QVBoxLayout(self)

        # Add image and ROI bbox
        self.win = pg.GraphicsLayoutWidget()
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
