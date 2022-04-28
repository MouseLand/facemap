## Import packages
import numpy as np
import torch
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from io import StringIO

from tqdm import tqdm

from . import pose_gui
from . import pose_helper_functions as pose_utils

"""
Fine-tuning the model using the pre-trained weights and refined training data
provided by the user.
"""


def train(
    train_dataset,
    train_loader,
    net,
    n_epochs,
    learning_rate,
    weight_decay,
    gui=None,
    gui_obj=None,
):
    """
    Fine-tuning the model using the pre-trained weights and refined training data provided by the user.
    Parameters
    ----------
    train_dataset : torch.utils.data.Dataset
        Dataset containing the training data i.e. the images and the corresponding keypoints.
    train_loader : torch.utils.data.DataLoader
        The dataloader object containing the training data.
    net : torch.nn.Module
        The model to be trained.
    n_epochs : int
        The number of epochs to be trained.
    learning_rate : float
        The learning rate for the optimizer.
    weight_decay : float
        The weight decay for the optimizer.
    gui : PyQt5.QMainWindow
        The main window of the application.
    gui_obj : QtWidgets
        The gui object of the application.
    """

    n_factor = 2**4 // (2**net.n_upsample)
    xmesh, ymesh = np.meshgrid(
        np.arange(train_dataset.img_size[0] / n_factor),
        np.arange(train_dataset.img_size[1] / n_factor),
    )
    ymesh = torch.from_numpy(ymesh).to(device)
    xmesh = torch.from_numpy(xmesh).to(device)

    sigma = 3 * 4 / n_factor
    Lx = 64

    ggmax = 50
    LR = learning_rate * np.ones(
        n_epochs,
    )
    LR[-6:-3] = learning_rate / 10
    LR[-3:] = learning_rate / 25

    # Initialize the optimizer
    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if gui is not None and gui_obj is not None:
        progress_bar = pose_gui.ProgressBarPopup(gui)
    progress_output = StringIO()

    for epoch in tqdm(range(n_epochs), file=progress_output):
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[epoch]
        pose_utils.set_seed(epoch)
        train_loss = 0
        train_mean = 0
        n_batches = 0
        gnorm_max = 0

        net.train()
        for batch in train_loader:
            images = batch["image"].to(device, dtype=torch.float32)
            lbl = batch["keypoints"].to(device, dtype=torch.float32)

            hm_pred, locx_pred, locy_pred = net(images)
            ######################################################################

            # do a lot of preparations for the true heatmaps and the location graphs
            lbl_mask = torch.isnan(lbl).sum(axis=-1)
            lbl[lbl_mask > 0] = 0
            lbl_nan = lbl_mask == 0
            lbl_nan = lbl_nan.to(device=device)
            lbl_batch = lbl

            # divide by the downsampling factor (typically 4)
            y_true = (lbl_batch[:, :, 0]) / n_factor
            x_true = (lbl_batch[:, :, 1]) / n_factor

            # relative locationsof keypoints
            locx = ymesh - x_true.unsqueeze(-1).unsqueeze(-1)
            locy = xmesh - y_true.unsqueeze(-1).unsqueeze(-1)

            # normalize the true heatmaps
            hm_true = torch.exp(-(locx**2 + locy**2) / (2 * sigma**2))
            hm_true = (
                10
                * hm_true
                / (1e-3 + hm_true.sum(axis=(-2, -1)).unsqueeze(-1).unsqueeze(-1))
            )

            # mask over which to train the location graphs
            mask = (locx**2 + locy**2) ** 0.5 <= sigma

            # normalize the location graphs for prediction
            locx = locx / (2 * sigma)
            locy = locy / (2 * sigma)

            # mask out nan's
            hm_true = hm_true[lbl_nan]
            y_true = y_true[lbl_nan]
            x_true = x_true[lbl_nan]
            locx = locx[lbl_nan]
            locy = locy[lbl_nan]
            mask = mask[lbl_nan]

            # subsample the non-nan heatmaps and location graphs
            hm_pred = hm_pred[lbl_nan]
            locx_pred = locx_pred[lbl_nan]
            locy_pred = locy_pred[lbl_nan]

            # heatmap loss
            loss = ((hm_true - hm_pred).abs()).sum(axis=(-2, -1))

            # loss from the location graphs, masked with mask
            # I use a weighting of 0.5. Much smaller or much bigger worked almost as well (0.05 and 5)
            loss += 0.5 * (
                mask * ((locx - locx_pred) ** 2 + (locy - locy_pred) ** 2) ** 0.5
            ).sum(axis=(-2, -1))

            with torch.no_grad():
                # this part computes the position error on the training set
                hm_pred = hm_pred.reshape(hm_pred.shape[0], Lx * Lx)
                locx_pred = locx_pred.reshape(locx_pred.shape[0], Lx * Lx)
                locy_pred = locy_pred.reshape(locy_pred.shape[0], Lx * Lx)

                nn = hm_pred.shape[0]
                imax = torch.argmax(hm_pred, 1)

                x_pred = (
                    ymesh.flatten()[imax] - (2 * sigma) * locx_pred[np.arange(nn), imax]
                )
                y_pred = (
                    xmesh.flatten()[imax] - (2 * sigma) * locy_pred[np.arange(nn), imax]
                )

                y_err = (y_true - y_pred).abs()
                x_err = (x_true - x_pred).abs()

                train_mean += ((y_err + x_err) / 2).mean().item()

            loss = loss.mean()
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            # this operation clips the gradient and returns its original norm
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), ggmax)
            # keep track of the largest gradient norm on this epoch
            gnorm_max = np.maximum(gnorm_max, gnorm.cpu())

            optimizer.step()

            n_batches += 1

        train_loss /= n_batches
        train_mean /= n_batches

        if epoch % 10 == 0:
            print("Epoch %d: loss %f, mean %f" % (epoch, train_loss, train_mean))

        if gui is not None and gui_obj is not None:
            progress_bar.update_progress_bar(progress_output, gui_obj)

    if gui is not None:
        progress_bar.close()

    return net
