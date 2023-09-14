"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
## Import packages
import os
from io import StringIO

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from ..gui import help_windows
from . import pose_helper_functions as pose_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Fine-tuning the model using the pre-trained weights and refined training data provided by the user.
"""

def train(
    train_dataloader,
    net,
    n_epochs,
    learning_rate,
    weight_decay,
    ggmax=50,
    test_dataloader=None,
    save_checkpoint=False,
    checkpoint_path=None,
    checkpoint_filename=None,
    gui=None,
    gui_obj=None,
):
    """
    Fine-tuning the model using the pre-trained weights and refined training data provided by the user.
    Parameters
    ----------
    train_dataloader : torch.utils.data.DataLoader
        The dataloader object containing the training data.
    net : torch.nn.Module
        The model to be trained.
    n_epochs : int
        The number of epochs to be trained.
    learning_rate : float
        The learning rate for the optimizer.
    weight_decay : float
        The weight decay for the optimizer.
    ggmax : float
        The maximum gradient norm for the optimizer.
    test_dataloader : torch.utils.data.DataLoader, optional
        The dataloader object containing the testing data.
    save_checkpoint : bool, optional
        Whether to save the best model. The default is False.
    checkpoint_path : str, optional
        The path to save the best model. The default is None.
    gui : qtpy.QMainWindow
        The main window of the application.
    gui_obj : QtWidgets
        The gui object of the application.
    """

    n_factor = 2**4 // (2**net.n_upsample)
    xmesh, ymesh = np.meshgrid(
        np.arange(train_dataloader.dataset.img_size[1] / n_factor),
        np.arange(train_dataloader.dataset.img_size[0] / n_factor),
    )
    ymesh = torch.from_numpy(ymesh).to(device)
    xmesh = torch.from_numpy(xmesh).to(device)

    sigma = 3 * 4 / n_factor
    Lx = 64

    LR = learning_rate * np.ones(
        n_epochs,
    )
    LR[-6:-3] = learning_rate / 10
    LR[-3:] = learning_rate / 25

    # Initialize the optimizer
    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    min_test_loss = np.inf
    avg_test_rmse = np.inf

    if gui is not None and gui_obj is not None:
        progress_bar = help_windows.ProgressBarPopup(gui, "Training model...")
    progress_output = StringIO()

    for epoch in tqdm(range(n_epochs), file=progress_output):
        for param_group in optimizer.param_groups:
            param_group["lr"] = LR[epoch]
        pose_utils.set_seed(epoch)
        train_loss = 0
        train_mean = 0
        n_batches = 0
        gnorm_max = 0

        for train_batch in train_dataloader:
            net.train()
            images = train_batch["image"].to(device, dtype=torch.float32)
            lbl = train_batch["keypoints"].to(device, dtype=torch.float32)

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

        if save_checkpoint:
            pred_keypoints, keypoints = get_test_predictions(net, test_dataloader)
            # Rescale error to be in image of shape (256, 256)
            bbox = test_dataloader.dataset.bbox
            rescale_factor = 256 / torch.max(bbox)
            keypoints[:, :, 0], keypoints[:, :, 1] = (
                keypoints[:, :, 0] * rescale_factor,
                keypoints[:, :, 1] * rescale_factor,
            )
            pred_keypoints[:, :, 0], pred_keypoints[:, :, 1] = (
                pred_keypoints[:, :, 0] * rescale_factor,
                pred_keypoints[:, :, 1] * rescale_factor,
            )
            test_rmse = pose_utils.get_rmse(
                pred_keypoints.cpu().numpy(), keypoints.cpu().numpy()
            )
            avg_test_rmse = np.nanmean(test_rmse)
            if avg_test_rmse < min_test_loss:
                min_test_loss = avg_test_rmse
                if checkpoint_filename is not None:
                    savepath = os.path.join(checkpoint_path, checkpoint_filename)
                else:
                    savepath = os.path.join(checkpoint_path, "checkpoint.pth")
                torch.save(net.state_dict(), savepath)
                print(
                    "~~~Saved checkpoint in %s: at epoch %d w/ min test loss: %f ~~~"
                    % (checkpoint_path, epoch, min_test_loss)
                )

        if epoch % 5 == 0:
            print(
                "Epoch %d: train loss %f, train mean %f, gnorm_max %f avg. test rmse %f"
                % (epoch, train_loss, train_mean, gnorm_max, avg_test_rmse)
            )

        if gui is not None and gui_obj is not None:
            progress_bar.update_progress_bar(progress_output, gui_obj)

    if gui is not None:
        progress_bar.close()

    return net


def get_test_predictions(net, test_dataloader):
    net.eval()

    pred_keypoints = torch.zeros(
        (test_dataloader.dataset.num_images, test_dataloader.dataset.num_keypoints, 2),
        dtype=torch.float32,
    )
    keypoints_original = torch.zeros(
        (test_dataloader.dataset.num_images, test_dataloader.dataset.num_keypoints, 2),
        dtype=torch.float32,
    )
    start_idx = 0

    for test_batch in test_dataloader:
        images = test_batch["image"].to(
            net.device, dtype=torch.float32
        )  # .cpu().numpy()
        bbox = test_batch["bbox"]
        keypoints = test_batch["keypoints"].to(net.device, dtype=torch.float32)

        # Keypoints prediction
        xlabels_pred, ylabels_pred, _ = pose_utils.predict(net, images, smooth=False)

        pred_keypoints[
            start_idx : start_idx + test_dataloader.batch_size, :, 0
        ] = xlabels_pred
        pred_keypoints[
            start_idx : start_idx + test_dataloader.batch_size, :, 1
        ] = ylabels_pred
        keypoints_original[
            start_idx : start_idx + test_dataloader.batch_size, :, 0
        ] = keypoints[:, :, 0]
        keypoints_original[
            start_idx : start_idx + test_dataloader.batch_size, :, 1
        ] = keypoints[:, :, 1]

    return pred_keypoints, keypoints
