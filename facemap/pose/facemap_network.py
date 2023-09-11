"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Network ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FMnet(nn.Module):
    def __init__(
        self,
        img_ch,
        output_ch,
        labels_id,
        channels,
        device,
        kernel=3,
        shape=(256, 256),
        n_upsample=2,
    ):
        super().__init__()
        self.n_upsample = n_upsample
        self.image_shape = shape
        self.bodyparts = labels_id
        self.device = device

        self.Conv = nn.Sequential()
        self.Conv.add_module(
            "conv0",
            convblock(ch_in=img_ch, ch_out=channels[0], kernel_sz=kernel, block=0),
        )
        for k in range(1, len(channels)):
            self.Conv.add_module(
                f"conv{k}",
                convblock(
                    ch_in=channels[k - 1], ch_out=channels[k], kernel_sz=kernel, block=k
                ),
            )

        self.Up_conv = nn.Sequential()
        for k in range(n_upsample):
            self.Up_conv.add_module(
                f"upconv{k}",
                convblock(
                    ch_in=channels[-1 - k] + channels[-2 - k],
                    ch_out=channels[-2 - k],
                    kernel_sz=kernel,
                ),
            )

        self.Conv2_1x1 = nn.Sequential()
        for j in range(3):
            self.Conv2_1x1.add_module(
                f"conv{j}",
                nn.Conv2d(channels[-2 - k], output_ch, kernel_size=1, padding=0),
            )

    def forward(self, x, normalize=False, smooth_confidence=False, verbose=False):
        # encoding path
        xout = []
        x = self.Conv[0](x)
        xout.append(x)
        for k in range(1, len(self.Conv)):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
            x = self.Conv[k](x)
            xout.append(x)

        for k in range(len(self.Up_conv)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.Up_conv[k](torch.cat((x, xout[-2 - k]), axis=1))

        locx = self.Conv2_1x1[1](x)
        locy = self.Conv2_1x1[2](x)
        hm = self.Conv2_1x1[0](x)
        hm = F.relu(hm)
        if self.training or normalize:
            hm = (
                10 * hm / (1e-4 + hm.sum(axis=(-2, -1)).unsqueeze(-1).unsqueeze(-1))
            )  # Normalize
        if smooth_confidence:
            hm = smooth_heatmap(hm, self.device)

        return hm, locx, locy


# Create a gaussian wavelet of a set bin size
def gaussian_wavelet(bin_size, sigma):
    x = np.arange(-bin_size // 2, bin_size // 2 + 1)
    gaussian = np.exp(-(x**2) / (2 * sigma**2))
    return gaussian / gaussian.sum()


# Write a function that takes a heatmap of dimensions (n_batches, n_keypoints, width, height)
# and returns a smoothed heatmap using the gaussian smoothing function
def smooth_heatmap(heatmap, device=None):
    """
    Parameters
    ----------------
    heatmap: ND-array of shape (n_batches, n_keypoints, width, height)
                A heatmap containing confidence scores for each key point
    Returns
    ----------------
    heatmap_smoothed_reshaped: ND-array of shape (n_batches, n_keypoints, width, height)
                            A smoothed heatmap containing confidence scores resulting from covolution
                             across a batch of images for each key point
    """
    n_batches = heatmap.shape[0]
    n_keypoints = heatmap.shape[1]
    lx = heatmap.shape[2]
    ly = heatmap.shape[3]
    bin_size = heatmap.shape[0] - 1

    gaussian_filter = torch.nn.Conv1d(
        in_channels=n_keypoints,
        out_channels=n_keypoints,
        kernel_size=bin_size + 1,
        bias=False,
        padding="same",
    )
    for i in range(n_keypoints):
        gaussian_filter.weight.data[i][i] = torch.tensor(
            gaussian_wavelet(bin_size, 2), dtype=torch.float32
        )
    gaussian_filter.weight.data = (gaussian_filter.weight.data).to(device)
    gaussian_filter.weight.requires_grad = False

    # Apply gaussian smoothing on each keypoint at a time using the gaussian smoothing function above
    heatmap_reshaped = heatmap.reshape(n_batches, n_keypoints, -1).permute(2, 1, 0)
    heatmap_smoothed = gaussian_filter(heatmap_reshaped)
    heatmap_smoothed_reshaped = heatmap_smoothed.permute(2, 1, 0).reshape(
        n_batches, n_keypoints, lx, ly
    )
    return heatmap_smoothed_reshaped


class convblock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_sz, block=-1):
        super().__init__()
        self.conv = nn.Sequential()
        self.block = block
        if self.block != 0:
            self.conv.add_module("conv_0", batchconv(ch_in, ch_out, kernel_sz))
        else:
            self.conv.add_module("conv_0", batchconv0(ch_in, ch_out, kernel_sz))
        self.conv.add_module("conv_1", batchconv(ch_out, ch_out, kernel_sz))

    def forward(self, x):
        x = self.conv[1](self.conv[0](x))
        return x


def batchconv0(ch_in, ch_out, kernel_sz):
    return nn.Sequential(
        nn.BatchNorm2d(ch_in, eps=1e-5, momentum=0.1),
        nn.Conv2d(ch_in, ch_out, kernel_sz, padding=kernel_sz // 2, bias=False),
    )


def batchconv(ch_in, ch_out, sz):
    return nn.Sequential(
        nn.BatchNorm2d(ch_in, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=True),
        nn.Conv2d(ch_in, ch_out, sz, padding=sz // 2, bias=False),
    )
