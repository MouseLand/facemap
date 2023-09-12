import torch
import torch.nn as nn
import torch.nn.functional as F

# This file contains all models used in the project 
# The models are defined as classes, and the forward pass is defined in the forward() function
# The models are called in main.py

class FMnet(nn.Module):
    def __init__(
        self,
        img_ch=1,
        output_ch=1,
        channels=[16, 32, 64, 128, 200],
        device=['cuda' if torch.cuda.is_available() else 'cpu'][0],
        kernel=3,
        n_upsample=4,
    ):
        super().__init__()
        self.n_upsample = n_upsample
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

    def forward(self, x, verbose=False):
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

        mask = self.Conv2_1x1[0](x)
        mask_edges = self.Conv2_1x1[1](x)
        mask_dist_to_boundary = self.Conv2_1x1[2](x)

        return mask, mask_edges, mask_dist_to_boundary


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

