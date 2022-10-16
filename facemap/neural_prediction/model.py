import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import keypoints


class KeypointsNetwork(nn.Module):
    """
    Keypoints to neural PCs / neural activity model
    """

    def __init__(
        self, n_in=28, n_kp=None, n_filt=10, kernel_size=201,
        n_core_layers=2, n_latents=256, n_out_layers=1,
        n_out=128, n_med=50, n_animals=1,
        identity=False, relu_wavelets=True, relu_latents=True,
    ):
        super().__init__()
        self.core = Core(
            n_in=n_in, n_kp=n_kp, n_filt=n_filt, kernel_size=kernel_size,
            n_layers=n_core_layers, n_med=n_med, n_latents=n_latents,
            identity=identity, 
            relu_wavelets=relu_wavelets, relu_latents=relu_latents,
        )
        self.readout = Readout(
            n_animals=n_animals, n_latents=n_latents, n_layers=n_out_layers, n_out=n_out
        )

    def forward(self, x, sample_inds=None, animal_id=0):
        latents = self.core(x)
        if sample_inds is not None:
            latents = latents[sample_inds]
        latents = latents.reshape(x.shape[0], -1, latents.shape[-1])
        y_pred = self.readout(latents, animal_id=animal_id)
        return y_pred, latents


class Core(nn.Module):
    """
    Core network of the KeypointsNetwork with the following structure:
        linear -> conv1d -> relu -> linear -> relu = latents for KeypointsNetwork model
    """

    def __init__(
        self, n_in=28, n_kp=None, n_filt=10, kernel_size=201,
        n_layers=1, n_med=50, n_latents=256, identity=False,
        relu_wavelets=True, relu_latents=True,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_kp = n_in if n_kp is None or identity else n_kp
        self.n_filt = (n_filt // 2) * 2  # must be even for initialization
        self.relu_latents = relu_latents
        self.relu_wavelets = relu_wavelets
        self.n_layers = n_layers
        self.n_latents = n_latents
        self.features = nn.Sequential()

        # combine keypoints into n_kp features
        if identity:
            self.features.add_module("linear0", nn.Identity(self.n_in))
        else:
            self.features.add_module(
                "linear0",
                nn.Sequential(
                    nn.Linear(self.n_in, self.n_kp),
                ),
            )
        # initialize filters with gabors
        f = np.geomspace(1, 10, self.n_filt // 2).astype("float32")
        gw0 = keypoints.gabor_wavelet(1, f[:, np.newaxis], 0, n_pts=kernel_size)
        gw1 = keypoints.gabor_wavelet(1, f[:, np.newaxis], np.pi / 2, n_pts=kernel_size)
        wav_init = np.vstack((gw0, gw1))
        # compute n_filt wavelet features of each one => n_filt * n_kp features
        self.features.add_module(
            "wavelet0",
            nn.Conv1d(1, self.n_filt, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False,
            ),
        )
        self.features[-1].weight.data = torch.from_numpy(wav_init).unsqueeze(1)
    
        for n in range(1, n_layers):
            n_in = self.n_kp * self.n_filt if n == 1 else n_med
            self.features.add_module(
                f"linear{n}",
                nn.Sequential(
                    nn.Linear(n_in, n_med),
                ),
            )

        # latent linear layer
        n_med = n_med if n_layers > 1 else self.n_filt * self.n_kp
        self.features.add_module(
            "latent",
            nn.Sequential(
                nn.Linear(n_med, n_latents),
            ),
        )

    def wavelets(self, x):
        """compute wavelets of keypoints through linear + conv1d + relu layer"""
        # x is (n_batches, time, features)
        out = self.features[0](x.reshape(-1, x.shape[-1]))
        out = out.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        # out is now (n_batches, n_kp, time)
        out = out.reshape(-1, out.shape[-1]).unsqueeze(1)
        # out is now (n_batches * n_kp, 1, time)
        out = self.features[1](out)
        # out is now (n_batches * n_kp, n_filt, time)
        out = out.reshape(-1, self.n_kp * self.n_filt, out.shape[-1]).transpose(
            2, 1
        )
        out = out.reshape(-1, self.n_kp * self.n_filt)
        if self.relu_wavelets:
            out = F.relu(out)

        # if n_layers > 1, go through more linear layers
        for n in range(1, self.n_layers):
            out = self.features[n + 1](out)
            out = F.relu(out)
        return out

    def forward(self, x=None, wavelets=None):
        """x is (n_batches, time, features)
        sample_inds is (sub_time) over batches
        """
        if wavelets is None:
            wavelets = self.wavelets(x)
        wavelets = wavelets.reshape(-1, wavelets.shape[-1])

        # latent layer
        latents = self.features[-1](wavelets)
        latents = latents.reshape(x.shape[0], -1, latents.shape[-1])
        if self.relu_latents:
            latents = F.relu(latents)
        latents = latents.reshape(-1, latents.shape[-1])
        return latents


class Readout(nn.Module):
    """
    Linear layer from latents to neural PCs or neurons
    """

    def __init__(self, n_animals=1, n_latents=256, n_layers=1, n_med=128, n_out=128):
        super().__init__()
        self.n_animals = n_animals
        self.features = nn.Sequential()
        self.bias = nn.Parameter(torch.zeros(n_out))
        if n_animals == 1:
            for j in range(n_layers):
                n_in = n_latents if j == 0 else n_med
                n_outc = n_out if j == n_layers - 1 else n_med
                self.features.add_module(f"linear{j}", nn.Linear(n_in, n_outc))
                if n_layers > 1 and j < n_layers - 1:
                    self.features.add_module(f"relu{j}", nn.ReLU())
        else:
            # no option for n_layers > 1
            for n in range(n_animals):
                self.features.add_module(f"linear0_{n}", nn.Linear(n_latents, n_out))
        self.bias.requires_grad = False

    def forward(self, latents, animal_id=0):
        if self.n_animals == 1:
            return self.features(latents) + self.bias
        else:
            return self.features[animal_id](latents) + self.bias
