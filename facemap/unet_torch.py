# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Network ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

def convbatchrelu(in_channels, out_channels, sz):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
      nn.BatchNorm2d(out_channels, eps=1e-5),
      nn.ReLU(inplace=True),
      )

class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t == 0:
                self.conv.add_module('conv_%d'%t,
                                 convbatchrelu(in_channels,
                                               out_channels,
                                               kernel_size))
            else:
                self.conv.add_module('conv_%d'%t,
                                 convbatchrelu(out_channels,
                                               out_channels,
                                               kernel_size))

    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x

class downsample(nn.Module):
    def __init__(self, nbase, kernel_size):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase) - 1):
            self.down.add_module('conv_down_%d'%n,
                               convdown(nbase[n],
                                        nbase[n + 1],
                                        kernel_size))

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class convup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('conv_0', convbatchrelu(in_channels,
                                                     out_channels,
                                                     kernel_size))
        self.conv.add_module('conv_1', convbatchrelu(out_channels,
                                                     out_channels,
                                                     kernel_size))

    def forward(self, x, y):
        x = self.conv[0](x)
        x = self.conv[1](x + y)
        return x

class upsample(nn.Module):
    def __init__(self, nbase, kernel_size):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        for n in range(len(nbase) - 1 , 0, -1):
            self.up.add_module('conv_up_%d'%(n - 1),
              convup(nbase[n], nbase[n - 1], kernel_size))

    def forward(self, xd):
        x = xd[-1]
        for n in range(0, len(self.up)):
            if n > 0:
                x = self.upsampling(x)
            x = self.up[n](x, xd[len(xd) - 1 - n])
        return x

class FMnet(nn.Module):
    def __init__(self, nbase, nout, kernel_size, labels_id):
        super(UNet, self).__init__()
        self.nbase = nbase
        self.nout = nout
        self.kernel_size = kernel_size
        self.heatmap_labels = labels_id
        
        self.downsample = downsample(nbase, kernel_size)
        nbaseup = nbase[1:]
        nbaseup.append(nbase[-1])
        self.upsample = upsample(nbaseup, kernel_size)
        #self.locref_output = nn.Conv2d(nbase[1], self.nout*2, kernel_size,
        #                        padding=kernel_size//2)
        self.hm_output = nn.Conv2d(nbase[1], self.nout, kernel_size,
                        padding=kernel_size//2)

    def forward(self, data):
        T0 = self.downsample(data)
        T0 = self.upsample(T0)
        #locref = self.locref_output(T0)
        heatmap = self.hm_output(T0)
        return heatmap

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            self.load_state_dict(torch.load(filename))
        else:
            self.__init__(self.nbase,
                        self.nout,
                        self.kernel_size,
                        self.concatenation)

            self.load_state_dict(torch.load(filename,
                                          map_location=torch.device('cpu')))