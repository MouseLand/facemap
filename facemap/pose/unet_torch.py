# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Network ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class FMnet(nn.Module):
    def __init__(self, img_ch, output_ch, labels_id, filters=64, kernel=3, device=None):
        super(FMnet, self).__init__()
        self.img_ch = img_ch
        self.output_ch = output_ch
        self.labels_id = labels_id
        self.filters = filters
        self.kernel = kernel
        self.DEVICE = device

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.heatmap_labels = self.labels_id

        channels = [self.filters]
        for i in range(4):
            channels.append(channels[-1]*2)
            
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=channels[0], kernel_sz=self.kernel)
        self.Conv2 = conv_block(ch_in=channels[0],ch_out=channels[1], kernel_sz=self.kernel)
        self.Conv3 = conv_block(ch_in=channels[1],ch_out=channels[2], kernel_sz=self.kernel)
        self.Conv4 = conv_block(ch_in=channels[2],ch_out=channels[3], kernel_sz=self.kernel)
        self.Conv5 = conv_block(ch_in=channels[3],ch_out=channels[4], kernel_sz=self.kernel)

        self.Up5 = up_conv(ch_in=channels[4], ch_out=channels[3], kernel_sz=kernel)
        self.Up_conv5 = conv_block(ch_in=channels[4], ch_out=channels[3], kernel_sz=self.kernel)

        self.Up4 = up_conv(ch_in=channels[3], ch_out=channels[2], kernel_sz=kernel)
        self.Up_conv4 = conv_block(ch_in=channels[3], ch_out=channels[2], kernel_sz=self.kernel)
        
        self.Conv_1x1_hm = nn.Conv2d(channels[2],output_ch,kernel_size=self.kernel,
                                  padding=self.kernel//2)
        self.Conv_1x1_locref = nn.Conv2d(channels[2],output_ch*2,kernel_size=self.kernel,
                                  padding=self.kernel//2)

    def forward(self, x, verbose=False):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5), dim=1) 
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4), dim=1)
        d4 = self.Up_conv4(d4)

        hm = self.Conv_1x1_hm(d4)
        locref = self.Conv_1x1_locref(d4)
        
        if verbose:
            print("down:",x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
            print("up", d5.shape, d4.shape)
            print('outc:', hm.shape, locref.shape)

        return hm, locref
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, cpu=False):
        if not cpu:
            self.load_state_dict(torch.load(filename))
        else:
            self.__init__(self.img_ch, self.output_ch, 
                        self.labels_id, self.filters, 
                        self.kernel)

            self.load_state_dict(torch.load(filename,
                                          map_location=torch.device('cpu')))

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_sz):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_sz, padding=kernel_sz//2),
            nn.BatchNorm2d(ch_out, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_sz, padding=kernel_sz//2),
            nn.BatchNorm2d(ch_out, eps=1e-5),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_sz):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=kernel_sz//2),
		    nn.BatchNorm2d(ch_out, eps=1e-5),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
