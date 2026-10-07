from __future__ import division
import os
from torch.nn.modules import module
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from .fusion import AsymBiChaFuse


from torchvision import transforms
from  torchvision.models.resnet import BasicBlock

class _FCNHead(nn.Module):

    def __init__(self, in_channels, channels, momentum, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,kernel_size=3, padding=1, bias=False),
        norm_layer(inter_channels, momentum=momentum),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(in_channels=inter_channels, out_channels=channels,kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ALC_No_Sigmoid(nn.Module):
    def __init__(self, in_channels=3, layers=[4,4,4], channels=[8,16,32,64], fuse_mode='AsymBi', act_dilation=16, classes=1, tinyFlag=False,
                 norm_layer=BatchNorm2d,groups=1,norm_kwargs=None, **kwargs):
        super(ALC_No_Sigmoid, self).__init__()

        self.layer_num = len(layers)
        self.tinyFlag = tinyFlag
        self.groups = groups
        self._norm_layer = norm_layer
        stem_width = int(channels[0])
        self.momentum=0.9
        if tinyFlag:
            self.stem = nn.Sequential(
                norm_layer(in_channels, self.momentum),
                nn.Conv2d(in_channels, out_channels=stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width * 2, momentum=self.momentum),
                nn.ReLU(inplace=True)
            )

        else:
            self.stem = nn.Sequential(


                norm_layer(in_channels, momentum=self.momentum),
                nn.Conv2d(in_channels=in_channels, out_channels=stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=stem_width, out_channels=stem_width, kernel_size=3, stride=1, padding=1,
                          bias=False),
                norm_layer(stem_width, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=stem_width, out_channels=stem_width * 2, kernel_size=3, stride=1, padding=1,
                          bias=False),
                norm_layer(stem_width * 2, momentum=self.momentum),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )


            self.head = _FCNHead(in_channels=channels[0], channels=classes, momentum=self.momentum)

            self.layer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                           out_channels=channels[1],
                                           in_channels=channels[1], stride=1)

            self.layer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                           out_channels=channels[2], stride=2,
                                           in_channels=channels[1])

            self.layer3 = self._make_layer(block=BasicBlock, blocks=layers[2],
                                           out_channels=channels[3], stride=2,
                                           in_channels=channels[2])
            self.deconv2 = nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2], kernel_size=(4, 4),

                                              stride=2, padding=1)
            self.uplayer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                             out_channels=channels[2], stride=1,
                                             in_channels=channels[2])


            self.deconv1 = nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[1], kernel_size=(4, 4),
                                              stride=2, padding=1)

            self.deconv0 = nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0], kernel_size=(4, 4),
                                              stride=2, padding=1)

            self.uplayer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                             out_channels=channels[1], stride=1,
                                             in_channels=channels[1])


            if self.layer_num == 4:
                self.layer4 = self._make_layer(block=BasicBlock, blocks=layers[3],
                                               out_channels=channels[3], stride=2,
                                               in_channels=channels[3])

            if self.layer_num == 4:
                self.fuse34 = self._fuse_layer(fuse_mode, channels=channels[3])

            self.fuse23 = self._fuse_layer(fuse_mode, channels=channels[2])
            self.fuse12 = self._fuse_layer(fuse_mode, channels=channels[1])


    def _make_layer(self, block, out_channels, in_channels, blocks, stride):

        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or out_channels != in_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels , stride),
                norm_layer(out_channels * block.expansion, momentum=self.momentum),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, self.groups, norm_layer=norm_layer))
        self.inplanes = out_channels  * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, self.groups, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _fuse_layer(self, fuse_mode, channels):

        if fuse_mode == 'AsymBi':
          fuse_layer = AsymBiChaFuse(channels=channels)
        else:
            raise ValueError('Unknown fuse_mode')
        return fuse_layer

    def forward(self, x):

        _, _, hei, wid = x.shape

        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)

        out = self.layer3(c2)

        if self.layer_num == 4:
            c4 = self.layer4(out)
            if self.tinyFlag:
                c4 = transforms.Resize([hei//4, wid//4])(c4)
            else:
                c4 =  transforms.Resize([hei//16, wid//16])(c4)
            out = self.fuse34(c4, out)

        if self.tinyFlag:
            out =  transforms.Resize([hei//2, wid//2])(out)
        else:
            out =  transforms.Resize([hei//16, wid//16])(out)

        out = self.deconv2(out)
        out = self.fuse23(out, c2)
        if self.tinyFlag:
            out =  transforms.Resize([hei, wid])(out)
        else:
            out =  transforms.Resize( [hei//8, wid//8])(out)

        out = self.deconv1(out)
        out = self.fuse12(out, c1)

        out = self.deconv0(out)
        pred = self.head(out)


        if self.tinyFlag:
            out = pred
        else:
            out = transforms.Resize( [hei, wid])(pred)


        return out

    def evaluate(self, x):
        return self.forward(x)
