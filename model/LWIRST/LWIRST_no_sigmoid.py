import torch.nn as nn
import torch
from thop import profile
import time

class dw_conv(nn.Module):
    def __init__(self, in_dim, out_dim, relu=True):
        super(dw_conv, self).__init__()
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.dw_conv_k3 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(out_dim),
            activation())
    def forward(self, x):
        x = self.dw_conv_k3(x)
        return x


class InitialBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)


        self.ext_branch = nn.MaxPool2d(3, stride=2, padding=1)


        self.batch_norm = nn.BatchNorm2d(out_channels)


        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)


        out = torch.cat((main, ext), 1)


        out = self.batch_norm(out)

        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 depthwise=False,
                 dilated=False,
                 regular=False,
                 dropout_prob=0.0,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        if asymmetric:
            internal_channels = channels // internal_ratio

            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif depthwise:
            internal_channels = channels * 2

            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = dw_conv(internal_channels,internal_channels)


            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif dilated:
            internal_channels = channels // internal_ratio

            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())


            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        elif regular:
            internal_channels = channels // internal_ratio

            self.ext_conv1 = nn.Sequential(
                nn.Conv2d(
                    channels,
                    internal_channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())

            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), nn.BatchNorm2d(internal_channels), activation())


            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,
                    kernel_size=1,
                    stride=1,
                    bias=bias), nn.BatchNorm2d(channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)


        self.out_activation = activation()

    def forward(self, x):

        main = x


        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)


        out = main + ext

        return self.out_activation(out)


class DownsamplingBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0.0,
                 bias=False,
                 relu=True):
        super().__init__()


        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU


        self.main_max1 = nn.MaxPool2d(
            2,
            stride=2)


        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())


        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), nn.BatchNorm2d(internal_channels), activation())


        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)


        self.out_activation = activation()

    def forward(self, x):


        main = self.main_max1(x)


        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)


        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)


        if main.is_cuda:
            padding = padding.cuda()


        main = torch.cat((main, padding), 1)


        out = main + ext

        return self.out_activation(out)


class LWIRST_No_Sigmoid(nn.Module):
    def __init__(self, n_classes=1, encoder_relu=False, decoder_relu=True, channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(3,5,7,9), padding=(1,2,3,4)):
        super().__init__()


        self.initial_block = InitialBlock(3, channel[0], relu=encoder_relu)


        self.downsample1_0 = DownsamplingBottleneck(
            channel[0],
            channel[1],
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.01, relu=encoder_relu)


        self.downsample2_0 = DownsamplingBottleneck(
            channel[1],
            channel[2],
            dropout_prob=0.1,
            relu=encoder_relu)

        self.Depthwise2_1 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous2_2 = RegularBottleneck(
            channel[2], dilation=dilations[0], padding=dilations[0], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric2_3 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[0],
            padding=padding[0],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous2_4 = RegularBottleneck(
            channel[2], dilation=dilations[1], padding=dilations[1], dilated=True, dropout_prob=0.1, relu=encoder_relu)


        self.Depthwise2_5 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous2_6 = RegularBottleneck(
            channel[2], dilation=dilations[2], padding=dilations[2], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric2_7 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[1],
            padding=padding[1],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous2_8 = RegularBottleneck(
            channel[2], dilation=dilations[3], padding=dilations[3], dilated=True, dropout_prob=0.1, relu=encoder_relu)


        self.Depthwise3_1 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous3_2 = RegularBottleneck(
            channel[2], dilation=dilations[0], padding=dilations[0], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric3_3 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[2],
            padding=padding[2],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous3_4 = RegularBottleneck(
            channel[2], dilation=dilations[1], padding=dilations[1], dilated=True, dropout_prob=0.1, relu=encoder_relu)


        self.Depthwise3_5 = RegularBottleneck(
            channel[2], padding=1, depthwise=True, dropout_prob=0.1, relu=encoder_relu)
        self.Atrous3_6 = RegularBottleneck(
            channel[2], dilation=dilations[2], padding=dilations[2], dilated=True, dropout_prob=0.1, relu=encoder_relu)
        self.Asymmetric3_7 = RegularBottleneck(
            channel[2],
            kernel_size=kernel_size[3],
            padding=padding[3],
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.Atrous3_8 = RegularBottleneck(
            channel[2], dilation=dilations[3], padding=dilations[3], dilated=True, dropout_prob=0.1, relu=encoder_relu)


        self.transposed4_conv = nn.ConvTranspose2d(
            channel[2],
            channel[1],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.regular4_1 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            channel[1], padding=1, regular=True, dropout_prob=0.1, relu=decoder_relu)


        self.transposed5_conv = nn.ConvTranspose2d(
            channel[1],
            channel[0],
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.regular5_1 = RegularBottleneck(
            channel[0], padding=1, regular=True, dropout_prob=0.1, relu=decoder_relu)

        self.transposed6_conv = nn.ConvTranspose2d(
            channel[0],
            n_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                stride=1,
                bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                16,
                8,
                kernel_size=1,
                stride=1,
                bias=False), nn.BatchNorm2d(8), nn.ReLU())

        self.conv1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(8, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):

        input_size = x.size()
        x1 = self.initial_block(x)


        stage1_input_size = x1.size()
        x2 = self.downsample1_0(x1)
        x2 = self.regular1_1(x2)
        x2 = self.regular1_2(x2)
        x2 = self.regular1_3(x2)
        x2 = self.regular1_4(x2)


        stage2_input_size = x2.size()
        x3 = self.downsample2_0(x2)

        x3 = self.Depthwise2_1(x3)
        x3 = self.Atrous2_2(x3)
        x3 = self.Asymmetric2_3(x3)
        x3 = self.Atrous2_4(x3)

        x3 = self.Depthwise2_5(x3)
        x3 = self.Atrous2_6(x3)
        x3 = self.Asymmetric2_7(x3)
        x3 = self.Atrous2_8(x3)


        x3 = self.Depthwise3_1(x3)
        x3 = self.Atrous3_2(x3)
        x3 = self.Asymmetric3_3(x3)
        x3 = self.Atrous3_4(x3)

        x3 = self.Depthwise3_5(x3)
        x3 = self.Atrous3_6(x3)
        x3 = self.Asymmetric3_7(x3)
        x3 = self.Atrous3_8(x3)


        x4 = self.transposed4_conv(x3, output_size=stage2_input_size)


        x4 = x4 + x2


        x4 = self.regular4_1(x4)
        x4 = self.regular4_2(x4)


        x5 = self.transposed5_conv(x4, output_size=stage1_input_size)


        x5 = x5 + x1


        x5 = self.regular5_1(x5)


        x6 = self.transposed6_conv(x5, output_size=input_size)


        return x6

if __name__ == '__main__':
    inputs = torch.randn((1, 3, 256, 256)).cuda()
    start = time.perf_counter()
    model = LWIRST_No_Sigmoid(channel=(8, 32, 64), dilations=(2,4,8,16), kernel_size=(7,7,7,7), padding=(3,3,3,3)).cuda()

    out = model(inputs)
    end = time.perf_counter()
    FLOPs, params = profile(model, inputs=(inputs,))
    running_FPS = 1 / (end - start)
    print('running_FPS:', running_FPS)
    print('FLOPs=', str(FLOPs/1000000.0) + '{}'.format('M'))
    print('params=', str(params / 1000000.0) + '{}'.format('M'))
    print(out.size())
