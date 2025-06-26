from torch import nn
import torch
from torch.nn import functional as F
from model.attention import UAFM_SpAtten, UAFM_ChAtten, UAFM_CS


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.InstanceNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # self.cs = UAFM_CS(skip_channels, in_channels, out_channels, ksize=3)
        self.sa = UAFM_SpAtten(skip_channels, in_channels, out_channels, ksize=3)

    def forward(self, x, skip):
        # x = F.interpolate(x, size=skip.size()[2:], mode="bilinear")
        x = self.sa(skip, x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()

        self.depth = len(encoder_channels)
        convs = dict()

        for d in range(self.depth - 1):
            if d == self.depth - 2:
                convs["conv{}".format(d)] = DecoderBlock(encoder_channels[d + 1],
                                                         encoder_channels[d],
                                                         decoder_channels[d])
            else:
                convs["conv{}".format(d)] = DecoderBlock(decoder_channels[d + 1],
                                                         encoder_channels[d],
                                                         decoder_channels[d])

        self.convs = nn.ModuleDict(convs)

        # self.final = nn.Sequential(
        #     nn.Conv2d(decoder_channels[0], 1, 3, padding=1),
        #     nn.Sigmoid())

    def forward(self, features):

        for d in range(self.depth - 2, -1, -1):
            features[d] = self.convs["conv{}".format(d)](features[d + 1], features[d])

        return features
        # return self.final(features[0])


class UnetMix(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()

        self.mean_decoder = UnetDecoder(encoder_channels, decoder_channels)
        self.std_decoder = UnetDecoder(encoder_channels, decoder_channels)
        # self.enc = encoder_channels
        # self.dec = decoder_channels

    def forward(self, features):

        mean = self.mean_decoder(features[:])
        std = self.std_decoder(features)
        return mean, std


class Identity(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
    ):
        super().__init__()
        convs = []
        for ec, dc in zip(encoder_channels, decoder_channels):
            convs.append(nn.Conv2d(ec, dc, 1))
        self.convs = nn.ModuleList(convs)

    def forward(self, features):
        return [c(f) for f, c in zip(features, self.convs)]
