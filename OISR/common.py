import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Blanced_attention import BlancedAttention


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(1, 0.25), res_scale=1):
        super(ResBlock, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([-1.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True)
        self.scale5 = nn.Parameter(torch.FloatTensor([1 / 6]), requires_grad=True)



    def forward(self, x):
        yn = x
        k1 = self.relu1(x)
        k1 = self.conv1(k1)
        yn_1 = k1 * self.scale1 + yn
        k2 = self.relu2(yn_1)
        k2 = self.conv2(k2)
        yn_2 = yn + self.scale2 * k2
        yn_2 = yn_2 + k1 * self.scale3
        k3 = self.relu3(yn_2)
        k3 = self.conv3(k3)
        yn_3 = k3 + k2 * self.scale4 + k1
        yn_3 = yn_3 * self.scale5
        out = yn_3 + yn
        return out


class ResBlock_blanced_attention(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(1, 0.25), res_scale=1):
        super(ResBlock_blanced_attention, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv3 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.relu3 = nn.PReLU(n_feats, 0.25)
        self.scale1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.FloatTensor([2.0]), requires_grad=True)
        self.scale3 = nn.Parameter(torch.FloatTensor([-1.0]), requires_grad=True)
        self.scale4 = nn.Parameter(torch.FloatTensor([4.0]), requires_grad=True)
        self.scale5 = nn.Parameter(torch.FloatTensor([1 / 6]), requires_grad=True)



    def forward(self, x):
        yn = x
        k1 = self.relu1(x)
        k1 = self.conv1(k1)
        yn_1 = k1 * self.scale1 + yn
        k2 = self.relu2(yn_1)
        k2 = self.conv2(k2)
        yn_2 = yn + self.scale2 * k2
        yn_2 = yn_2 + k1 * self.scale3
        k3 = self.relu3(yn_2)
        k3 = self.conv3(k3)
        yn_3 = k3 + k2 * self.scale4 + k1
        yn_3 = yn_3 * self.scale5
        out = yn_3 + yn
        return out

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

