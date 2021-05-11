import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from Blanced_attention import BlancedAttention

def swish(x):
    return x * F.sigmoid( x )

def conv(inp, oup, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = ((kernel_size -1) * dilation + 1) // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
    )

def conv_bn( inp, oup, kernel_size, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True, bn = True ):
    modules = []
    modules.append( nn.Conv2d(inp, oup, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias) )
    if bn == True : modules.append( nn.BatchNorm2d( oup ) )
    return nn.Sequential(*modules)

class ConvBNSwish(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True, bn = True ):
        super(ConvBNSwish, self).__init__()
        self.body = conv_bn( inp, oup, kernel_size, stride , padding , dilation , groups, bias, bn )

    def forward(self, x ):
        return swish( self.body( x ) )


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, inp, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        modules = []
        for i in range(2):
            modules.append(conv(inp, inp, kernel_size, bias=bias))
            if bn: modules.append(nn.BatchNorm2d(inp))
            if i == 0: modules.append(act)

        self.body = nn.Sequential(*modules)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class ResBlock_blanced_attention(nn.Module):
    def __init__(
        self, inp, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock_blanced_attention, self).__init__()
        modules = []
        for i in range(2):
            modules.append(conv(inp, inp, kernel_size, bias=bias))
            if bn: modules.append(nn.BatchNorm2d(inp))
            if i == 0: modules.append(act)

        self.body = nn.Sequential(*modules)
        self.res_scale = res_scale
        self.attention=BlancedAttention(inp)

    def forward(self, x):
        res = self.attention(self.body(x)).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale, inp, bn=False, act=False, bias=True, choice=0):

        modules = []
        if choice == 0: #subpixel
           if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
               for _ in range(int(math.log(scale, 2))):
                   modules.append(conv(inp, 4 * inp, 3, bias=bias))
                   modules.append(nn.PixelShuffle(2))
                   if bn: modules.append(nn.BatchNorm2d(inp))
                   if act: modules.append(act())
           elif scale == 3:
               modules.append(conv(inp, 9 * inp, 3, bias=bias))
               modules.append(nn.PixelShuffle(3))
               if bn: modules.append(nn.BatchNorm2d(inp))
               if act: modules.append(act())
           else:
               raise NotImplementedError
        elif choice == 1: #decov
           modules.append(nn.ConvTranspose2d(inp, inp, scale, stride=scale))
        else: #bilinear
           modules.append(nn.Upsample(mode='bilinear', scale_factor=scale, align_corners=True))

        super(Upsampler, self).__init__(*modules)
