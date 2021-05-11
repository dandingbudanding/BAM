import os
import sys
import math
import torch
import torch.nn as nn
import numpy as np
import MDSR.common as common
from Blanced_attention import BlancedAttention


class MDSR(nn.Module):
    def __init__(self, scale):
        super(MDSR, self).__init__()

        # args
        self.scale_list = [scale]
        input_channel = 3
        output_channel = 3
        num_block = 32
        inp = 64
        rgb_range = 255
        res_scale = 0.1
        act = nn.ReLU(True)
        # act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

        # head
        self.head = nn.Sequential(common.conv(3, inp, input_channel))

        # pre_process
        self.pre_process = nn.ModuleDict([str(scale),
                                          nn.Sequential(common.ResBlock(inp, bias=True, act=act, res_scale=res_scale),
                                                        common.ResBlock(inp, bias=True, act=act, res_scale=res_scale))]
                                         for scale in self.scale_list)

        # body
        self.body = nn.Sequential(
            *[common.ResBlock(inp, bias=True, act=act, res_scale=res_scale) for _ in range(num_block)])
        self.body.add_module(str(num_block), common.conv(inp, inp, 3))

        # upsample
        self.upsample = nn.ModuleDict(
            [str(scale), common.Upsampler(scale, inp, act=False, choice=0)] for scale in self.scale_list)

        # tail
        self.tail = nn.Sequential(common.conv(inp, 3, output_channel))

        self.sub_mean = common.MeanShift(rgb_range, sign=-1)
        self.add_mean = common.MeanShift(rgb_range, sign=1)


    def forward(self, x,scale):
        scale_id = str(scale)
        # x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[scale_id](x)

        res = self.body(x)
        res += x

        x = self.upsample[scale_id](res)
        x = self.tail(x)
        # x = self.add_mean(x)
        return x

    def forward_pred(self, x, scale_id):
        scale = torch.from_numpy(np.array([scale_id]))
        return self.forward(x, scale)

    def _initialize_weights(self):
        for (name, m) in self.named_modules():
            if name.endswith('_mean'):
                print('Do not initilize {}'.format(name))
            elif isinstance(m, nn.Conv2d) and isinstance(m, nn.ReLU):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d) and isinstance(m, nn.LeakyReLU):
                nn.init.kaiming_normal_(m.weight, a=0.05, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MDSR_blanced_attention(nn.Module):
    def __init__(self, scale):
        super(MDSR_blanced_attention, self).__init__()

        # args
        self.scale_list = [scale]
        input_channel = 3
        output_channel = 3
        num_block = 32
        inp = 64
        rgb_range = 255
        res_scale = 0.1
        act = nn.ReLU(True)
        # act = nn.LeakyReLU(negative_slope=0.05, inplace=True)

        # head
        self.head = nn.Sequential(common.conv(3, inp, input_channel))

        # pre_process
        self.pre_process = nn.ModuleDict([str(scale),
                                          nn.Sequential(common.ResBlock(inp, bias=True, act=act, res_scale=res_scale),
                                                        common.ResBlock(inp, bias=True, act=act, res_scale=res_scale))]
                                         for scale in self.scale_list)

        # body
        self.body = nn.Sequential(
            *[common.ResBlock_blanced_attention(inp, bias=True, act=act, res_scale=res_scale) for _ in range(num_block)])
        self.body.add_module(str(num_block), common.conv(inp, inp, 3))

        # upsample
        self.upsample = nn.ModuleDict(
            [str(scale), common.Upsampler(scale, inp, act=False, choice=0)] for scale in self.scale_list)

        # tail
        self.tail = nn.Sequential(common.conv(inp, 3, output_channel))

        self.sub_mean = common.MeanShift(rgb_range, sign=-1)
        self.add_mean = common.MeanShift(rgb_range, sign=1)


    def forward(self, x,scale):
        scale_id = str(scale)
        # x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[scale_id](x)

        res = self.body(x)
        res += x

        x = self.upsample[scale_id](res)
        x = self.tail(x)
        # x = self.add_mean(x)
        return x

    def forward_pred(self, x, scale_id):
        scale = torch.from_numpy(np.array([scale_id]))
        return self.forward(x, scale)

    def _initialize_weights(self):
        for (name, m) in self.named_modules():
            if name.endswith('_mean'):
                print('Do not initilize {}'.format(name))
            elif isinstance(m, nn.Conv2d) and isinstance(m, nn.ReLU):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d) and isinstance(m, nn.LeakyReLU):
                nn.init.kaiming_normal_(m.weight, a=0.05, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()