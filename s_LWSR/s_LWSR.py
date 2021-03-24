import s_LWSR.common as common

import torch.nn as nn

import torch
from Blanced_attention import BlancedAttention

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
## Channel Attention (CA) Layer
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class ResBlock2(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock2, self).__init__()
        self.conv = conv(n_feat*2, n_feat, 1, bias=bias)

        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        ori = self.conv(x)
        res = self.body(ori).mul(self.res_scale)
        res += ori

        return res

## Residual Group (RG)
# class ResidualGroup(nn.Module):
#     def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
#         super(ResidualGroup, self).__init__()
#         modules_body = []
#         modules_body = [
#             RCAB(
#                 conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
#             for _ in range(n_resblocks)]
#         modules_body.append(conv(n_feat, n_feat, kernel_size))
#         self.body = nn.Sequential(*modules_body)
#
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res

## Residual Channel Attention Network (RCAN)
class LWSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(LWSR, self).__init__()

#        n_resgroups = args.n_resgroups
#        n_resblocks = args.n_resblocks
        n_feats = args.n_feats_s_LWSR
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        # define body module

  #  def __init__(
  #      self, conv, n_feat, kernel_size,
  #      bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        self.feat2_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat2_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat2_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)


        self.feat3_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat3_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat3_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

        self.feat4_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat4_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat4_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)


        self.feat5_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat5_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat5_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)


        self.feat6_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat6_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

        self.conv1 = conv(6*n_feats, n_feats, 1, bias=False)

        self.feat7_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat7_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.conv7 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
     #   self.conv7 = ConvBlock(base_filter2, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.feat8_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat8_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.conv8 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
     #   self.conv8 = ConvBlock(base_filter2, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.feat9_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat9_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
     #   self.conv9 = nn.conv2d(base_filter2, base_filter, kernel_size = 1, stride = 1, padding = 0)
        self.conv9 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)

        self.feat10_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat10_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
     #   self.conv10 = nn.conv2d(base_filter2, base_filter, kernel_size = 1, stride = 1, padding = 0)
        self.conv10 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)


        # self.attention = BlancedAttention(n_feats)


  #      self.conv11 = torch.nn.Conv2d(base_filter*2, base_filter, 1, 1, 0)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]


        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x1 = self.head(x)


        feat2_1 = self.feat2_1(x1)
        feat2_2 = self.feat2_2(feat2_1)
        feat2_3 = self.feat2_3(feat2_2)

        feat3_1 = self.feat3_1(torch.add(feat2_1, feat2_3))
        feat3_2 = self.feat3_2(feat3_1)
        feat3_3 = self.feat3_3(feat3_2)

        feat4_1 = self.feat4_1(torch.add(feat3_1, feat3_3))
        feat4_2 = self.feat4_2(feat4_1)
        feat4_3 = self.feat4_3(feat4_2)

        feat5_1 = self.feat5_1(torch.add(feat4_1, feat4_3))
        feat5_2 = self.feat5_2(feat5_1)
        feat5_3 = self.feat5_3(feat5_2)

        feat6_1 = self.feat6_1(torch.add(feat5_1, feat5_3))
        feat6_2 = self.feat6_2(feat6_1)

        bool = torch.cat([x1, feat3_1, feat4_1, feat5_1, feat6_1, feat6_2], 1)
        conv1 = self.conv1(bool)

        concat_7 = torch.cat([feat6_2, 0.5*feat5_3 + 0.5*conv1],1)
        feat7_1 = self.feat7_1(concat_7)
        feat7_2 = self.feat7_2(feat7_1)
        conv7 = self.conv7(feat7_2)

        concat_8 = torch.cat([conv7, 0.5*feat4_3+0.5*conv1],1)
        feat8_1 = self.feat8_1(concat_8)
        feat8_2 = self.feat8_2(feat8_1)
        conv8 = self.conv8(feat8_2)

        concat_9 = torch.cat([conv8, 0.5*feat3_3+0.5*conv1],1)
        feat9_1 = self.feat9_1(concat_9)
        feat9_2 = self.feat9_2(feat9_1)
        conv9 = self.conv9(feat9_2)

        concat_10 = torch.cat([conv9, 0.5*feat2_3+0.5*conv1],1)
        feat10_1 = self.feat10_1(concat_10)
        feat10_2 = self.feat10_2(feat10_1)
        conv10 = self.conv10(feat10_2)

        # conv10=self.attention(conv10)

        conv10 += x1

        x = self.tail(conv10)
        # x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

## Residual Channel Attention Network (RCAN)
class LWSR_blanced_attention(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(LWSR_blanced_attention, self).__init__()

#        n_resgroups = args.n_resgroups
#        n_resblocks = args.n_resblocks
        n_feats = args.n_feats_s_LWSR
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        # define body module

  #  def __init__(
  #      self, conv, n_feat, kernel_size,
  #      bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        self.feat2_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat2_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat2_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)


        self.feat3_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat3_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat3_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

        self.feat4_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat4_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat4_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)


        self.feat5_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat5_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.feat5_3 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)


        self.feat6_1 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat6_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)

        self.conv1 = conv(6*n_feats, n_feats, 1, bias=False)

        self.feat7_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat7_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.conv7 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
     #   self.conv7 = ConvBlock(base_filter2, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.feat8_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat8_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
        self.conv8 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
     #   self.conv8 = ConvBlock(base_filter2, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.feat9_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat9_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
     #   self.conv9 = nn.conv2d(base_filter2, base_filter, kernel_size = 1, stride = 1, padding = 0)
        self.conv9 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)

        self.feat10_1 = ResBlock2(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        self.feat10_2 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)
     #   self.conv10 = nn.conv2d(base_filter2, base_filter, kernel_size = 1, stride = 1, padding = 0)
        self.conv10 = ResBlock(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(False), res_scale=1)


        self.attention = BlancedAttention(n_feats)


  #      self.conv11 = torch.nn.Conv2d(base_filter*2, base_filter, 1, 1, 0)

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]


        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x1 = self.head(x)


        feat2_1 = self.feat2_1(x1)
        feat2_2 = self.feat2_2(feat2_1)
        feat2_3 = self.feat2_3(feat2_2)

        feat3_1 = self.feat3_1(torch.add(feat2_1, feat2_3))
        feat3_2 = self.feat3_2(feat3_1)
        feat3_3 = self.feat3_3(feat3_2)

        feat4_1 = self.feat4_1(torch.add(feat3_1, feat3_3))
        feat4_2 = self.feat4_2(feat4_1)
        feat4_3 = self.feat4_3(feat4_2)

        feat5_1 = self.feat5_1(torch.add(feat4_1, feat4_3))
        feat5_2 = self.feat5_2(feat5_1)
        feat5_3 = self.feat5_3(feat5_2)

        feat6_1 = self.feat6_1(torch.add(feat5_1, feat5_3))
        feat6_2 = self.feat6_2(feat6_1)

        bool = torch.cat([x1, feat3_1, feat4_1, feat5_1, feat6_1, feat6_2], 1)
        conv1 = self.conv1(bool)

        concat_7 = torch.cat([feat6_2, 0.5*feat5_3 + 0.5*conv1],1)
        feat7_1 = self.feat7_1(concat_7)
        feat7_2 = self.feat7_2(feat7_1)
        conv7 = self.conv7(feat7_2)

        concat_8 = torch.cat([conv7, 0.5*feat4_3+0.5*conv1],1)
        feat8_1 = self.feat8_1(concat_8)
        feat8_2 = self.feat8_2(feat8_1)
        conv8 = self.conv8(feat8_2)

        concat_9 = torch.cat([conv8, 0.5*feat3_3+0.5*conv1],1)
        feat9_1 = self.feat9_1(concat_9)
        feat9_2 = self.feat9_2(feat9_1)
        conv9 = self.conv9(feat9_2)

        concat_10 = torch.cat([conv9, 0.5*feat2_3+0.5*conv1],1)
        feat10_1 = self.feat10_1(concat_10)
        feat10_2 = self.feat10_2(feat10_1)
        conv10 = self.conv10(feat10_2)

        conv10=self.attention(conv10)

        conv10 += x1

        x = self.tail(conv10)
        # x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))