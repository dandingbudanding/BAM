import torch.nn as nn
from collections import OrderedDict
import torch
import torch.nn.functional as F

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.PReLU(in_planes // ratio)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(max_out)
        return self.sigmoid(x)

#调用BlancedAttention 模块即可
class BlancedAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(BlancedAttention, self).__init__()

        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention()
        # self.conv=nn.Conv2d(in_planes*2,in_planes,1)
        # self.norm=nn.BatchNorm2d(in_planes)

    def forward(self, x):
        ca_ch = self.ca(x)
        sa_ch = self.sa(x)
        out=ca_ch.mul(sa_ch)*x
        # out_fused = self.conv(torch.cat([ca_ch, sa_ch], dim=1))
        return out

class ChannelAttention_maxpool(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_maxpool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.PReLU(in_planes // ratio)
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out+max_out
        return self.sigmoid(out)

class SpatialAttention_averagepool(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_averagepool, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BlancedAttention_CAM_SAM_ADD(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(BlancedAttention_CAM_SAM_ADD, self).__init__()

        self.ca = ChannelAttention_maxpool(in_planes, reduction)
        self.sa = SpatialAttention_averagepool()
        # self.conv=nn.Conv2d(in_planes*2,in_planes,1)
        # self.norm=nn.BatchNorm2d(in_planes)

    def forward(self, x):
        ca_ch = self.ca(x)
        sa_ch = self.sa(x)
        out=ca_ch.mul(sa_ch)*x
        # out_fused = self.conv(torch.cat([ca_ch, sa_ch], dim=1))
        return out

# class ChannelAttention2(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention2, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc1   = nn.Conv2d(2*in_planes, 2*in_planes // ratio, 1, bias=False)
#         self.relu1 = nn.PReLU(2*in_planes // ratio)
#         self.fc2   = nn.Conv2d(2*in_planes // ratio, in_planes, 1, bias=False)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg=self.avg_pool(x)
#         max=self.max_pool(x)
#         out = self.fc2(self.relu1(self.fc1(torch.cat([avg,max],1))))
#         return self.sigmoid(out)
#
# class SpatialAttention2(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention2, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         # std_out=torch.std(x, dim=1, keepdim=True)
#         # real_out=avg_out+std_out
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
#
# class BlancedAttention3(nn.Module):
#     def __init__(self, in_planes, reduction=16):
#         super(BlancedAttention3, self).__init__()
#
#         self.ca = ChannelAttention2(in_planes, reduction)
#         self.sa = SpatialAttention2()
#         self.conv=nn.Conv2d(in_planes*2,in_planes,1)
#
#     def forward(self, x):
#         ca_ch = self.ca(x)* x
#         sa_ch = self.sa(x) * x
#         out_fused = self.conv(torch.cat([ca_ch, sa_ch], dim=1))
#         return out_fused
#
# ####demo
# def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
#     padding = int((kernel_size - 1) / 2) * dilation
#     return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
#                      groups=groups)
# def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
#     act_type = act_type.lower()
#     if act_type == 'relu':
#         layer = nn.ReLU(inplace)
#     elif act_type == 'lrelu':
#         layer = nn.LeakyReLU(neg_slope, inplace)
#     elif act_type == 'prelu':
#         layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
#     else:
#         raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
#     return layer
# class IMDModule_blancedattention(nn.Module):
#     def __init__(self, in_channels, distillation_rate=0.25):
#         super(IMDModule_blancedattention, self).__init__()
#         self.distilled_channels = int(in_channels * distillation_rate)
#         self.remaining_channels = int(in_channels - self.distilled_channels)
#         self.c1 = conv_layer(in_channels, in_channels, 3)
#         self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
#         self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
#         self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
#         self.act = activation('lrelu', neg_slope=0.05)
#         self.attention = BlancedAttention(self.distilled_channels*4,reduction=16)#此处调用
#
#     def forward(self, input):
#         out_c1 = self.act(self.c1(input))
#         distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
#         out_c2 = self.act(self.c2(remaining_c1))
#         distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
#         out_c3 = self.act(self.c3(remaining_c2))
#         distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
#         out_c4 = self.c4(remaining_c3)
#         out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
#         out_fused = self.attention(out) + input #此处调用
#         return out_fused