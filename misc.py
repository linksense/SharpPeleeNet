import torch
import torch.nn as nn

from collections import OrderedDict


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SCSEBlock: Spatial-Channel Squeeze & Excitation (SCSE)
#            namely, Spatial-wise and Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class SCSEBlock(nn.Module):
    def __init__(self, channel, reduct_ratio=16):
        super(SCSEBlock, self).__init__()

        self.channel_se = nn.Sequential(OrderedDict([("avgpool", nn.AdaptiveAvgPool2d(1)),
                                                     ("linear1", nn.Conv2d(channel, channel // reduct_ratio,
                                                                           kernel_size=1, stride=1, padding=0)),
                                                     ("relu", nn.ReLU(inplace=True)),
                                                     ("linear2", nn.Conv2d(channel // reduct_ratio, channel,
                                                                           kernel_size=1, stride=1, padding=0)),
                                                     ("score", nn.Sigmoid())]))

        self.spatial_se = nn.Sequential(OrderedDict([("conv", nn.Conv2d(channel, 1, kernel_size=1, stride=1,
                                                                        padding=0, bias=False)),
                                                     ("score", nn.Sigmoid())]))

    def forward(self, x):
        inputs = x.clone()

        chn_se = self.channel_se(x).exp()
        spa_se = self.spatial_se(x).exp()

        return torch.mul(torch.mul(inputs, chn_se), spa_se)


class SCSABlock(nn.Module):
    def __init__(self, in_chns, reduct_ratio=16, is_res=True, scale=0.25):
        super(SCSABlock, self).__init__()
        self.is_res = is_res
        self.scale = scale
        # ------------------------------------------ #
        # Channel-wise Attention Model
        # ------------------------------------------ #
        self.ch_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_max_pool = nn.AdaptiveMaxPool2d(1)
        self.se_block = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                                kernel_size=1, stride=1, padding=0),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_chns // reduct_ratio, in_chns,
                                                kernel_size=1, stride=1, padding=0))

        self.sp_conv = nn.Sequential(nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
                                     nn.BatchNorm2d(1))

        # self.sp_conv = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
        #                                        kernel_size=1, stride=1, padding=0, bias=False),
        #                              nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
        #                                        kernel_size=3, stride=1, padding=dilation,
        #                                        dilation=dilation, bias=False),
        #                              nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
        #                                        kernel_size=3, stride=1, padding=dilation,
        #                                        dilation=dilation, bias=False),
        #                              nn.Conv2d(in_chns // reduct_ratio, 1, kernel_size=1,
        #                                        stride=1, padding=0, bias=False),
        #                              nn.BatchNorm2d(1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ------------------------------------------ #
        # 1. Channel-wise Attention Model
        # ------------------------------------------ #
        res = x
        avg_p = self.se_block(self.ch_avg_pool(x))
        max_p = self.se_block(self.ch_max_pool(x))

        ch_att = torch.mul(x, self.sigmoid(avg_p + max_p).exp())

        # ------------------------------------------ #
        # 2. Spatial-wise Attention Model
        # ------------------------------------------ #
        ch_avg = torch.mean(ch_att, dim=1, keepdim=True)
        ch_max = torch.max(ch_att, dim=1, keepdim=True)[0]

        sp_att = torch.mul(ch_att, self.sigmoid(self.sp_conv(torch.cat([ch_avg, ch_max], dim=1))).exp())

        if self.is_res:
            return sp_att + res

        return sp_att


class PBCSABlock(nn.Module):
    def __init__(self, in_chns, reduct_ratio=16, dilation=4, is_res=True, scale=1.0):
        super(PBCSABlock, self).__init__()
        self.is_res = is_res
        self.scale = scale
        # ------------------------------------------ #
        # Channel-wise Attention Model
        # ------------------------------------------ #
        self.ch_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ch_max_pool = nn.AdaptiveMaxPool2d(1)

        self.se_block = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                                kernel_size=1, stride=1, padding=0),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_chns // reduct_ratio, in_chns,
                                                kernel_size=1, stride=1, padding=0))
        # self.ch_conv = nn.Sequential(nn.Conv2d(in_chns, in_chns,
        #                                        kernel_size=1, stride=1, padding=0),
        #                              nn.BatchNorm2d(in_chns))

        self.sp_conv = nn.Sequential(nn.Conv2d(in_chns, in_chns // reduct_ratio,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
                                               kernel_size=3, stride=1, padding=dilation,
                                               dilation=dilation, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, in_chns // reduct_ratio,
                                               kernel_size=3, stride=1, padding=dilation,
                                               dilation=dilation, bias=False),
                                     nn.Conv2d(in_chns // reduct_ratio, 1, kernel_size=1,
                                               stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ------------------------------------------ #
        # 1. Channel-wise Attention Model
        # ------------------------------------------ #
        res = x
        avg_p = self.se_block(self.ch_avg_pool(x))
        max_p = self.se_block(self.ch_max_pool(x))
        ch_att = avg_p + max_p

        ch_att = torch.mul(x, self.sigmoid(ch_att).exp())

        # ------------------------------------------ #
        # 2. Spatial-wise Attention Model
        # ------------------------------------------ #
        sp_att = torch.mul(x, self.sigmoid(self.sp_conv(x)).exp())

        if self.is_res:
            return sp_att + res + ch_att

        return sp_att + ch_att
