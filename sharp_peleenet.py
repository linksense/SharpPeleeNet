import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from collections import OrderedDict
from modules.misc import SCSABlock, PBCSABlock


class ChannelShuffle(nn.Module):
    def __init__(self, in_chns, out_chns, kernel_size=1, groups=3):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
        self.fusion = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=out_chns,
                                              kernel_size=kernel_size, stride=1, padding=0, bias=False),
                                    nn.BatchNorm2d(out_chns),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Channel shuffle: [N,C,H,W] -> [N,g,C//g,H,W] -> [N,C//g,g,H,w] -> [N,C,H,W]
        :param x:
        :return:
        """
        batch_size, channels, height, width = x.size()
        x.view(batch_size, self.groups, channels // self.groups, height, width).permute(0, 2, 1, 3, 4).contiguous().view(
            batch_size, channels, height, width)

        return self.fusion(x)


class StemBlock(nn.Module):
    def __init__(self, in_chns=3, out_chns=32):
        super(StemBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=out_chns,
                                             kernel_size=3, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(out_chns),
                                   nn.ReLU(inplace=True))

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=out_chns, out_channels=int(out_chns//2),
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(int(out_chns//2)),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels=int(out_chns//2), out_channels=out_chns,
                                               kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(out_chns),
                                     nn.ReLU(inplace=True)
                                     )
        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.chn_shuffle = ChannelShuffle(in_chns=out_chns * 3, out_chns=out_chns, kernel_size=1, groups=3)

    def forward(self, x):
        x = self.conv1(x)
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        return self.chn_shuffle(torch.cat([x0, x1, x2], dim=1))


class TransitionBlock(nn.Module):
    def __init__(self, chns, reduce_ratio=0.5):
        super(TransitionBlock, self).__init__()
        self.mid_chns = int(chns * reduce_ratio)

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=chns, out_channels=self.mid_chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(self.mid_chns),
                                     nn.ReLU(inplace=True),

                                     nn.Conv2d(in_channels=self.mid_chns, out_channels=self.mid_chns,
                                               kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(self.mid_chns),
                                     nn.ReLU(inplace=True),

                                     nn.Conv2d(in_channels=self.mid_chns, out_channels=chns,
                                               kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(chns),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.branch3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.chn_shuffle = ChannelShuffle(in_chns=chns * 3, out_chns=chns, kernel_size=1, groups=3)

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)
        x2 = self.branch3(x)
        return self.chn_shuffle(torch.cat([x0, x1, x2], dim=1))


class TwoWayDenseBlock(nn.Module):
    def __init__(self, in_chns=32, mid_chns=16, out_chns=16, with_relu=False):
        super(TwoWayDenseBlock, self).__init__()
        self.with_relu = with_relu

        if with_relu:
            self.relu = nn.ReLU(inplace=True)

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=mid_chns,
                                                  kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ReLU(inplace=True),

                                     nn.Conv2d(in_channels=mid_chns, out_channels=out_chns,
                                               kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(out_chns))

        self.branch2 = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=mid_chns,
                                                  kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(mid_chns),
                                     nn.ReLU(inplace=True),

                                     nn.Conv2d(in_channels=mid_chns, out_channels=out_chns,
                                               kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(out_chns),
                                     nn.ReLU(inplace=True),

                                     nn.Conv2d(in_channels=out_chns, out_channels=out_chns,
                                               kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(out_chns))

    def forward(self, x):
        x0 = self.branch1(x)
        x1 = self.branch2(x)

        out = torch.cat([2.0 * x0 - x1, x, x1], dim=1)

        if self.with_relu:
            out = self.relu(out)
        return out


class SharpPeleeNet(nn.Module):
    def __init__(self, num_classes=1000, in_size=(224, 224), growth_rate=32):
        super(SharpPeleeNet, self).__init__()
        assert in_size[0] % 32 == 0
        assert in_size[1] % 32 == 0
        self.in_size = in_size

        self.last_channel = 704

        self.num_chns = [32, 0, 0, 0, 0]
        self.repeat = [3, 4, 8, 6]
        self.width_multi = [1, 2, 4, 4]

        self.half_growth_rate = int(growth_rate//2)

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoders
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        self.entry = StemBlock(in_chns=3, out_chns=self.num_chns[0])

        in_chns = self.num_chns[0]

        mid_chns = int(self.half_growth_rate * self.width_multi[0] / 4) * 4
        encode_block1 = OrderedDict()

        for i in range(self.repeat[0]):
            encode_block1["dens_{}".format(i)] = TwoWayDenseBlock(in_chns=in_chns, mid_chns=mid_chns,
                                                                  out_chns=self.half_growth_rate, with_relu=True)
            in_chns += 32

        # encode_block1["scse2"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block1["dropout"] = nn.Dropout2d(p=0.1)
        self.encoder1 = nn.Sequential(encode_block1)
        # -------------------------- 1/4 End -------------------------- #

        self.num_chns[1] = in_chns
        self.transition1 = TransitionBlock(chns=in_chns)

        mid_chns = int(self.half_growth_rate * self.width_multi[1] / 4) * 4
        encode_block2 = OrderedDict()
        for i in range(self.repeat[1]):
            encode_block2["dens_{}".format(i)] = TwoWayDenseBlock(in_chns=in_chns, mid_chns=mid_chns,
                                                                  out_chns=self.half_growth_rate, with_relu=True)
            in_chns += 32

        # encode_block2["scse"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block2["dropout"] = nn.Dropout2d(p=0.1)
        self.encoder2 = nn.Sequential(encode_block2)
        # -------------------------- 1/8 End -------------------------- #

        self.num_chns[2] = in_chns
        self.transition2 = TransitionBlock(chns=in_chns)

        mid_chns = int(self.half_growth_rate * self.width_multi[2] / 4) * 4
        encode_block3 = OrderedDict()

        for i in range(self.repeat[2]):
            encode_block3["dens_{}".format(i)] = TwoWayDenseBlock(in_chns=in_chns, mid_chns=mid_chns,
                                                                  out_chns=self.half_growth_rate, with_relu=True)
            in_chns += 32

        # encode_block3["scse"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block3["dropout"] = nn.Dropout2d(p=0.1)
        self.encoder3 = nn.Sequential(encode_block3)
        # -------------------------- 1/16 End -------------------------- #

        self.num_chns[3] = in_chns
        self.transition3 = TransitionBlock(chns=in_chns)

        mid_chns = int(self.half_growth_rate * self.width_multi[3] / 4) * 4
        encode_block4 = OrderedDict()

        for i in range(self.repeat[3]):
            encode_block4["dens_{}".format(i)] = TwoWayDenseBlock(in_chns=in_chns, mid_chns=mid_chns,
                                                                  out_chns=self.half_growth_rate, with_relu=True)
            in_chns += 32

        # encode_block4["scse"] = PBCSABlock(in_chns=in_chns, reduct_ratio=16, dilation=4, is_res=True)
        # encode_block4["dropout"] = nn.Dropout2d(p=0.1)
        self.encoder4 = nn.Sequential(encode_block4)
        self.num_chns[4] = in_chns
        # -------------------------- 1/32 End -------------------------- #

        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 2. Classifier
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        self.final_feat = nn.Sequential(nn.Conv2d(in_channels=in_chns, out_channels=self.last_channel,
                                                  kernel_size=1, stride=1, padding=0,  bias=False),
                                        nn.BatchNorm2d(num_features=self.last_channel),
                                        nn.ReLU(inplace=True)  # ,
                                        # PBCSABlock(in_chns=self.last_channel, reduct_ratio=16,
                                        #            dilation=2, is_res=True)
										)

        self.avg_pool = nn.AvgPool2d(kernel_size=(self.in_size[0] // 32, self.in_size[1] // 32))
        self.linear = nn.Sequential(nn.Dropout(p=0.10), nn.Linear(self.last_channel, num_classes))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0.0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0.0, mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias.data, 0.0)

    def __classifier(self, x):
        x = self.avg_pool(x)
        x = x.view(-1, self.last_channel)
        x = self.linear(x)
        return x

    def forward(self, x):
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        # 1. Encoder
        # +++++++++++++++++++++++++++++++++++++++++++++++ #
        x = self.entry(x)            # [N, 128, H/4, W/4]
        x = self.encoder1(x)         # [N, 320, H/4, W/4]         <----

        x = self.transition1(x)      # [N, 192, H/8, W/8]
        x = self.encoder2(x)         # [N, 576, H/8, W/8]         <----

        x = self.transition2(x)      # [N, 576, H/16, W/16]
        x = self.encoder3(x)         # [N, 768, H/16, W/16]       <----

        x = self.transition3(x)      # [N, 576, H/32, W/32]
        x = self.encoder4(x)         # [N, 768, H/32, W/32]       <----
        x = self.final_feat(x)

        x = self.__classifier(x)
        return x


if __name__ == "__main__":
    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2,1"

    net_h, net_w = 224, 224
    dummy_in = torch.randn(1, 3, net_h, net_w).cuda().requires_grad_()
    # dummy_target = torch.ones(1, net_h, net_w).cuda().long()

    model = SharpPeleeNet(num_classes=1000, in_size=(net_h, net_w)).cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.90, weight_decay=5e-4)

    while True:
        model.train()

        start_time = time.time()
        dummy_final = model(dummy_in)
        end_time = time.time()
        print("Inference time: {}s".format(end_time - start_time))

