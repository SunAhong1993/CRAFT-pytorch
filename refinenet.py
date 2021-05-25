"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""
import x2paddle
from x2paddle import torch2paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import to_tensor
from basenet.vgg16_bn import init_weights


class RefineNet(nn.Layer):

    def __init__(self):
        super(RefineNet, self).__init__()
        self.last_conv = nn.Sequential(nn.Conv2D(34, 64, kernel_size=3,
            padding=1), nn.BatchNorm2D(64), x2paddle.torch2paddle.ReLU(
            inplace=True), nn.Conv2D(64, 64, kernel_size=3, padding=1), nn.
            BatchNorm2D(64), x2paddle.torch2paddle.ReLU(inplace=True), nn.
            Conv2D(64, 64, kernel_size=3, padding=1), nn.BatchNorm2D(64),
            x2paddle.torch2paddle.ReLU(inplace=True))
        self.aspp1 = nn.Sequential(nn.Conv2D(64, 128, kernel_size=3,
            dilation=6, padding=6), nn.BatchNorm2D(128), x2paddle.
            torch2paddle.ReLU(inplace=True), nn.Conv2D(128, 128,
            kernel_size=1), nn.BatchNorm2D(128), x2paddle.torch2paddle.ReLU
            (inplace=True), nn.Conv2D(128, 1, kernel_size=1))
        self.aspp2 = nn.Sequential(nn.Conv2D(64, 128, kernel_size=3,
            dilation=12, padding=12), nn.BatchNorm2D(128), x2paddle.
            torch2paddle.ReLU(inplace=True), nn.Conv2D(128, 128,
            kernel_size=1), nn.BatchNorm2D(128), x2paddle.torch2paddle.ReLU
            (inplace=True), nn.Conv2D(128, 1, kernel_size=1))
        self.aspp3 = nn.Sequential(nn.Conv2D(64, 128, kernel_size=3,
            dilation=18, padding=18), nn.BatchNorm2D(128), x2paddle.
            torch2paddle.ReLU(inplace=True), nn.Conv2D(128, 128,
            kernel_size=1), nn.BatchNorm2D(128), x2paddle.torch2paddle.ReLU
            (inplace=True), nn.Conv2D(128, 1, kernel_size=1))
        self.aspp4 = nn.Sequential(nn.Conv2D(64, 128, kernel_size=3,
            dilation=24, padding=24), nn.BatchNorm2D(128), x2paddle.
            torch2paddle.ReLU(inplace=True), nn.Conv2D(128, 128,
            kernel_size=1), nn.BatchNorm2D(128), x2paddle.torch2paddle.ReLU
            (inplace=True), nn.Conv2D(128, 1, kernel_size=1))
        init_weights(self.last_conv.modules())
        init_weights(self.aspp1.modules())
        init_weights(self.aspp2.modules())
        init_weights(self.aspp3.modules())
        init_weights(self.aspp4.modules())

    def forward(self, y, upconv4):
        refine = torch2paddle.concat([y.permute(0, 3, 1, 2), upconv4], dim=1)
        refine = self.last_conv(refine)
        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)
        out = aspp1 + aspp2 + aspp3 + aspp4
        return out.permute(0, 2, 3, 1)
