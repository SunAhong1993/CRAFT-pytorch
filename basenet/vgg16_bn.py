import x2paddle
from collections import namedtuple
import paddle
import paddle.nn as nn
import x2paddle.torch2paddle as init
from x2paddle import models
from x2paddle.models import vgg_pth_urls


def init_weights(modules):
    for m in modules:
        if isinstance(m, paddle.nn.Conv2D):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, paddle.nn.BatchNorm2D):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, paddle.nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class vgg16_bn(paddle.nn.Layer):

    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        vgg_pth_urls['vgg16_bn'] = \
        vgg_pth_urls['vgg16_bn'].replace('https://', 'http://')
        vgg_pretrained_features = models.vgg16_bn_pth(pretrained=pretrained
            ).features
        self.slice1 = paddle.nn.Sequential()
        self.slice2 = paddle.nn.Sequential()
        self.slice3 = paddle.nn.Sequential()
        self.slice4 = paddle.nn.Sequential()
        self.slice5 = paddle.nn.Sequential()
        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        self.slice5 = paddle.nn.Sequential(nn.MaxPool2D(kernel_size=3,
            stride=1, padding=1), nn.Conv2D(512, 1024, kernel_size=3,
            padding=6, dilation=6), nn.Conv2D(1024, 1024, kernel_size=1))
        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())
        init_weights(self.slice5.modules())
        if freeze:
            for param in self.slice1.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple('VggOutputs', ['fc7', 'relu5_3', 'relu4_3',
            'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out
