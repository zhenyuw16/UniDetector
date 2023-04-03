# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16
from collections import OrderedDict
from mmdet.models.backbones import ResNet
from mmdet.models.builder import SHARED_HEADS
from mmdet.models.utils import ResLayer as _ResLayer

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)), 
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


@SHARED_HEADS.register_module()
class CLIPResLayer(nn.Module):

    def __init__(self, layers, output_dim=512, input_resolution=224, width=64, pretrained=None, **kwargs):
        super(CLIPResLayer, self).__init__( )
        if width == 64:
            self._inplanes = 1024
        else:
            self._inplanes = 1280
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
    
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)


    @auto_fp16()
    def forward(self, x):
        out = self.layer4(x)
        return out

    def train(self, mode=True):
        super(CLIPResLayer, self).train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
