"""
Wide Resnet

https://github.com/
meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

from torch import nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

from . import diagnostics


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True)


def _conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class _WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout, stride=1):
        super(_WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, kernel_size=1, stride=stride,
                    bias=True), )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    """Creates a wide residual network"""

    def __init__(self, depth, widen_factor, dropout, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = _conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(_WideBasic, nStages[1], n, dropout, 1)
        self.layer2 = self._wide_layer(_WideBasic, nStages[2], n, dropout, 2)
        self.layer3 = self._wide_layer(_WideBasic, nStages[3], n, dropout, 2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout, first_stride):
        strides = [first_stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def predict(self, x):
        """ Produce pre-softmax logits from net """
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward(self, x, y):
        out = self.predict(x)
        return F.cross_entropy(out, y, size_average=True)

    def diagnose(self, x, y, size_average=True):
        """ Calculate loss and other diagnostics e.g. accuracy"""
        out = self.predict(x)
        loss = F.cross_entropy(out, y, size_average=size_average).item()
        accuracy = diagnostics.accuracy(out, y, size_average=size_average)
        return {'loss': loss, 'accuracy': accuracy}
