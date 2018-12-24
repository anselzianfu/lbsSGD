"""
LeNet implementation.

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied
to document recognition. Proceedings of the IEEE, november 1998.
"""

from torch import nn
import torch.nn.functional as F

from . import diagnostics


class LeNet(nn.Module):
    """Outputs logits directly.
    Note original lenet uses 32x32 input, this one accepts 28x28"""

    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        # assumes one channel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        assert num_classes == 10, num_classes
        self.fc3 = nn.Linear(84, 10)
        # TODO: shouldn't we init or something

    def predict(self, x):
        """ Produce pre-softmax logits from net """
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def forward(self, x, y):
        out = self.predict(x)
        return F.cross_entropy(out, y, size_average=False)

    def diagnose(self, x, y, size_average=True):
        """ Calculate loss and other diagnostics e.g. accuracy"""
        out = self.predict(x)
        loss = F.cross_entropy(out, y, size_average=size_average).item()
        accuracy = diagnostics.accuracy(out, y, size_average=size_average)
        return {'loss': loss, 'accuracy': accuracy}
