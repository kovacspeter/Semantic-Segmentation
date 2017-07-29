import torch.nn as nn
import torch.nn.functional as F

class Conv2d_BatchNorm_ReLU(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(int(n_filters))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, average=False):
        super().__init__()

        self.loss = nn.NLLLoss2d(weight, size_average=average)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)