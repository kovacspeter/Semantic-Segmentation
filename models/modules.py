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


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss
