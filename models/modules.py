import torch.nn as nn

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


