import torch.nn as nn
from torchvision import models

from models import modules


class SegNet_VGG16(nn.Module):

    def __init__(self, n_classes=21, n_channels=3, pretrained=True):
        """
        Init parameters are default set to match PASCAL VOC dataset

        Args:
            n_classes (int): Number of classes in dataset
            n_channels (int): Image channels
            pretrained (bool): Use pretraind VGG
            use_cuda (bool): Use cuda
        """
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.encoder = SegNet_VGG16_encoder(n_channels)
        self.decoder = SegNet_VGG16_decoder(n_classes)

        if pretrained:
            self.load_vgg16()

    def load_vgg16(self):
        layers = list(models.vgg16_bn(pretrained=True).features.children())
        index = 0

        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                self.encoder.layers[index].conv.weight.data = layer.weight.data
                self.encoder.layers[index].conv.bias.data = layer.bias.data
            elif isinstance(layer, nn.BatchNorm2d):
                self.encoder.layers[index].bn.weight.data = layer.weight.data
                self.encoder.layers[index].bn.bias.data = layer.bias.data
                index += 1

    def forward(self, x):
        encoded, indices = self.encoder(x)
        segmented = self.decoder(encoded, indices)

        return segmented

class SegNet_VGG16_encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # First block, in_channels -> 64
        self.conv1_in_to_64 = modules.Conv2d_BatchNorm_ReLU(in_channels, 64, 3, 1, 1)
        self.conv2_64_to_64 = modules.Conv2d_BatchNorm_ReLU(64, 64, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

        # Second block, 64 -> 128
        self.conv3_64_to_128 = modules.Conv2d_BatchNorm_ReLU(64, 128, 3, 1, 1)
        self.conv4_128_to_128 = modules.Conv2d_BatchNorm_ReLU(128, 128, 3, 1, 1)
        # maxpool - we will reuse -> self.maxpool_with_argmax

        # Third block, 128 -> 256
        self.conv5_128_to_256 = modules.Conv2d_BatchNorm_ReLU(128, 256, 3, 1, 1)
        self.conv6_256_to_256 = modules.Conv2d_BatchNorm_ReLU(256, 256, 3, 1, 1)
        self.conv7_256_to_256 = modules.Conv2d_BatchNorm_ReLU(256, 256, 3, 1, 1)
        # maxpool - we will reuse -> self.maxpool_with_argmax

        # Fourth block, 256 -> 512
        self.conv8_256_to_512 = modules.Conv2d_BatchNorm_ReLU(256, 512, 3, 1, 1)
        self.conv9_512_to_512 = modules.Conv2d_BatchNorm_ReLU(512, 512, 3, 1, 1)
        self.conv10_512_to_512 = modules.Conv2d_BatchNorm_ReLU(512, 512, 3, 1, 1)
        # maxpool - we will reuse -> self.maxpool_with_argmax

        # Fifth block, 512 -> 512
        self.conv11_512_to_512 = modules.Conv2d_BatchNorm_ReLU(512, 512, 3, 1, 1)
        self.conv12_512_to_512 = modules.Conv2d_BatchNorm_ReLU(512, 512, 3, 1, 1)
        self.conv13_512_to_512 = modules.Conv2d_BatchNorm_ReLU(512, 512, 3, 1, 1)
        # maxpool - we will reuse -> self.maxpool_with_argmax

        self.layers = [
            self.conv1_in_to_64,
            self.conv2_64_to_64,
            self.conv3_64_to_128,
            self.conv4_128_to_128,
            self.conv5_128_to_256,
            self.conv6_256_to_256,
            self.conv7_256_to_256,
            self.conv8_256_to_512,
            self.conv9_512_to_512,
            self.conv10_512_to_512,
            self.conv11_512_to_512,
            self.conv12_512_to_512,
            self.conv13_512_to_512
        ]

    def forward(self, x):

        # First block, in_channels -> 64
        x = self.conv1_in_to_64(x)
        x = self.conv2_64_to_64(x)
        x, indices1 = self.maxpool_with_argmax(x)

        # Second block, 64 -> 128
        x = self.conv3_64_to_128(x)
        x = self.conv4_128_to_128(x)
        x, indices2 = self.maxpool_with_argmax(x)

        # Third block, 128 -> 256
        x = self.conv5_128_to_256(x)
        x = self.conv6_256_to_256(x)
        x = self.conv7_256_to_256(x)
        x, indices3 = self.maxpool_with_argmax(x)

        # Fourth block, 256 -> 512
        x = self.conv8_256_to_512(x)
        x = self.conv9_512_to_512(x)
        x = self.conv10_512_to_512(x)
        x, indices4 = self.maxpool_with_argmax(x)

        # Fifth block, 512 -> 512
        x = self.conv11_512_to_512(x)
        x = self.conv12_512_to_512(x)
        x = self.conv13_512_to_512(x)
        x, indices5 = self.maxpool_with_argmax(x)

        return x, [indices5, indices4, indices3, indices2, indices1]


class SegNet_VGG16_decoder(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        # Fifth block, 512 -> 512
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv13_512_to_512 = modules.Conv2d_BatchNorm_ReLU(512, 512, 3, 1, 1)
        self.conv12_512_to_512 = modules.Conv2d_BatchNorm_ReLU(512, 512, 3, 1, 1)
        self.conv11_512_to_512 = modules.Conv2d_BatchNorm_ReLU(512, 512, 3, 1, 1)

        # Fourth block, 512 -> 256
        # reuse -> self.unpool
        self.conv10_512_to_512 = modules.Conv2d_BatchNorm_ReLU(512, 512, 3, 1, 1)
        self.conv9_512_to_512 = modules.Conv2d_BatchNorm_ReLU(512, 512, 3, 1, 1)
        self.conv8_512_to_256 = modules.Conv2d_BatchNorm_ReLU(512, 256, 3, 1, 1)

        # Third block, 256 -> 128
        # reuse -> self.unpool
        self.conv7_256_to_256 = modules.Conv2d_BatchNorm_ReLU(256, 256, 3, 1, 1)
        self.conv6_256_to_256 = modules.Conv2d_BatchNorm_ReLU(256, 256, 3, 1, 1)
        self.conv5_256_to_128 = modules.Conv2d_BatchNorm_ReLU(256, 128, 3, 1, 1)

        # Second block 128 -> 64
        # reuse -> self.unpool
        self.conv4_128_to_128 = modules.Conv2d_BatchNorm_ReLU(128, 128, 3, 1, 1)
        self.conv3_128_to_64 = modules.Conv2d_BatchNorm_ReLU(128, 64, 3, 1, 1)

        # First block 64 -> n_classes
        # reuse -> self.unpool
        self.conv2_64_to_64 = modules.Conv2d_BatchNorm_ReLU(64, 64, 3, 1, 1)
        # TODO should there be ReLU at the end of file?
        self.final = nn.Conv2d(64, n_classes, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x, indices):
        i5, i4, i3, i2, i1 = indices

        # Fifth block, 512 -> 512
        x = self.unpool(x, indices=i5)
        x = self.conv13_512_to_512(x)
        x = self.conv12_512_to_512(x)
        x = self.conv11_512_to_512(x)

        # Fourth block, 512 -> 256
        x = self.unpool(x, indices=i4)
        x = self.conv10_512_to_512(x)
        x = self.conv9_512_to_512(x)
        x = self.conv8_512_to_256(x)

        # Third block, 256 -> 128
        x = self.unpool(x, indices=i3)
        x = self.conv7_256_to_256(x)
        x = self.conv6_256_to_256(x)
        x = self.conv5_256_to_128(x)

        # Second block 128 -> 64
        x = self.unpool(x, indices=i2)
        x = self.conv4_128_to_128(x)
        x = self.conv3_128_to_64(x)

        # First block 64 -> n_classes
        x = self.unpool(x, indices=i1)
        x = self.conv2_64_to_64(x)
        x = self.final(x)

        return x

