# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .activation import HSigmoid
from .activation import HSwish

__all__ = [
    "DepthWiseConv", "DiscriminatorForVGG", "GeneratorForMobileNet",
    "InvertedResidual", "InvertedResidualSEModule", "SEModule", "channel_shuffle"
]


class DepthWiseConv(nn.Module):
    r"""Deep separable convolution implemented in mobilenet version 1.

    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_

    """

    def __init__(self, channels):
        r""" This is a structure for simple versions.
        """
        super(DepthWiseConv, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, input: Tensor) -> Tensor:
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        return out + input


class DiscriminatorForVGG(nn.Module):
    r"""The main architecture of the discriminator. Similar to VGG structure."""

    def __init__(self):
        super(DiscriminatorForVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # input is 3 x 216 x 216
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # state size. (64) x 108 x 108
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # state size. 128 x 54 x 54
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),  # state size. 256 x 27 x 27
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # state size. 512 x 14 x 14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # state size. 512 x 7 x 7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)

        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        )

    def _initialize_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.zeros_(m.bias)

    def forward(self, input: Tensor) -> Tensor:
        out = self.features(input)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return torch.sigmoid(out)


class GeneratorForMobileNet(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""
    __constants__ = ["upscale_factor", "block"]

    def __init__(self, upscale_factor, block="v1", num_depth_wise_conv_block=16):
        r""" This is an ssrgan model defined by the author himself.

        Args:
            upscale_factor (int): Image magnification factor. (Default: 4).
            block (str): Select the structure used by the backbone network. (Default: ``v1``).
            num_depth_wise_conv_block (int): How many depth wise conv block are combined. (Default: 16).
            init_weights (optional, bool): Whether to initialize the initial neural network. (Default: ``True``).
        """
        super(GeneratorForMobileNet, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 2))

        if block == "v1":  # For MobileNet v1
            block = DepthWiseConv
        elif block == "v2":  # For MobileNet v2
            block = InvertedResidual
        elif block == "v3":  # For MobileNet v3
            block = InvertedResidualSEModule
        else:
            raise NameError("Please check the block name, the block name must be `v1`, `v2` or `v3`.")

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, bias=False),
            nn.PReLU()
        )

        # 16 layer similar stack block structure.
        blocks = []
        for _ in range(num_depth_wise_conv_block):
            blocks.append(block(64))
        self.Trunk = nn.Sequential(*blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        # Upsampling layers
        upsampling = []
        for out_features in range(num_upsample_block):
            upsampling += [
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        out1 = self.conv1(input)

        # Similar to the structure block of MobileNet, the overall structure is similar to SRGAN.
        out = self.Trunk(out1)
        # The stacking block is followed by a convolution layer.
        out2 = self.conv2(out)
        # Fusing features before stacking blocks and features after stacking blocks convolution.
        out = out1 + out2

        out = self.upsampling(out)
        out = self.conv3(out)

        return out


class InvertedResidualSEModule(nn.Module):
    r""" Base on MobileNetV2 + Squeeze-and-Excite.

    `"Searching for MobileNetV3" <https://arxiv.org/pdf/1905.02244.pdf>`_

    """

    def __init__(self, channels):
        r""" This is a structure for simple versions.
        """
        super(InvertedResidualSEModule, self).__init__()
        # pw
        self.conv1 = nn.Conv2d(channels, channels * 6, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels * 6)
        self.hswish1 = HSwish()
        # dw
        self.conv2 = nn.Conv2d(channels * 6, channels * 6, kernel_size=5, stride=1, padding=2, groups=channels * 6,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels * 6)
        # SE-Module
        self.SEModule = SEModule(channels * 6)
        self.hswish2 = HSwish()
        # pw-linear
        self.conv3 = nn.Conv2d(channels * 6, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def _initialize_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.zeros_(m.bias)

    def forward(self, input: Tensor) -> Tensor:
        # Expansion convolution
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.hswish1(out)
        # DepthWise convolution
        out = self.conv2(out)
        out = self.bn2(out)
        # Squeeze-and-Excite
        out = self.SEModule(out)
        out = self.hswish2(out)
        # Projection convolution
        out = self.conv3(out)
        out = self.bn3(out)

        return out + input


class InvertedResidual(nn.Module):
    r""" Improved convolution method based on MobileNet-v1 version.

    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_

    """

    def __init__(self, channels, init_weights=True):
        r""" This is a structure for simple versions.

        Args:
            init_weights (optional, bool): Whether to initialize the initial neural network. (Default: ``True``).
        """
        super(InvertedResidual, self).__init__()
        # pw
        self.conv1 = nn.Conv2d(channels, channels * 6, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels * 6)
        self.relu1 = nn.ReLU6(inplace=True)
        # dw
        self.conv2 = nn.Conv2d(channels * 6, channels * 6, kernel_size=3, stride=1, padding=1, groups=channels * 6,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels * 6)
        self.relu2 = nn.ReLU6(inplace=True)
        # pw-linear
        self.conv3 = nn.Conv2d(channels * 6, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                torch.nn.init.normal_(m.weight, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                torch.nn.init.normal_(m.weight, 1.0, 0.02)
                torch.nn.init.zeros_(m.bias)

    def forward(self, input: Tensor) -> Tensor:
        # Expansion convolution
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu1(out)
        # DepthWise convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # Projection convolution
        out = self.conv3(out)
        out = self.bn3(out)

        return out + input


class ShuffleNetV1(nn.Module):
    pass


class SEModule(nn.Module):
    r""" Base on MobileNetV2 + Squeeze-and-Excite.

    `"Searching for MobileNetV3" <https://arxiv.org/pdf/1905.02244.pdf>`_

    """

    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            HSigmoid()
        )

    def forward(self, input: Tensor) -> Tensor:
        b, c, _, _ = input.size()
        out = self.avgpool(input).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        return input * out.expand_as(input)


# Source from `https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py`
def channel_shuffle(input: Tensor, groups: int) -> Tensor:
    """
    Example:
        >>> d = np.array([0,1,2,3,4,5,6,7,8])
        >>> x = np.reshape(d, (3,3))
        >>> x = np.transpose(x, [1,0])
        >>> x = np.reshape(x, (9,))
        '[0 1 2 3 4 5 6 7 8] --> [0 3 6 1 4 7 2 5 8]'
    """
    batch_size, num_channels, height, width = input.data.size()
    channels_per_group = num_channels // groups

    # reshape
    input = input.view(batch_size, groups, channels_per_group, height, width)

    input = torch.transpose(input, 1, 2).contiguous()

    # flatten
    input = input.view(batch_size, -1, height, width)

    return input
