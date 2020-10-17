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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "Discriminator", "DepthWiseConv", "Generator",
]


class Discriminator(nn.Module):
    r"""The main architecture of the discriminator. Similar to VGG structure."""

    def __init__(self, init_weights=True):
        """
        Args:
            init_weights (optional, bool): Whether to initialize the initial neural network. (Default: ``True``).
        """
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )

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

    def forward(self, input: Tensor = None) -> Tensor:
        out = self.features(input)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return F.sigmoid(out)


class DepthWiseConv(nn.Module):
    r"""Deep separable convolution implemented in mobilenet version 1.
    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_
    """

    def __init__(self, channels):
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


class Generator(nn.Module):
    r"""The main architecture of the generator."""

    def __init__(self, upscale_factor, num_depth_wise_conv_block=8, init_weights=True):
        r""" This is an ssrgan model defined by the author himself.

        Args:
            upscale_factor (int): Image magnification factor. (Default: 4).
            num_depth_wise_conv_block (int): How many depth wise conv block are combined. (Default: 8).
            init_weights (optional, bool): Whether to initialize the initial neural network. (Default: ``True``).
        """
        num_upsample_block = int(math.log(upscale_factor, 2))

        super(Generator, self).__init__()
        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # 8 Depth Wise Convolution blocks
        depth_wise_conv_blocks = []
        for _ in range(num_depth_wise_conv_block):
            depth_wise_conv_blocks.append(DepthWiseConv(64))
        self.depth_wise_conv_blocks = nn.Sequential(*depth_wise_conv_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling layers
        upsampling = []
        for out_features in range(num_upsample_block):
            upsampling += [
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

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

    def forward(self, input: Tensor = None) -> Tensor:
        out1 = self.conv1(input)

        out = self.depth_wise_conv_blocks(out1)
        out2 = self.conv2(out)

        out = out1 + out2

        out = self.upsampling(out)
        out = self.conv3(out)

        return out
