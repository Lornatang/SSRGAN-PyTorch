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
import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["InvertedResidual", "MobileNetV2"]


class InvertedResidual(nn.Module):
    r""" Improved convolution method based on MobileNet-v2 version.

    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_

    """

    def __init__(self, in_channels, out_channels, expand_factor=1):
        r""" Modules introduced in MobileNetV2 paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, int): Number of channels produced by the expand convolution. (Default: 1).
        """
        super(InvertedResidual, self).__init__()
        hidden_channels = int(round(in_channels * expand_factor))

        # pw
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # dw
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # pw-linear
        self.pointwise_linear = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input: Tensor) -> Tensor:
        # Expansion convolution
        out = self.pointwise(input)
        # DepthWise convolution
        out = self.depthwise(out)
        # Projection convolution
        out = self.pointwise_linear(out)

        return out + input


class MobileNetV2(nn.Module):
    r""" It is mainly based on the mobilenet-v2 network as the backbone network generator"""

    def __init__(self):
        r""" This is made up of mobilenet-v2 network structure.
        """
        super(MobileNetV2, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # Eight structures similar to MobileNet network.
        self.trunk = nn.Sequential(
            InvertedResidual(64, 64),
            InvertedResidual(64, 64),
            InvertedResidual(64, 64),
            InvertedResidual(64, 64),
            InvertedResidual(64, 64),
            InvertedResidual(64, 64),
            InvertedResidual(64, 64),
            InvertedResidual(64, 64)
        )

        self.mobilenet = nn.Sequential(
            InvertedResidual(64, 64)
        )

        # Upsampling layers
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            InvertedResidual(64, 64),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            InvertedResidual(64, 64)
        )

        # Next layer after upper sampling
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input: Tensor) -> Tensor:
        conv1 = self.conv1(input)
        trunk = self.trunk(conv1)
        mobilenet = self.mobilenet(trunk)
        out = torch.add(conv1, mobilenet)
        out = self.upsampling(out)
        out = self.conv2(out)
        out = self.conv3(out)

        return out
