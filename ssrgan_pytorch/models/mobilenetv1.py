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

__all__ = ["DepthwiseSeparableConvolution", "MobileNetV1"]


class DepthwiseSeparableConvolution(nn.Module):
    r""" Depthwise separable convolution implemented in mobilenet version 1.

    `"MobileNets: Efficient Convolutional Neural Networks for
    Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_ paper

    """

    def __init__(self, in_channels, out_channels):
        r""" Modules introduced in MobileNetV1 paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """
        super(DepthwiseSeparableConvolution, self).__init__()

        # dw
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # pw
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

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
        # DepthWise convolution
        out = self.depthwise(input)
        # Projection convolution
        out = self.pointwise(out)

        return out


class MobileNetV1(nn.Module):
    r""" It is mainly based on the mobilenet-v1 network as the backbone network generator"""

    def __init__(self):
        r""" This is made up of mobilenet-v1 network structure.
        """
        super(MobileNetV1, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Eight structures similar to MobileNetV1 network.
        trunk = []
        for _ in range(8):
            trunk.append(DepthwiseSeparableConvolution(64, 64))
        self.Trunk = nn.Sequential(*trunk)

        self.mobilenet = DepthwiseSeparableConvolution(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(1):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                DepthwiseSeparableConvolution(64, 64),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2),
                DepthwiseSeparableConvolution(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next layer after upper sampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input: Tensor) -> Tensor:
        conv1 = self.conv1(input)
        trunk = self.Trunk(conv1)
        mobilenet = self.mobilenet(trunk)
        out = torch.add(conv1, mobilenet)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out
