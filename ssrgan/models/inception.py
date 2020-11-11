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

from ssrgan.activation import FReLU

__all__ = ["FReLU", "InceptionA", "InceptionX", "Inception"]


class InceptionA(nn.Module):
    r""" InceptionA implemented in inception version 4.

    `"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning" <https://arxiv.org/abs/1602.07261>`_ paper
    """

    def __init__(self, in_channels):
        r""" Modules introduced in InceptionV4 paper.

        Args:
            in_channels (int): Number of channels in the input image.
        """
        super(InceptionA, self).__init__()

        branch_features = int(in_channels // 4)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        return out


class InceptionX(nn.Module):
    r""" It is improved by referring to the structure of the original paper.
    """

    def __init__(self, in_channels, out_channels):
        r""" Modules introduced in InceptionX paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """
        super(InceptionX, self).__init__()

        branch_features = int(in_channels // 4)

        # Squeeze style layer
        self.branch1_1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features // 4, kernel_size=1, stride=1, padding=0, bias=False),
            FReLU(branch_features // 4)
        )
        self.branch1_2 = nn.Sequential(
            nn.Conv2d(branch_features // 4, branch_features // 2, kernel_size=1, stride=1, padding=0, bias=False),
            FReLU(branch_features // 2)
        )
        self.branch1_3 = nn.Sequential(
            nn.Conv2d(branch_features // 4, branch_features // 2, kernel_size=3, stride=1, padding=1, bias=False),
            FReLU(branch_features // 2)
        )

        # InvertedResidual style layer
        self.branch2_1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features * 2, kernel_size=1, stride=1, padding=0, bias=False),
            FReLU(branch_features * 2)
        )
        self.branch2_2 = nn.Sequential(
            nn.Conv2d(branch_features * 2, branch_features * 2, kernel_size=3, stride=1, padding=1,
                      groups=branch_features * 2, bias=False),
            FReLU(branch_features * 2)
        )
        self.branch2_3 = nn.Sequential(
            nn.Conv2d(branch_features * 2, branch_features, kernel_size=3, stride=1, padding=1, bias=False),
            FReLU(branch_features)
        )

        # Inception style layer 1
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            FReLU(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=(3, 1), stride=1, padding=(1, 0),
                      groups=branch_features, bias=False),
            FReLU(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            FReLU(branch_features)
        )

        # Inception style layer 2
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
            FReLU(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=(1, 3), stride=1, padding=(0, 1),
                      groups=branch_features, bias=False),
            FReLU(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            FReLU(branch_features)
        )

        self.branch_concat = nn.Conv2d(branch_features * 2, branch_features * 2, kernel_size=1, bias=False)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

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
        # Squeeze layer.
        branch1_1 = self.branch1_1(input)  # Squeeze convolution
        branch1_2 = self.branch1_2(branch1_1)  # Expand convolution 1x1
        branch1_3 = self.branch1_3(branch1_1)  # Expand convolution 3x3
        squeeze_out = torch.cat([branch1_2, branch1_3], dim=1)

        # InvertedResidual layer.
        branch2_1 = self.branch2_1(input)
        branch2_2 = self.branch2_2(branch2_1)
        mobile_out = self.branch2_3(branch2_2)

        # Concat Squeeze and InvertedResidual layer.
        branch_concat1_2 = torch.cat([squeeze_out, mobile_out], dim=1)
        branch_out1 = self.branch_concat(branch_concat1_2)

        # Inception layer
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)
        branch_concat3_4 = torch.cat([branch3, branch4], dim=1)
        branch_out2 = self.branch_concat(branch_concat3_4)

        # Concat layer
        out = torch.cat([branch_out1, branch_out2], dim=1)
        out = self.conv1x1(out)

        return out + input


class Inception(nn.Module):
    r""" It is mainly based on the mobilenet-v1 network as the backbone network generator"""

    def __init__(self):
        r""" This is made up of Inception network structure.
        """
        super(Inception, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Eight structures similar to InceptionX network.
        trunk = []
        for _ in range(8):
            trunk.append(InceptionX(64, 64))
        self.trunk = nn.Sequential(*trunk)

        self.inception = InceptionX(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(1):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                InceptionX(64, 64),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2),
                InceptionX(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

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
        inception = self.inception(trunk)
        out = torch.add(conv1, inception)
        out = self.upsampling(out)
        out = self.conv2(out)
        out = self.conv3(out)

        return out
