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
from typing import Any

import torch
import torch.nn as nn
from ssrgan.models.utils import Conv
from ssrgan.models.utils import dw_conv
from torch.hub import load_state_dict_from_url

__all__ = ["InceptionA", "InceptionX",
           "Inception", "inception"]

model_urls = {
    "inception": ""
}


class InceptionA(nn.Module):
    r""" InceptionA implemented in inception version 4.

    `"Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning" <https://arxiv.org/abs/1602.07261>`_ paper
    """

    def __init__(self, in_channels: int = 64) -> None:
        r""" Modules introduced in InceptionV4 paper.

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
        """
        super(InceptionA, self).__init__()

        branch_features = int(in_channels // 4)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(branch_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        return out


class InceptionX(nn.Module):
    r""" It is improved by referring to the structure of the original paper.
    """

    def __init__(self, in_channels: int, out_channels: int, expand_factor: int = 0.25):
        r""" Modules introduced in InceptionX paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, float): Number of channels produced by the expand convolution. (Default: 0.25).
        """
        super(InceptionX, self).__init__()

        branch_features = int(in_channels * expand_factor)

        # Squeeze style layer.
        self.branch1_1 = Conv(in_channels, branch_features // 4, kernel_size=1, stride=1, padding=0)
        self.branch1_2 = Conv(branch_features // 4, branch_features // 2, kernel_size=1, stride=1, padding=0)
        self.branch1_3 = Conv(branch_features // 4, branch_features // 2, kernel_size=3, stride=1, padding=1)

        # InvertedResidual style layer
        self.branch2_1 = Conv(in_channels, branch_features * 2, kernel_size=1, stride=1, padding=0)
        self.branch2_2 = Conv(branch_features * 2, branch_features * 2, kernel_size=3, stride=1, padding=1)
        self.branch2_3 = Conv(branch_features * 2, branch_features, kernel_size=1, stride=1, padding=0, act=False)

        # Inception style layer 1
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(branch_features, branch_features, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0)
        )

        # Inception style layer 2
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(branch_features, branch_features, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0)
        )

        self.branch_concat = nn.Conv2d(branch_features * 2, branch_features * 2, kernel_size=1, stride=1, padding=0)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
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
    r""" It is mainly based on the Inception network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        super(Inception, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Twenty-three structures similar to InceptionX network.
        trunk = []
        for _ in range(23):
            trunk.append(InceptionX(64, 64))
        self.trunk = nn.Sequential(*trunk)

        self.inception = InceptionX(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                InceptionX(64, 64),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                InceptionX(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next conv layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # First conv layer.
        conv1 = self.conv1(input)

        # InceptionX trunk.
        trunk = self.trunk(conv1)
        # Concat conv1 and inceptionX trunk.
        out = torch.add(conv1, trunk)

        # InceptionX layer.
        inception = self.inception(out)
        # Concat conv1 and inceptionX layer.
        out = torch.add(conv1, inception)

        # Upsampling layers.
        out = self.upsampling(out)
        # Next conv layer.
        out = self.conv2(out)
        # Final output layer.
        out = self.conv3(out)

        return torch.tanh(out)


def inception(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> Inception:
    r"""Inception model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1602.07261>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = Inception(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["inception"], progress=progress)
        model.load_state_dict(state_dict)
    return model
