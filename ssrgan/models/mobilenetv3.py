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
from torch.hub import load_state_dict_from_url

from ssrgan.activation import FReLU
from ssrgan.activation import HSigmoid
from ssrgan.activation import HSwish
from .utils import conv1x1
from .utils import conv3x3
from .utils import conv5x5

__all__ = [
    "SEModule", "MobileNetV3Bottleneck", "MobileNetV3", "mobilenetv3"
]

model_urls = {
    "mobilenetv3": ""
}


class SEModule(nn.Module):
    r""" Squeeze-and-Excite module.

    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile" <https://arxiv.org/pdf/1807.11626.pdf>`_

    """

    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        r""" Modules introduced in MnasNet paper.
        Args:
            in_channels (int): Number of channels in the input image.
            reduction (optional, int): Reduce the number of channels by several times. (Default: 4).
        """
        super(SEModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            HSigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = input.size()
        out = self.avgpool(input).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        return input * out.expand_as(input)


class MobileNetV3Bottleneck(nn.Module):
    r""" Base on MobileNetV2 + Squeeze-and-Excite.

    `"Searching for MobileNetV3" <https://arxiv.org/pdf/1905.02244.pdf>`_ paper.

    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, expand_factor=6):
        r""" Modules introduced in MobileNetV3 paper.

        Args:
            in_channels (optional, int): Number of channels in the input image. (Default: 64).
            out_channels (optional, int): Number of channels produced by the convolution. (Default: 64).
            expand_factor (optional, int): Number of channels produced by the expand convolution. (Default: 6).
        """
        super(MobileNetV3Bottleneck, self).__init__()
        hidden_channels = int(round(in_channels * expand_factor))

        self.shortcut = nn.Sequential(
            conv1x1(in_channels, out_channels),
            FReLU(out_channels)
        )

        # pw
        self.pointwise = nn.Sequential(
            conv1x1(in_channels, hidden_channels),
            FReLU(hidden_channels)
        )

        # dw
        self.depthwise = conv5x5(hidden_channels, hidden_channels)

        # squeeze and excitation module.
        self.SEModule = nn.Sequential(
            SEModule(hidden_channels),
            HSwish()
        )

        # pw-linear
        self.pointwise_linear = conv1x1(hidden_channels, out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Expansion convolution
        out = self.pointwise(input)
        # DepthWise convolution
        out = self.depthwise(out)
        # Squeeze-and-Excite
        out = self.SEModule(out)
        # Projection convolution
        out = self.pointwise_linear(out)

        return out + self.shortcut(input)


class MobileNetV3(nn.Module):
    r""" It is mainly based on the MobileNetV3 network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of SRGAN network structure."""
        super(MobileNetV3, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 2))

        # First layer
        self.conv1 = conv3x3(3, 64)

        # Twenty-three structures similar to MobileNetV3Bottleneck network.
        trunk = []
        for _ in range(23):
            trunk.append(MobileNetV3Bottleneck(64, 64))
        self.trunk = nn.Sequential(*trunk)

        self.conv2 = conv3x3(64, 64, groups=1)

        # Upsampling layers.
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                conv3x3(64, 256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        self.upsampling = nn.Sequential(*upsampling)

        self.conv3 = conv3x3(64, 64, groups=1)

        # Final output layer.
        self.conv4 = conv3x3(64, 3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        trunk = self.trunk(conv1)
        conv2 = self.conv2(trunk)
        out = torch.add(conv1, conv2)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return torch.tanh(out)


def mobilenetv3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    r"""MobileNetV3 model architecture from the
    `"One weird trick..." <https://arxiv.org/pdf/1905.02244.pdf>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["mobilenetv3"], progress=progress)
        model.load_state_dict(state_dict)
    return model
