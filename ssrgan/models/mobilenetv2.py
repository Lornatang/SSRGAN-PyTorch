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

from .utils import conv1x1
from .utils import conv3x3

__all__ = [
    "InvertedResidual", "MobileNetV2", "mobilenetv2"
]

model_urls = {
    "mobilenetv2": ""
}


class InvertedResidual(nn.Module):
    r""" Improved convolution method based on MobileNet-v2 version.

    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_ paper.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, expand_factor: int = 6) -> None:
        r""" Modules introduced in MobileNetV2 paper.
        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            out_channels (int): Number of channels produced by the convolution. (Default: 64).
            expand_factor (optional, int): Number of channels produced by the expand convolution. (Default: 6).
        """
        super(InvertedResidual, self).__init__()
        hidden_channels = int(round(in_channels * expand_factor))

        # pw
        self.pointwise = nn.Sequential(
            conv1x1(in_channels, hidden_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # dw
        self.depthwise = nn.Sequential(
            conv3x3(hidden_channels, hidden_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # pw-linear
        self.pointwise_linear = conv1x1(hidden_channels, out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Expansion convolution
        out = self.pointwise(input)
        # DepthWise convolution
        out = self.depthwise(out)
        # Projection convolution
        out = self.pointwise_linear(out)

        return out + input


class MobileNetV2(nn.Module):
    r""" It is mainly based on the MobileNetV1 network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of SRGAN network structure."""
        super(MobileNetV2, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 2))

        # First layer
        self.conv1 = conv3x3(3, 64)

        # Eight structures similar to MobileNetV1 network.
        trunk = []
        for _ in range(8):
            trunk.append(InvertedResidual(64, 64))
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


def mobilenetv2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    r"""MobileNetV2 model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1801.04381>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["mobilenetv2"], progress=progress)
        model.load_state_dict(state_dict)
    return model
