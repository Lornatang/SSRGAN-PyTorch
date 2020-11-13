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
from .utils import conv1x1
from .utils import conv3x3

__all__ = [
    "FReLU", "DepthwiseSeparableConvolution", "MobileNetV1", "mobilenetv1"
]

model_urls = {
    "mobilenetv1": ""
}


class DepthwiseSeparableConvolution(nn.Module):
    r""" Depthwise separable convolution implemented in mobilenet version 1.

    `"MobileNets: Efficient Convolutional Neural Networks for
    Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_ paper.

    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64) -> None:
        r""" Modules introduced in MobileNetV1 paper.
        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            out_channels (int): Number of channels produced by the convolution. (Default: 64).
        """
        super(DepthwiseSeparableConvolution, self).__init__()

        # dw
        self.depthwise = nn.Sequential(
            conv3x3(in_channels, in_channels, groups=in_channels),
            FReLU(in_channels)
        )

        # pw
        self.pointwise = nn.Sequential(
            conv1x1(in_channels, out_channels),
            FReLU(out_channels)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # DepthWise convolution
        out = self.depthwise(input)
        # Projection convolution
        out = self.pointwise(out)

        return out


class MobileNetV1(nn.Module):
    r""" It is mainly based on the MobileNetV1 network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of SRGAN network structure."""
        super(MobileNetV1, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 2))

        # First layer
        self.conv1 = conv3x3(3, 64)

        # Twenty-three structures similar to MobileNetV1 network.
        trunk = []
        for _ in range(23):
            trunk.append(DepthwiseSeparableConvolution(64, 64))
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


def mobilenetv1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV1:
    r"""MobileNetV1 model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1704.04861>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV1(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["mobilenetv1"], progress=progress)
        model.load_state_dict(state_dict)
    return model
