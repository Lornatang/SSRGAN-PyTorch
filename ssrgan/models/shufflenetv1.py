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
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from ssrgan.activation import FReLU
from .utils import channel_shuffle

__all__ = [
    "BottleNeck",
    "ShuffleNetV1", "shufflenetv1"
]

model_urls = {
    "shufflenetv1": ""
}


class BottleNeck(nn.Module):
    r""" Depthwise separable convolution implemented in shufflenet version 1.
    
    `"ShuffleNet: An Extremely Efficient Convolutional Neural Network
    for Mobile Devices" <https://arxiv.org/pdf/1707.01083.pdf>`_ paper.

    """

    def __init__(self, in_channels: int, out_channels: int, expand_factor=0.25) -> None:
        r""" Modules introduced in ShuffleNetV1 paper.
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, float): Number of channels produced by the expand convolution. (Default: 0.25).
        """
        super(BottleNeck, self).__init__()

        channels = int(in_channels * expand_factor)

        # pw
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, groups=4),
            nn.BatchNorm2d(channels),
            FReLU(channels)
        )

        # dw
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            FReLU(channels)
        )

        # pw-linear
        self.pointwise_linear = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, groups=4),
            nn.BatchNorm2d(out_channels)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Expansion convolution
        out = self.pointwise(input)
        # Channel shuffle
        out = channel_shuffle(out, 4)
        # DepthWise convolution
        out = self.depthwise(out)
        # Projection convolution
        out = self.pointwise_linear(out)
        # Fusion input and out
        out = torch.add(input, out)
        out = F.relu(out, inplace=True)

        return out


class ShuffleNetV1(nn.Module):
    r""" It is mainly based on the ShuffleNetV1 network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of SRGAN network structure."""
        super(ShuffleNetV1, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Twenty-three structures similar to BottleNeck network.
        trunk = []
        for _ in range(23):
            trunk.append(BottleNeck(64, 64))
        self.trunk = nn.Sequential(*trunk)

        self.bottleneck = BottleNeck(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                BottleNeck(64, 64),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                BottleNeck(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next conv layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # First conv layer.
        conv1 = self.conv1(input)

        # Bottleneck trunk.
        trunk = self.trunk(conv1)
        # Concat conv1 and bottleneck trunk.
        out = torch.add(conv1, trunk)

        # Bottleneck layer.
        bottleneck = self.bottleneck(out)
        # Concat conv1 and bottleneck layer.
        out = torch.add(conv1, bottleneck)

        # Upsampling layers.
        out = self.upsampling(out)
        # Next conv layer.
        out = self.conv2(out)
        # Final output layer.
        out = self.conv3(out)

        return torch.tanh(out)


def shufflenetv1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV1:
    r"""MobileNetV1 model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1707.01083>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ShuffleNetV1(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["shufflenetv1"], progress=progress)
        model.load_state_dict(state_dict)
    return model
