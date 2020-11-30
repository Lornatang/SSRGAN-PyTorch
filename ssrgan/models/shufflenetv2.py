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
from .utils import channel_shuffle

__all__ = [
    "FReLU", "BottleNeck", "ShuffleNetV2", "shufflenetv2"
]

model_urls = {
    "shufflenetv2": ""
}


class BottleNeck(nn.Module):
    r""" Depthwise separable convolution implemented in shufflenet version 2.

    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    " <https://arxiv.org/pdf/1807.11164.pdf>`_ paper.

    """

    def __init__(self, channels: int, expand_factor=0.5) -> None:
        r""" Modules introduced in ShuffleNetV2 paper.
        Args:
            channels (int): Number of channels in the input image.
            expand_factor (optional, float): Number of channels produced by the expand convolution. (Default: 0.5).
        """
        super(BottleNeck, self).__init__()

        branch_features = int(channels * expand_factor)

        self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=1, padding=1, groups=branch_features,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x1, x2 = input.chunk(2, dim=1)
        out = torch.cat((x1, self.branch2(x2)), dim=1)
        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    r""" It is mainly based on the ShuffleNetV2 network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of SRGAN network structure."""
        super(ShuffleNetV2, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Twenty-three structures similar to BottleNeck network.
        trunk = []
        for _ in range(23):
            trunk.append(BottleNeck(64))
        self.trunk = nn.Sequential(*trunk)

        self.bottleneck = BottleNeck(64)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                BottleNeck(64),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                BottleNeck(64)
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


def shufflenetv2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ShuffleNetV2:
    r"""MobileNetV2 model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1807.11164>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ShuffleNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["shufflenetv2"], progress=progress)
        model.load_state_dict(state_dict)
    return model
