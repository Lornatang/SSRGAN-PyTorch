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
    "Fire", "SqueezeNet", "squeezenet"
]

model_urls = {
    "squeezenet": ""
}


class Fire(nn.Module):
    r""" SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """

    def __init__(self, in_channels: int = 64, squeeze_channels: int = 8, expand1x1_channels: int = 16,
                 expand3x3_channels: int = 16):
        r""" Modules introduced in SqueezeNet paper.

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            squeeze_channels (int): Number of channels produced by the squeeze layer. (Default: 8).
            expand1x1_channels (int): Number of channels produced by the expand 1x1 layer. (Default: 16).
            expand3x3_channels (int): Number of channels produced by the expand 3x3 layer. (Default: 16).
        """
        super(Fire, self).__init__()

        # squeeze
        self.squeeze = nn.Sequential(
            conv1x1(in_channels, squeeze_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # expand 1x1
        self.expand1x1 = nn.Sequential(
            conv1x1(squeeze_channels, expand1x1_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # expand 3x3
        self.expand3x3 = nn.Sequential(
            conv3x3(squeeze_channels, expand3x3_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Squeeze convolution
        out = self.squeeze(input)
        # Expand convolution
        out = torch.cat([self.expand1x1(out), self.expand3x3(out)], 1)

        return out


class SqueezeNet(nn.Module):
    r""" It is mainly based on the SqueezeNet network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of SqueezeNet network structure."""
        super(SqueezeNet, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 2))

        # First layer
        self.conv1 = conv3x3(3, 64)

        # Twenty-three structures similar to SqueezeNet network.
        trunk = []
        for _ in range(23):
            trunk.append(Fire(64))
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


def squeezenet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet:
    r"""SqueezeNet model architecture from the
    `"One weird trick..." <https://arxiv.org/pdf/1602.07360.pdf>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = SqueezeNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["squeezenet"], progress=progress)
        model.load_state_dict(state_dict)
    return model
