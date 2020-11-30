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
from .utils import Conv

__all__ = [
    "Fire",
    "SqueezeNet", "squeezenet"
]

model_urls = {
    "squeezenet": ""
}


class Fire(nn.Module):
    r""" SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """

    def __init__(self, in_channels: int = 64, squeeze_channels: int = 8, expand1x1_channels: int = 32,
                 expand3x3_channels: int = 32):
        r""" Modules introduced in SqueezeNet paper.

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            squeeze_channels (int): Number of channels produced by the squeeze layer. (Default: 8).
            expand1x1_channels (int): Number of channels produced by the expand 1x1 layer. (Default: 32).
            expand3x3_channels (int): Number of channels produced by the expand 3x3 layer. (Default: 32).
        """
        super(Fire, self).__init__()

        # squeeze
        self.squeeze = nn.Sequential(
            Conv(in_channels, squeeze_channels, 1, 1, 0, dilation=1, groups=1, act=True),
            FReLU(squeeze_channels)
        )

        # expand 1x1
        self.expand1x1 = nn.Sequential(
            Conv(squeeze_channels, expand1x1_channels, 1, 1, 0, dilation=1, groups=1, act=True),
            FReLU(expand1x1_channels)

        )

        # expand 3x3
        self.expand3x3 = nn.Sequential(
            Conv(squeeze_channels, expand3x3_channels, 3, 1, 1, dilation=1, groups=1, act=True),
            FReLU(expand3x3_channels)

        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

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
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = Conv(3, 64, 3, 1, 1, dilation=1, groups=1, act=True)

        # Twenty-three structures similar to Fire network.
        trunk = []
        for _ in range(23):
            trunk.append(Fire(64, 64))
        self.trunk = nn.Sequential(*trunk)

        self.fire = Fire(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                Fire(64, 64),
                Conv(64, 256, 3, 1, 1, dilation=1, groups=1, act=True),
                nn.PixelShuffle(upscale_factor=2),
                Fire(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next conv layer
        self.conv2 = Conv(64, 64, 3, 1, 1, dilation=1, groups=1, act=True)

        # Final output layer
        self.conv3 = Conv(64, 3, 3, 1, 1, dilation=1, groups=1, act=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # First conv layer.
        conv1 = self.conv1(input)

        # Fire trunk.
        trunk = self.trunk(conv1)
        # Concat conv1 and fire trunk.
        out = torch.add(conv1, trunk)

        # Fire layer.
        fire = self.fire(out)
        # Concat conv1 and fire layer.
        out = torch.add(conv1, fire)

        # Upsampling layers.
        out = self.upsampling(out)
        # Next conv layer.
        out = self.conv2(out)
        # Final output layer.
        out = self.conv3(out)

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
