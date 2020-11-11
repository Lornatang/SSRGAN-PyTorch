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

from .utils import conv3x3

__all__ = ["ResidualBlock", "SRGAN", "srgan"]

model_urls = {
    "srgan": ""
}


class ResidualBlock(nn.Module):
    r"""Main residual block structure"""

    def __init__(self, in_channels: int = 64) -> None:
        r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.

        Args:
            in_channels (int): Number of channels in the input image.
        """
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            conv3x3(in_channels, in_channels, groups=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            conv3x3(in_channels, in_channels, groups=1),
            nn.BatchNorm2d(in_channels)
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.main(input)

        return out + input


class SRGAN(nn.Module):
    r""" It is mainly based on the SRGAN network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of SRGAN network structure.
                """
        super(SRGAN, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 2))

        # First layer
        self.conv1 = nn.Sequential(
            conv3x3(3, 64),
            nn.PReLU()
        )

        # Sixteen structures similar to SRGAN network.
        trunk = []
        for _ in range(16):
            trunk.append(ResidualBlock(64))
        self.trunk = nn.Sequential(*trunk)

        self.conv2 = nn.Sequential(
            conv3x3(64, 64, groups=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                conv3x3(64, 256, groups=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        trunk = self.trunk(conv1)
        conv2 = self.conv2(trunk)
        out = torch.add(conv1, conv2)
        out = self.upsampling(out)
        out = self.conv3(out)

        return torch.tanh(out)


def srgan(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SRGAN:
    r"""SRGAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1505.04597>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = SRGAN(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["srgan"], progress=progress)
        model.load_state_dict(state_dict)
    return model
