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

import numpy as np
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [
    "get_upsample_filter", "ConvBlock", "LapSRN", "lapsrn"
]

model_urls = {
    "lapsrn": ""
}


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class ConvBlock(nn.Module):
    r"""Main residual block structure"""

    def __init__(self, in_channels: int = 64, out_channels: int = 64) -> None:
        r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """
        super(ConvBlock, self).__init__()
        block = []
        for _ in range(10):
            block += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]
        block += [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.block = nn.Sequential(*block)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.main(input)

        return out


class LapSRN(nn.Module):
    r""" It is mainly based on the LapSRN network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of LapSRN network structure.
                """
        super(LapSRN, self).__init__()
        self.upscale_factor = upscale_factor

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Extract the first layer features.
        self.conv2_1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = ConvBlock(64, 64)

        # Extract the second layer features.
        self.conv3_1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3_2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = ConvBlock(64, 64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                n, c, h, w = m.weight.data.image_size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(n, c, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        out = self.conv1(input)

        conv2_1 = self.conv2_1(out)
        conv2_2 = self.conv2_2(input)
        conv2_3 = self.conv2_3(conv2_1)
        conv2 = conv2_2 + conv2_3

        conv3_1 = self.conv3_1(conv2_1)
        conv3_2 = self.conv3_2(conv2)
        conv3_3 = self.conv3_3(conv3_1)
        conv3 = conv3_2 + conv3_3

        if self.upscale_factor == 4:
            return conv3
        else:
            return conv2


def lapsrn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> LapSRN:
    r"""LapSRN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1710.01992>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = LapSRN(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["lapsrn"], progress=progress)
        model.load_state_dict(state_dict)
    return model
