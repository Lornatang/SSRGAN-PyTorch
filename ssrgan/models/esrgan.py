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

__all__ = [
    "ResidualDenseBlock", "ResidualInResidualDenseBlock",
    "ESRGAN", "esrgan"
]

model_urls = {
    "esrgan": ""
}


class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, in_channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2) -> None:
        """
        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + 0 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat([input, conv1], dim=1))
        conv3 = self.conv3(torch.cat([input, conv1, conv2], dim=1))
        conv4 = self.conv4(torch.cat([input, conv1, conv2, conv3], dim=1))
        conv5 = self.conv5(torch.cat([input, conv1, conv2, conv3, conv4], dim=1))

        return conv5.mul(self.scale_ratio) + input


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, in_channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2) -> None:
        """
        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(input)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out.mul(self.scale_ratio) + input


class ESRGAN(nn.Module):
    r""" It is mainly based on the SRGAN network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of SRGAN network structure."""
        super(ESRGAN, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 2))

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Twenty-three structures similar to ESRGAN network.
        trunk = []
        for _ in range(23):
            trunk.append(ResidualInResidualDenseBlock(64, 32, 0.2))
        self.trunk = nn.Sequential(*trunk)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Upsampling layers.
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        self.upsampling = nn.Sequential(*upsampling)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Final output layer.
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        trunk = self.trunk(conv1)
        conv2 = self.conv2(trunk)
        out = torch.add(conv1, conv2)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return torch.tanh(out)


def esrgan(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ESRGAN:
    r"""ESRGAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1809.00219>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ESRGAN(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["esrgan"], progress=progress)
        model.load_state_dict(state_dict)
    return model
