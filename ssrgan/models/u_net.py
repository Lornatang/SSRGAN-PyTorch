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

__all__ = ["SymmetricBlock", "UNet", "unet"]

model_urls = {
    "unet": ""
}


class SymmetricBlock(nn.Module):
    r""" U-shaped network.

    `"U-Net: Convolutional Networks for Biomedical
    Image Segmentation" <https://arxiv.org/abs/1505.04597>`_ paper

    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64) -> None:
        r""" Modules introduced in U-Net paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """
        super(SymmetricBlock, self).__init__()

        # Down sampling
        self.down = nn.Sequential(
            conv3x3(in_channels, in_channels, stride=2),
            FReLU(in_channels),
            conv1x1(in_channels, in_channels),
            FReLU(in_channels)
        )

        # Residual block1
        self.body1 = nn.Sequential(
            conv3x3(in_channels, in_channels),
            FReLU(in_channels),
            conv1x1(in_channels, in_channels // 2),
            FReLU(in_channels // 2),

            conv3x3(in_channels // 2, in_channels // 2),
            FReLU(in_channels // 2),
            conv1x1(in_channels // 2, out_channels),
            FReLU(out_channels)
        )

        # Up sampling
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Residual block1
        self.body2 = nn.Sequential(
            conv3x3(out_channels, out_channels),
            FReLU(out_channels),
            conv1x1(out_channels, in_channels // 2),
            FReLU(in_channels // 2),

            conv3x3(in_channels // 2, in_channels // 2),
            FReLU(in_channels // 2),
            conv1x1(in_channels // 2, in_channels),
            FReLU(in_channels)
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
        # Down sampling
        out = self.down(input)
        # Down body
        out = self.body1(out)
        # Up sampling
        out = self.up(out)
        # Up body
        out = self.body2(out)

        return out + input


class UNet(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        super(UNet, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = conv3x3(3, 64)

        # Twenty-three structures similar to SymmetricBlock network.
        trunk = []
        for _ in range(23):
            trunk.append(SymmetricBlock(64, 64))
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


def unet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> UNet:
    r"""UNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1505.04597>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = UNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["unet"], progress=progress)
        model.load_state_dict(state_dict)
    return model
