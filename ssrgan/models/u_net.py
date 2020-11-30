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

from ssrgan.models.utils import Conv

__all__ = ["SymmetricBlock", "UNet", "unet"]

model_urls = {
    "unet": ""
}


class SymmetricBlock(nn.Module):
    r""" U-shaped network.

    `"U-Net: Convolutional Networks for Biomedical
    Image Segmentation" <https://arxiv.org/abs/1505.04597>`_ paper

    """

    def __init__(self, in_channels: int, out_channels: int, expand_factor=0.5) -> None:
        r""" Modules introduced in U-Net paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, float): Number of channels produced by the expand convolution. (Default: 0.5).
        """
        super(SymmetricBlock, self).__init__()
        hidden_channels = int(out_channels * expand_factor)

        # Down sampling.
        self.down = nn.Sequential(
            Conv(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1),

            # Residual block.
            Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            Conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        # Up sampling.
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channels, hidden_channels, kernel_size=2, stride=2),

            # Residual block.
            Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, act=False),
            Conv(hidden_channels, in_channels, kernel_size=3, stride=1, padding=1, act=False),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Down sampling.
        out = self.down(input)
        # Up sampling.
        out = self.up(out)

        return out + input


class UNet(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        super(UNet, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Twenty-three structures similar to SymmetricBlock network.
        trunk = []
        for _ in range(23):
            trunk.append(SymmetricBlock(64, 64))
        self.trunk = nn.Sequential(*trunk)

        self.unet = SymmetricBlock(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                SymmetricBlock(64, 64),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                SymmetricBlock(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next conv layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # First conv layer.
        conv1 = self.conv1(input)

        # U-Net trunk.
        trunk = self.trunk(conv1)
        # Concat conv1 and unet trunk.
        out = torch.add(conv1, trunk)

        # SymmetricBlock layer.
        unet = self.unet(out)
        # Concat conv1 and unet layer.
        out = torch.add(conv1, unet)

        # Upsampling layers.
        out = self.upsampling(out)
        # Next conv layer.
        out = self.conv2(out)
        # Final output layer.
        out = self.conv3(out)

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
