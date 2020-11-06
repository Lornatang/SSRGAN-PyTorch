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
import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["SymmetricBlock", "UNet"]


class SymmetricBlock(nn.Module):
    r""" U-shaped network.

    `"U-Net: Convolutional Networks for Biomedical
    Image Segmentation" <https://arxiv.org/abs/1505.04597>`_ paper

    """

    def __init__(self, in_channels=64, out_channels=64, ):
        r""" Modules introduced in U-Net paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """
        super(SymmetricBlock, self).__init__()

        # Down sampling
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # Residual block1
        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1, groups=in_channels // 2,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Up sampling
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Residual block1
        self.body2 = nn.Sequential(
            nn.Conv2d(out_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1, groups=in_channels // 2,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
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

    def forward(self, input: Tensor) -> Tensor:
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

    def __init__(self):
        r""" This is made up of u-net network structure.
        """
        super(UNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # Eight structures similar to U-Net network.
        self.trunk = nn.Sequential(
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64)
        )

        self.unet = nn.Sequential(
            SymmetricBlock(64, 64)
        )

        # Upsampling layers
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            SymmetricBlock(64, 64),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            SymmetricBlock(64, 64)
        )

        # Next layer after upper sampling
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input: Tensor) -> Tensor:
        conv1 = self.conv1(input)
        trunk = self.trunk(conv1)
        unet = self.unet(trunk)
        out = torch.add(conv1, unet)
        out = self.upsampling(out)
        out = self.conv2(out)
        out = self.conv3(out)

        return out
