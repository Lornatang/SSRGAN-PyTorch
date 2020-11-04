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

from ssrgan_pytorch.models.inception import InceptionA
from ssrgan_pytorch.models.mobilenetv1 import DepthwiseSeparableConvolution
from ssrgan_pytorch.models.u_net import SymmetricBlock

__all__ = ["InceptionA", "DepthwiseSeparableConvolution", "SymmetricBlock", "BioNet"]


class BioNet(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""

    def __init__(self):
        super(BioNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # Eight structures similar to U-Net network.
        self.trunk_A = nn.Sequential(
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64)
        )
        self.trunk_B = nn.Sequential(
            DepthwiseSeparableConvolution(64, 64),
            DepthwiseSeparableConvolution(64, 64)
        )
        self.trunk_C = nn.Sequential(
            InceptionA(64, 64),
            InceptionA(64, 64)
        )
        self.trunk_D = nn.Sequential(
            DepthwiseSeparableConvolution(64, 64),
            DepthwiseSeparableConvolution(64, 64)
        )

        self.inception = nn.Sequential(
            InceptionA(64, 64)
        )

        # Upsampling layers
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            InceptionA(64, 64),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            InceptionA(64, 64)
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
        trunk_a = self.trunk_A(conv1)
        out = torch.add(conv1, trunk_a)
        trunk_b = self.trunk_B(out)
        out = torch.add(conv1, trunk_b)
        trunk_c = self.trunk_C(out)
        out = torch.add(conv1, trunk_c)
        trunk_d = self.trunk_D(out)
        out = torch.add(conv1, trunk_d)
        out = self.upsampling(out)
        out = self.conv2(out)
        out = self.conv3(out)

        return out
