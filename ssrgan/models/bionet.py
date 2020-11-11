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

from ssrgan.activation import FReLU
from ssrgan.models.inception import InceptionX
from ssrgan.models.mobilenetv1 import DepthwiseSeparableConvolution
from ssrgan.models.u_net import SymmetricBlock

__all__ = ["FReLU", "InceptionX", "DepthwiseSeparableConvolution", "SymmetricBlock", "BioNet"]


class BioNet(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""

    def __init__(self):
        super(BioNet, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.trunk_a = nn.Sequential(
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64)
        )
        self.trunk_b = nn.Sequential(
            DepthwiseSeparableConvolution(64, 64),
            DepthwiseSeparableConvolution(64, 64),
            DepthwiseSeparableConvolution(64, 64),
            DepthwiseSeparableConvolution(64, 64)
        )
        self.trunk_c = nn.Sequential(
            InceptionX(64, 64),
            InceptionX(64, 64)
        )

        self.bionet = InceptionX(64, 64)

        # Upsampling layers
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            InceptionX(64, 64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            FReLU(256),
            nn.PixelShuffle(upscale_factor=2),
            InceptionX(64, 64)
        )

        # Next conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            FReLU(64)
        )

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        # First conv layer.
        conv1 = self.conv1(input)

        # U-Net trunk.
        trunk_a = self.trunk_a(conv1)
        # Concat conv1 and trunk a.
        out1 = torch.add(conv1, trunk_a)

        # MobileNet trunk.
        trunk_b = self.trunk_b(out1)
        # Concat conv1 and trunk b.
        out2 = torch.add(conv1, trunk_b)

        # InceptionX trunk.
        trunk_c = self.trunk_c(out2)
        # Concat conv1 and trunk-c.
        out = torch.add(conv1, trunk_c)

        out = self.bionet(out)
        # Upsampling layers.
        out = self.upsampling(out)
        # Next conv layer.
        out = self.conv2(out)
        # Final output layer.
        out = self.conv3(out)

        return torch.tanh(out)
