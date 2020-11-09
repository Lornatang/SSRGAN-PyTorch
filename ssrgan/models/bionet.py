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

from ssrgan.models.inception import InceptionX
from ssrgan.models.mobilenetv1 import DepthwiseSeparableConvolution
from ssrgan.models.u_net import SymmetricBlock

__all__ = ["InceptionX", "DepthwiseSeparableConvolution", "SymmetricBlock", "BioNet"]


class BioNet(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""

    def __init__(self):
        super(BioNet, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Eight structures similar to BioNet network.
        self.trunk_A = nn.Sequential(
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64),
            SymmetricBlock(64, 64)
        )
        self.trunk_B = nn.Sequential(
            InceptionX(64, 64),
            InceptionX(64, 64),
            InceptionX(64, 64),
            InceptionX(64, 64)
        )

        self.bionet = InceptionX(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(2):
            upsampling += [
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
                nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.PixelShuffle(upscale_factor=2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv2 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        # First conv layer.
        conv1 = self.conv1(input)

        # Four U-Net block.
        trunk_a = self.trunk_A(conv1)
        out = torch.add(conv1, trunk_a)
        # Four InceptionA block.
        trunk_b = self.trunk_B(out)
        out = torch.add(conv1, trunk_b)
        bionet = self.bionet(out)
        out = torch.add(conv1, bionet)

        # Upsampling layers
        out = self.upsampling(out)

        # Final output layer
        out = self.conv2(out)

        return torch.tanh(out)
