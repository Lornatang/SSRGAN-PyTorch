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

__all__ = ["Fire", "SqueezeNet"]


class Fire(nn.Module):
    r""" SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """

    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        r""" Modules introduced in SqueezeNet paper.

        Args:
            in_channels (int): Number of channels in the input image.
            squeeze_channels (int): Number of channels produced by the squeeze layer.
            expand1x1_channels (int): Number of channels produced by the expand 1x1 layer.
            expand3x3_channels (int): Number of channels produced by the expand 3x3 layer.
        """
        super(Fire, self).__init__()

        # squeeze
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # expand 1x1
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # expand 3x3
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, stride=1, padding=1, bias=False),
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
        # Squeeze convolution
        out = self.squeeze(input)
        # Expand convolution
        out = torch.cat([self.expand1x1(out), self.expand3x3(out)], 1)

        return out


class SqueezeNet(nn.Module):
    r""" It is mainly based on the SqueezeNet network as the backbone network generator"""

    def __init__(self):
        r""" This is made up of SqueezeNet network structure.
        """
        super(SqueezeNet, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Eight structures similar to SqueezeNet network.
        trunk = []
        for _ in range(8):
            trunk.append(Fire(64, 64, 32, 32))
        self.Trunk = nn.Sequential(*trunk)

        self.fire = Fire(64, 64, 32, 32)

        # Upsampling layers
        upsampling = []
        for _ in range(1):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                Fire(64, 64, 32, 32),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2),
                Fire(64, 64, 32, 32),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next layer after upper sampling
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input: Tensor) -> Tensor:
        conv1 = self.conv1(input)
        trunk = self.Trunk(conv1)
        fire = self.fire(trunk)
        out = torch.add(conv1, fire)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out
