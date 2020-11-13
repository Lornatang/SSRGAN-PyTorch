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

from .esrgan import ResidualInResidualDenseBlock
from .utils import conv1x1
from .utils import conv3x3

__all__ = [
    "ReceptiveFieldBlock", "ReceptiveFieldDenseBlock", "ResidualOfReceptiveFieldDenseBlock", "RFBESRGAN", "rfb_esrgan"
]

model_urls = {
    "rfb_esrgan": ""
}


class ReceptiveFieldBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, in_channels: int, out_channels: int, scale_ratio: float = 0.2, non_linearity: bool = True):
        """
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            scale_ratio (optional, float): Residual channel scaling column. (Default: 0.2)
            non_linearity (optional, bool): Does the last layer use nonlinear activation. (Default: ``True``).
        """
        super(ReceptiveFieldBlock, self).__init__()
        channels = in_channels // 4
        # shortcut layer
        self.shortcut = conv1x1(in_channels, out_channels)

        self.branch1 = nn.Sequential(
            conv1x1(in_channels, channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(channels, channels, padding=1, dilation=1, groups=1)
        )

        self.branch2 = nn.Sequential(
            conv1x1(in_channels, channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(channels, channels, padding=3, dilation=3, groups=1)
        )

        self.branch3 = nn.Sequential(
            conv1x1(in_channels, channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0), groups=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(channels, channels, padding=3, dilation=3, groups=1)
        )

        self.branch4 = nn.Sequential(
            conv1x1(in_channels, channels // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(channels // 2, (channels // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3((channels // 4) * 3, channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(channels, channels, padding=5, dilation=5, groups=1)
        )

        self.conv1x1 = conv1x1(in_channels, out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True) if non_linearity else None

        self.scale_ratio = scale_ratio

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
        shortcut = self.shortcut(input)

        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)

        out = torch.cat((branch1, branch2, branch3, branch4), 1)
        out = self.conv1x1(out)

        out = out.mul(self.scale_ratio) + shortcut
        if self.lrelu is not None:
            out = self.lrelu(out)

        return out


class ReceptiveFieldDenseBlock(nn.Module):
    """Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
        RFB-SSD proposed Receptive Fields Block (RFB) for object detection
    """

    def __init__(self, in_channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2) -> None:
        """
        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ReceptiveFieldDenseBlock, self).__init__()
        self.RFB1 = ReceptiveFieldBlock(in_channels, growth_channels, scale_ratio)
        self.RFB2 = ReceptiveFieldBlock(in_channels + 1 * growth_channels, growth_channels, scale_ratio)
        self.RFB3 = ReceptiveFieldBlock(in_channels + 2 * growth_channels, growth_channels, scale_ratio)
        self.RFB4 = ReceptiveFieldBlock(in_channels + 3 * growth_channels, growth_channels, scale_ratio)
        self.RFB5 = ReceptiveFieldBlock(in_channels + 4 * growth_channels, in_channels, scale_ratio,
                                        non_linearity=False)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        rfb1 = self.RFB1(input)
        rfb2 = self.RFB2(torch.cat((input, rfb1), 1))
        rfb3 = self.RFB3(torch.cat((input, rfb1, rfb2), 1))
        rfb4 = self.RFB4(torch.cat((input, rfb1, rfb2, rfb3), 1))
        rfb5 = self.RFB5(torch.cat((input, rfb1, rfb2, rfb3, rfb4), 1))

        return rfb5.mul(self.scale_ratio) + input


class ResidualOfReceptiveFieldDenseBlock(nn.Module):
    r"""The residual block structure of traditional RFB-ESRGAN is defined"""

    def __init__(self, in_channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.2) -> None:
        """
        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualOfReceptiveFieldDenseBlock, self).__init__()
        self.RFDB1 = ReceptiveFieldDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RFDB2 = ReceptiveFieldDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RFDB3 = ReceptiveFieldDenseBlock(in_channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.RFDB1(input)
        out = self.RFDB2(out)
        out = self.RFDB3(out)

        return out.mul(self.scale_ratio) + input


class RFBESRGAN(nn.Module):
    r""" It is mainly based on the SRGAN network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of SRGAN network structure.
                """
        super(RFBESRGAN, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = conv3x3(3, 64)

        trunk_a = []
        trunk_rfb = []
        # Sixteen structures similar to RFB-ESRGAN(Trunk-A) network.
        for _ in range(16):
            trunk_a.append(ResidualInResidualDenseBlock(64, 32, 0.2))
        self.trunk_a = nn.Sequential(*trunk_a)
        # Eight structures similar to RFB-ESRGAN(Trunk-RFB) network.
        for _ in range(8):
            trunk_rfb.append(ResidualOfReceptiveFieldDenseBlock(64, 32, 0.2))
        self.trunk_rfb = nn.Sequential(*trunk_rfb)

        self.rfbesrgan = ReceptiveFieldBlock(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                ReceptiveFieldBlock(64, 64),
                conv3x3(64, 256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2),
                ReceptiveFieldBlock(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        self.conv2 = conv3x3(64, 64, groups=1)

        # Final output layer
        self.conv3 = conv3x3(64, 3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(input)
        trunk_a = self.trunk_a(conv1)
        trunk_rfb = self.trunk_rfb(trunk_a)
        out = torch.add(conv1, trunk_rfb)
        out = self.esrgan(out)
        out = self.upsampling(out)
        out = self.conv2(out)
        out = self.conv3(out)
        return torch.tanh(out)


def rfb_esrgan(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> RFBESRGAN:
    r"""RFB-ESRGAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/2005.12597>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = RFBESRGAN(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["rfb_esrgan"], progress=progress)
        model.load_state_dict(state_dict)
    return model
