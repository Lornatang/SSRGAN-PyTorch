# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from .utils import ReceptiveFieldBlock
from .utils import ResidualInResidualDenseBlock
from .utils import ResidualOfReceptiveFieldDenseBlock

model_urls = {
    "rfb_4x4": None,
    "rfb": None
}


class SubpixelConvolutionLayer(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
        """
        super(SubpixelConvolutionLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.rfb1 = ReceptiveFieldBlock(channels, channels)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.rfb2 = ReceptiveFieldBlock(channels, channels)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample(x)
        out = self.rfb1(out)
        out = self.leaky_relu1(out)
        out = self.conv(out)
        out = self.pixel_shuffle(out)
        out = self.rfb2(out)
        out = self.leaky_relu2(out)

        return out


class Generator(nn.Module):
    def __init__(self, upscale_factor: int = 16) -> None:
        r"""
        Args:
            upscale_factor (int): How many times to upscale the picture. (Default: 16)
        """
        super(Generator, self).__init__()
        # Calculating the number of subpixel convolution layers.
        num_subpixel_convolution_layers = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # 16 ResidualInResidualDenseBlock layer.
        residual_residual_dense_blocks = []
        for _ in range(16):
            residual_residual_dense_blocks += [ResidualInResidualDenseBlock(channels=64, growth_channels=32)]
        self.Trunk_a = nn.Sequential(*residual_residual_dense_blocks)

        # 8 ResidualOfReceptiveFieldDenseBlock layer.
        residual_residual_fields_dense_blocks = []
        for _ in range(8):
            residual_residual_fields_dense_blocks += [ResidualOfReceptiveFieldDenseBlock(channels=64, growth_channels=32)]
        self.Trunk_RFB = nn.Sequential(*residual_residual_fields_dense_blocks)

        # Second conv layer post residual field blocks
        self.RFB = ReceptiveFieldBlock(64,  64)

        # Sub-pixel convolution layers.
        subpixel_conv_layers = []
        for _ in range(num_subpixel_convolution_layers):
            subpixel_conv_layers.append(SubpixelConvolutionLayer(channels=64))
        self.subpixel_conv = nn.Sequential(*subpixel_conv_layers)

        # Next conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer.
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)

        trunk_a = self.Trunk_a(conv1)
        trunk_rfb = self.Trunk_RFB(trunk_a)
        out = torch.add(conv1, trunk_rfb)

        out = self.RFB(out)
        out = self.subpixel_conv(out)
        out = self.conv2(out)
        out = self.conv3(out)

        return out


def _gan(arch: str, upscale_factor: int, pretrained: bool, progress: bool) -> Generator:
    model = Generator(upscale_factor)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def rfb_4x4(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/2005.12597>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("rfb_4x4", 4, pretrained, progress)


def rfb(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/2005.12597>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("rfb", 16, pretrained, progress)
