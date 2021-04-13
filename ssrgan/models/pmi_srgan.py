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
from ssrgan.activation import Mish
from .utils import DepthWise
from .utils import Symmetric
from .utils import InceptionX

model_urls = {
    "pmi_srgan": None,
}


# Source code reference `https://arxiv.org/pdf/2005.12597v1.pdf` paper.
class SubpixelConvolutionLayer(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        """
        Args:
            channels (int): Number of channels in the input image. (Default: 32)
        """
        super(SubpixelConvolutionLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.inception = InceptionX(channels)
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.mish = Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample(x)
        out = self.inception(out)
        out = self.mish(out)
        out = self.conv(out)
        out = self.pixel_shuffle(out)
        out = self.inception(out)
        out = self.mish(out)

        return out


class Generator(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r"""

        Args:
            upscale_factor (int): How many times to upscale the picture. (Default: 4)
        """
        super(Generator, self).__init__()
        # Calculating the number of subpixel convolution layers.
        num_subpixel_convolution_layers = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        self.trunk_a = nn.Sequential(
            DepthWise(32),
            DepthWise(32)
        )

        self.trunk_b = nn.Sequential(
            Symmetric(32),
            Symmetric(32)
        )

        self.trunk_c = nn.Sequential(
            DepthWise(32),
            DepthWise(32)
        )

        self.trunk_d = nn.Sequential(
            InceptionX(32),
            InceptionX(32),
        )

        self.symmetric_conv = Symmetric(32)

        # Sub-pixel convolution layers.
        subpixel_conv_layers = []
        for _ in range(num_subpixel_convolution_layers):
            subpixel_conv_layers.append(SubpixelConvolutionLayer(32))
        self.subpixel_conv = nn.Sequential(*subpixel_conv_layers)

        # Next conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            Mish()
        )

        # Final output layer.
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)

        trunk_a = self.trunk_a(conv1)
        out1 = torch.add(conv1, trunk_a)
        trunk_b = self.trunk_b(out1)
        out2 = torch.add(conv1, trunk_b)
        trunk_c = self.trunk_c(out2)
        out3 = torch.add(conv1, trunk_c)
        trunk_d = self.trunk_d(out3)
        out4 = torch.add(conv1, trunk_d)

        symmetric_conv = self.symmetric_conv(out4)
        out = torch.add(conv1, symmetric_conv)

        out = self.subpixel_conv(out)
        out = self.conv2(out)
        out = self.conv3(out)

        return out


def _gan(arch: str, upscale_factor: int, pretrained: bool, progress: bool) -> Generator:
    r""" Used to create GAN model.

    Args:
        arch (str): GAN model architecture name.
        upscale_factor (int): How many times to upscale the picture.
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.

    Returns:
        Generator model.
    """
    model = Generator(upscale_factor)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def pmi_srgan(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/2021.00000>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _gan("pmi_srgan", 4, pretrained, progress)
