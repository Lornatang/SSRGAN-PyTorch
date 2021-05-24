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

model_urls = {
    "srgan_2x2": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/v0.2.2/SRGAN_2x2_ImageNet2012-8a6c37a4f51bd78920271bae11c9ab7882f8df066afba425cde360ff645a681a.pth",
    "srgan": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/v0.2.2/SRGAN_ImageNet2012-992702908bcbce3b6e2bc2d15eb5b4eb7a5c816468654819c6efbbd79ce671ea.pth",
    "srgan_8x8": "https://github.com/Lornatang/SRGAN-PyTorch/releases/download/v0.2.2/SRGAN_8x8_ImageNet2012-56374e6208b19eebe1a3fb2dd06a50c29fd7e61b75714c037e22de3a97d86135.pth"
}


class SubpixelConvolutionLayer(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
        """
        super(SubpixelConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + x

        return out


class Generator(nn.Module):
    def __init__(self, upscale_factor: int = 4) -> None:
        r"""
        Args:
            upscale_factor (int): How many times to upscale the picture. (Default: 4)
        """
        super(Generator, self).__init__()
        # Calculating the number of subpixel convolution layers.
        num_subpixel_convolution_layers = int(math.log(upscale_factor, 2))

        # First layer.
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # 16 Residual blocks.
        trunk = []
        for _ in range(16):
            trunk.append(ResidualBlock(channels=64))
        self.trunk = nn.Sequential(*trunk)

        # Second conv layer post residual blocks.
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        # 2 Sub-pixel convolution layers.
        subpixel_conv_layers = []
        for _ in range(num_subpixel_convolution_layers):
            subpixel_conv_layers.append(SubpixelConvolutionLayer(64))
        self.subpixel_conv = nn.Sequential(*subpixel_conv_layers)

        # Final output layer.
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        trunk = self.trunk(conv1)
        conv2 = self.conv2(trunk)
        out = torch.add(conv1, conv2)
        out = self.subpixel_conv(out)
        out = self.conv3(out)

        return out


def _gan(arch: str, upscale_factor: int, pretrained: bool, progress: bool) -> Generator:
    model = Generator(upscale_factor)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def srgan_2x2(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1609.04802>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("srgan_2x2", 2, pretrained, progress)


def srgan(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1609.04802>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("srgan", 4, pretrained, progress)


def srgan_8x8(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1609.04802>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("srgan_8x8", 8, pretrained, progress)
