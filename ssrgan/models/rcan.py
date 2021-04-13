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
import torch.nn as nn
import torch

from torch.hub import load_state_dict_from_url

model_urls = {
    "rcan_2x2": None,
    "rcan_3x3": None,
    "rcan": None,
    "rcan_8x8": None
}


# Source code reference `https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py`.
class ChannelAttention(nn.Module):
    def __init__(self, channels: int = 64, reduction: int = 16) -> None:
        r"""In order to make the network focus on more informative features, we exploit the interdependencies among feature channels, resulting in
        a channel attention (CA) mechanism

        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            reduction (int): Number of reduction channels in the input image. (Default: 16)
        """
        super(ChannelAttention, self).__init__()
        # Global average pooling: feature --> point.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Feature channel downscale and upscale --> channel weight.
        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.global_pool(x)

        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        # Multiplication of characteristic matrices
        out = torch.mul(x, out)

        return out


# Source code reference `https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py`.
class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channels: int = 64, reduction: int = 16) -> None:
        r"""As discussed above, residual groups and long skip connection allow the main parts of network to focus on more informative components
        of the LR features.

        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            reduction (int): Number of reduction channels in the input image. (Default: 16)
        """
        super(ResidualChannelAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.CA = ChannelAttention(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.CA(out)

        out = torch.add(out, x)

        return out


# Source code reference `https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py`.
class ResidualGroup(nn.Module):
    def __init__(self, channels: int = 64, reduction: int = 16) -> None:
        super(ResidualGroup, self).__init__()
        # Contains 20 ResidualChannelAttentionBlock.
        rcab_blocks = []
        for _ in range(20):
            rcab_blocks.append(ResidualChannelAttentionBlock(channels, reduction))
        self.rcab = nn.Sequential(*rcab_blocks)

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rcab(x)
        out = self.conv(out)

        out = torch.add(out, x)

        return out


# Source code reference `https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py`.
class SubpixelConvolutionLayer(nn.Module):
    def __init__(self, channels: int = 64, upscale_factor: int = 4) -> None:
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            upscale_factor (int): How many times to upscale the picture. (Default: 4)
        """
        super(SubpixelConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.pixel_shuffle(out)

        return out


# Source code reference `https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py`.
class RCAN(nn.Module):
    r"""Residual Channel Attention Network"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r"""
        Args:
            upscale_factor (int): How many times to upscale the picture. (Default: 4)
        """
        super(RCAN, self).__init__()

        # First layer convolution layer.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Contains 10 ResidualGroup.
        rg_blocks = []
        for _ in range(10):
            rg_blocks.append(ResidualGroup(64, 16))
        self.trunk = nn.Sequential(*rg_blocks)

        # Second conv layer post residual blocks.
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Sub-pixel convolution layers.
        self.subpixel_conv = SubpixelConvolutionLayer(64, upscale_factor=upscale_factor)

        # Output RGB channel image.
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First convolution layer.
        conv1 = self.conv1(x)

        # ResidualGroup network with 10 layers.
        trunk = self.trunk(conv1)

        # Second convolution layer.
        conv2 = self.conv2(trunk)
        # First convolution and second convolution feature image fusion.
        out = torch.add(conv1, conv2)
        # Using sub-pixel convolution layer to improve image resolution.
        out = self.subpixel_conv(out)
        # Output RGB channel image.
        out = self.conv3(out)

        return out


def _rcan(arch: str, upscale_factor: int, pretrained: bool, progress: bool) -> RCAN:
    model = RCAN(upscale_factor)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def rcan_2x2(pretrained: bool = False, progress: bool = True) -> RCAN:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1807.02758>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _rcan("rcan_2x2", 2, pretrained, progress)


def rcan_3x3(pretrained: bool = False, progress: bool = True) -> RCAN:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1807.02758>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _rcan("rcan_3x3", 3, pretrained, progress)


def rcan(pretrained: bool = False, progress: bool = True) -> RCAN:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1807.02758>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _rcan("rcan", 4, pretrained, progress)


def rcan_8x8(pretrained: bool = False, progress: bool = True) -> RCAN:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/1807.02758>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _rcan("rcan_8x8", 8, pretrained, progress)
