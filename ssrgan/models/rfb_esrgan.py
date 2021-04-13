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
    "rfb_4x4": None,
    "rfb": None
}


# Source code reference from `https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py`.
class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int = 64, growth_channels: int = 32):
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), dim=1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), dim=1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), dim=1))
        conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), dim=1))

        out = torch.add(conv5 * 0.2, x)

        return out


# Source code reference from `https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py`.
class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, channels: int = 64, growth_channels: int = 32):
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(channels, growth_channels)
        self.RDB2 = ResidualDenseBlock(channels, growth_channels)
        self.RDB3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)

        out = torch.add(out * 0.2, x)

        return out


# Source code reference from `https://github.com/ruinmessi/RFBNet/blob/master/models/RFB_Net_vgg.py`.
class ReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64):
        r""" Modules introduced in RFBNet paper.
        Args:
            in_channels (int): Number of channels in the input image. (Default: 64)
            out_channels (int): Number of channels produced by the convolution. (Default: 64)
        """
        super(ReceptiveFieldBlock, self).__init__()
        branch_channels = in_channels // 4

        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 3, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels // 2, (branch_channels // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d((branch_channels // 4) * 3, branch_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=5, dilation=5)
        )

        self.conv_linear = nn.Conv2d(4 * branch_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        out = self.conv_linear(out)

        out = self.leaky_relu(torch.add(out * 0.1, shortcut))

        return out


# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ReceptiveFieldDenseBlock(nn.Module):
    r""" Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
        RFB-SSD proposed Receptive Fields Block (RFB) for object detection
    """

    def __init__(self, channels: int = 64, growth_channels: int = 32):
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        """
        super(ReceptiveFieldDenseBlock, self).__init__()
        self.rfb1 = nn.Sequential(
            ReceptiveFieldBlock(channels + 0 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb2 = nn.Sequential(
            ReceptiveFieldBlock(channels + 1 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb3 = nn.Sequential(
            ReceptiveFieldBlock(channels + 2 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb4 = nn.Sequential(
            ReceptiveFieldBlock(channels + 3 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb5 = ReceptiveFieldBlock(channels + 4 * growth_channels, channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rfb1 = self.rfb1(x)
        rfb2 = self.rfb2(torch.cat((x, rfb1), dim=1))
        rfb3 = self.rfb3(torch.cat((x, rfb1, rfb2), dim=1))
        rfb4 = self.rfb4(torch.cat((x, rfb1, rfb2, rfb3), dim=1))
        rfb5 = self.rfb5(torch.cat((x, rfb1, rfb2, rfb3, rfb4), dim=1))

        out = torch.add(rfb5 * 0.1, x)

        return out


# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ResidualOfReceptiveFieldDenseBlock(nn.Module):
    r"""The residual block structure of traditional RFB-ESRGAN is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 32):
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        """
        super(ResidualOfReceptiveFieldDenseBlock, self).__init__()
        self.RFDB1 = ReceptiveFieldDenseBlock(channels, growth_channels)
        self.RFDB2 = ReceptiveFieldDenseBlock(channels, growth_channels)
        self.RFDB3 = ReceptiveFieldDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RFDB1(x)
        out = self.RFDB1(out)
        out = self.RFDB1(out)

        out = torch.add(out * 0.1, x)

        return out


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
        self.RFB = ReceptiveFieldBlock(64, 64)

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
        # First convolution layer.
        conv1 = self.conv1(x)

        # ResidualInResidualDenseBlock network with 16 layers.
        trunk_a = self.Trunk_a(conv1)
        trunk_rfb = self.Trunk_RFB(trunk_a)
        # First convolution and ResidualOfReceptiveFieldDenseBlock feature image fusion.
        out = torch.add(conv1, trunk_rfb)

        # First ReceptiveFieldBlock layer.
        out = self.RFB(out)

        # Using sub-pixel convolution layer to improve image resolution.
        out = self.subpixel_conv(out)
        # Second convolution layer.
        out = self.conv2(out)
        # Output RGB channel image.
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
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _gan("rfb_4x4", 4, pretrained, progress)


def rfb(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the `"One weird trick..." <https://arxiv.org/abs/2005.12597>` paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _gan("rfb", 16, pretrained, progress)
