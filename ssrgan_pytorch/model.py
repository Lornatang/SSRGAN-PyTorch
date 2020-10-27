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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .activation import HSigmoid
from .activation import HSwish

__all__ = [
    "DepthWiseSeperabelConvolution", "DiscriminatorForVGG", "Fire", "Generator",
    "InvertedResidual", "MobileNetV3Bottleneck", "ReceptiveFieldBlock", "ReceptiveFieldDenseBlock",
    "ResidualBlock", "ResidualDenseBlock", "ResidualInResidualDenseBlock", "ResidualOfReceptiveFieldDenseBlock",
    "SEModule", "ShuffleNetV1", "ShuffleNetV2", "channel_shuffle"
]


class DepthWiseSeperabelConvolution(nn.Module):
    r"""Deep separable convolution implemented in mobilenet version 1. 

    `"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" <https://arxiv.org/abs/1704.04861>`_

    """

    def __init__(self, channels):
        r""" This is a structure for simple versions.

        Args:
            channels (int): Number of channels in the input image.
        """
        super(DepthWiseSeperabelConvolution, self).__init__()

        # dw
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # pw
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
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
        # DepthWise convolution
        out = self.depthwise(input)
        # Projection convolution
        out = self.pointwise(out)

        return out


class DiscriminatorForVGG(nn.Module):
    r"""The main architecture of the discriminator. Similar to VGG structure."""

    def __init__(self):
        super(DiscriminatorForVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # input is 3 x 216 x 216
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),  # state size. 64 x 108 x 108
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),  # state size. 128 x 54 x 54
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=2, bias=False),  # state size. 256 x 28 x 28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),  # state size. 512 x 14 x 14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),  # state size. 512 x 7 x 7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, input: Tensor) -> Tensor:
        out = self.features(input)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Fire(nn.Module):
    r""" SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    """

    def __init__(self, in_channels, out_channels, expand_channels):
        r""" Modules introduced in SqueezeNet paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_channels (int): Number of channels produced by the expand layer.
        """
        super(Fire, self).__init__()

        # squeeze
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

        # expand 1x1
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(out_channels, expand_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

        # expand 3x3
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(out_channels, expand_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
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


class Generator(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""
    __constants__ = ["upscale_factor", "block"]

    def __init__(self, upscale_factor=4, block="srgan", num_block=16):
        r""" This is an ssrgan model defined by the author himself.

        Args:
            upscale_factor (int): Image magnification factor. (Default: 4).
            block (str): Select the structure used by the backbone network. (Default: ``srgan``).
            num_block (int): How many block are combined. (Default: 16).
        """
        super(Generator, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 2))

        if block == "srgan":  # For SRGAN
            block = ResidualBlock(64)
        elif block == "esrgan":  # For ESRGAN
            block = ResidualInResidualDenseBlock(64, 32, 0.2)
        elif block == "rfb-esrgan":  # For RFB-ESRGAN
            block = ResidualOfReceptiveFieldDenseBlock(64, 32, 0.2)
        elif block == "squeezenet":  # For SqueezeNet
            block = Fire(64, 16, 32)
        elif block == "mobilenet-v1":  # For MobileNet v1
            block = DepthWiseSeperabelConvolution(64)
        elif block == "mobilenet-v2":  # For MobileNet v2
            block = InvertedResidual(24, 64)
        elif block == "mobilenet-v3":  # For MobileNet v3
            block = MobileNetV3Bottleneck(24, 64)
        elif block == "shufflenet-v1":  # For ShuffleNet v1
            block = ShuffleNetV1(64)
        elif block == "shufflenet-v2":  # For ShuffleNet v2
            block = ShuffleNetV2(64)
        else:
            raise NameError("Please check the block name, the block name must be "
                            "`srgan`, `esrgan`, `rfb-esrgan` or "
                            "`squeezenet` or "
                            "`mobilenet-v1`, `mobilenet-v2`, `mobilenet-v3` or "
                            "`shufflenet-v1`, `shufflenet-v2`.")

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 16 layer similar stack block structure.
        blocks = []
        for _ in range(num_block):
            blocks.append(block)
        self.Trunk = nn.Sequential(*blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2)
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
        out1 = self.conv1(input)
        out = self.Trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class InvertedResidual(nn.Module):
    r""" Improved convolution method based on MobileNet-v2 version.

    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_

    """

    def __init__(self, in_channels, out_channels, expand_factor=6):
        r""" This is a structure for simple versions.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, int): Channel expansion multiple. (Default: 6).
        """
        super(InvertedResidual, self).__init__()
        channels = in_channels * expand_factor

        self.shortcut = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # pw
        self.pointwise = nn.Sequential(
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # dw
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # pw-linear
        self.pointwise_linear = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
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
        # Expansion convolution
        out = self.pointwise(input)
        # DepthWise convolution
        out = self.depthwise(out)
        # Projection convolution
        out = self.pointwise_linear(out)

        return out + self.shortcut(input)


class MobileNetV3Bottleneck(nn.Module):
    r""" Base on MobileNetV2 + Squeeze-and-Excite.

    `"Searching for MobileNetV3" <https://arxiv.org/pdf/1905.02244.pdf>`_

    """

    def __init__(self, in_channels, out_channels, expand_factor=6):
        r""" This is a structure for simple versions.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution
            expand_factor (optional, int): Channel expansion multiple. (Default: 6).
        """
        super(MobileNetV3Bottleneck, self).__init__()
        channels = in_channels * expand_factor

        self.shortcut = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # pw
        self.pointwise = nn.Sequential(
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channels),
            HSwish()
        )

        # dw
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        # squeeze and excitation module.
        self.SEModule = nn.Sequential(
            SEModule(channels),
            HSwish()
        )

        # pw-linear
        self.pointwise_linear = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
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
        # Expansion convolution
        out = self.pointwise(input)
        # DepthWise convolution
        out = self.depthwise(out)
        # Squeeze-and-Excite
        out = self.SEModule(out)
        # Projection convolution
        out = self.pointwise_linear(out)

        return out + self.shortcut(input)


class ReceptiveFieldBlock(nn.Module):
    r"""This structure is similar to the main building blocks in the GoogLeNet model.
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    """

    def __init__(self, in_channels, out_channels, scale_ratio=0.2, non_linearity=True):
        super(ReceptiveFieldBlock, self).__init__()
        channels = in_channels // 4
        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, channels // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, (channels // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d((channels // 4) * 3, channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
        )

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
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

    def forward(self, input: Tensor) -> Tensor:
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
    r""" Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
        RFB-SSD proposed Receptive Fields Block (RFB) for object detection
    """

    def __init__(self, in_channels, growth_channels, scale_ratio):
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

    def forward(self, input: Tensor) -> Tensor:
        rfb1 = self.RFB1(input)
        rfb2 = self.RFB2(torch.cat((input, rfb1), 1))
        rfb3 = self.RFB3(torch.cat((input, rfb1, rfb2), 1))
        rfb4 = self.RFB4(torch.cat((input, rfb1, rfb2, rfb3), 1))
        rfb5 = self.RFB5(torch.cat((input, rfb1, rfb2, rfb3, rfb4), 1))

        return rfb5.mul(self.scale_ratio) + input


class ResidualBlock(nn.Module):
    r"""Main residual block structure"""

    def __init__(self, channels):
        r"""Initializes internal Module state, shared by both nn.Module and ScriptModule.

        Args:
            channels (int): Number of channels in the input image.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

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
        out = self.conv1(input)
        out = self.prelu(out)
        out = self.conv2(out)

        return out + input


class ResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional SRGAN and Dense model is defined"""

    def __init__(self, in_channels, growth_channels, scale_ratio):
        """

        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + 0 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels + 1 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1, bias=False)

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

    def forward(self, input: Tensor) -> Tensor:
        conv1 = self.conv1(input)
        conv2 = self.conv2(torch.cat((input, conv1), 1))
        conv3 = self.conv3(torch.cat((input, conv1, conv2), 1))
        conv4 = self.conv4(torch.cat((input, conv1, conv2, conv3), 1))
        conv5 = self.conv5(torch.cat((input, conv1, conv2, conv3, conv4), 1))

        return conv5.mul(self.scale_ratio) + input


class ResidualInResidualDenseBlock(nn.Module):
    r"""The residual block structure of traditional ESRGAN and Dense model is defined"""

    def __init__(self, in_channels, growth_channels, scale_ratio):
        """

        Args:
            in_channels (int): Number of channels in the input image.
            growth_channels (int): how many filters to add each layer (`k` in paper).
            scale_ratio (float): Residual channel scaling column.
        """
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB1 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RDB2 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)
        self.RDB3 = ResidualDenseBlock(in_channels, growth_channels, scale_ratio)

        self.scale_ratio = scale_ratio

    def forward(self, input: Tensor) -> Tensor:
        out = self.RDB1(input)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return out.mul(self.scale_ratio) + input


class ResidualOfReceptiveFieldDenseBlock(nn.Module):
    r"""The residual block structure of traditional RFB-ESRGAN is defined"""

    def __init__(self, in_channels=64, growth_channels=32, scale_ratio=0.2):
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

    def forward(self, input: Tensor) -> Tensor:
        out = self.RFDB1(input)
        out = self.RFDB2(out)
        out = self.RFDB3(out)

        return out.mul(self.scale_ratio) + input


class SEModule(nn.Module):
    r""" Squeeze-and-Excite module.

    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile" <https://arxiv.org/pdf/1807.11626.pdf>`_

    """

    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            HSigmoid()
        )

    def forward(self, input: Tensor) -> Tensor:
        b, c, _, _ = input.size()
        out = self.avgpool(input).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        return input * out.expand_as(input)


class ShuffleNetV1(nn.Module):
    r""" It mainly realizes the channel shuffling operation

    `"ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices" <https://arxiv.org/pdf/1707.01083.pdf>`_

    """

    def __init__(self, in_channels):
        r""" This is a structure for simple versions.

        Args:
            in_channels (int): Number of channels in the input image.
        """
        super(ShuffleNetV1, self).__init__()

        channels = in_channels // 4

        # pw
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, groups=4, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # dw
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # pw-linear
        self.pointwise_linear = nn.Sequential(
            nn.Conv2d(channels, in_channels, kernel_size=1, stride=1, padding=0, groups=4, bias=False),
            nn.BatchNorm2d(in_channels)
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
        # Expansion convolution
        out = self.pointwise(input)
        # Channel shuffle
        out = channel_shuffle(out, 4)
        # DepthWise convolution
        out = self.depthwise(out)
        # Projection convolution
        out = self.pointwise_linear(out)
        # Fusion input and out
        out = torch.add(input, out)
        out = F.relu(out, inplace=True)

        return out


class ShuffleNetV2(nn.Module):
    r""" It mainly realizes the channel shuffling operation

    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164.pdf>`_

    """

    def __init__(self, channels):
        r""" This is a structure for simple versions.

        Args:
            channels (int): Number of channels in the input image.
        """
        super(ShuffleNetV2, self).__init__()
        branch_features = channels // 2

        self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=1, padding=1, groups=branch_features,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
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
        x1, x2 = input.chunk(2, dim=1)
        out = torch.cat((x1, self.branch2(x2)), dim=1)
        out = channel_shuffle(out, 2)

        return out


# Source from `https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py`
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
