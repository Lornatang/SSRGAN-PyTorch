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

model_urls = {
    "pmi_srgan": "https://github.com/Lornatang/SSRGAN-PyTorch/releases/download/0.1.0/RRDBNet_4x4_16_DF2K-e31a1b2e.pth"
}


class Symmetric(nn.Module):
    r""" PyTorch implementation U-Net module.

    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>` paper.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        r""" Modules introduced in U-Net paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """
        super(Symmetric, self).__init__()

        # The first layer convolution block of down-sampling module in U-Net network.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, 1, 1),
            Mish(),
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, 1),
            Mish()
        )

        # Down sampling layer.
        self.down_sampling_layer = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 2, 1),
            Mish()
        )

        # The second layer convolution block of up-sampling module in U-Net network.
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, 3, 1, 1),
            Mish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            Mish()
        )

        # Up sampling layer.
        self.up_sampling_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            Mish(),
            nn.Conv2d(out_channels, out_channels // 2, 3, 1, 1),
            Mish()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Down-sampling layer.
        conv1 = self.conv1(x)
        down_sampling_layer = self.down_sampling_layer(conv1)
        # Up-sampling layer.
        conv2 = self.conv2(down_sampling_layer)
        up_sampling_layer = self.up_sampling_layer(conv2)
        # Concat up layer and down layer.
        out = torch.cat((up_sampling_layer, conv1), 1)

        return out + x


# Source code reference from `https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py`.
class DepthWise(nn.Module):
    r""" PyTorch implementation MobileNet-v2 module.

    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_ paper.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        r""" Modules introduced in MobileNetV2 paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """
        super(DepthWise, self).__init__()

        # pw
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            Mish()
        )

        # dw
        self.depthwise = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            Mish()
        )

        # pw-linear
        self.pointwise_linear = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expansion convolution
        out = self.pointwise(x)
        # DepthWise convolution
        out = self.depthwise(out)
        # Projection convolution
        out = self.pointwise_linear(out)

        return out + x


class InceptionX(nn.Module):
    r""" PyTorch implementation RFBNet/Inception-V4 module.

    `"Receptive Field Block Net for Accurate and Fast Object Detection <https://arxiv.org/pdf/1711.07767v3.pdf>` paper.
    `"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning <https://arxiv.org/pdf/1602.07261v2.pdf>` paper.
    """

    def __init__(self, in_channels: int, out_channels: int, scale: float) -> None:
        r""" Modules introduced in RFBNet/Inception-V4 paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            scale (float): Scaling output channel number factor.
        """
        super(InceptionX, self).__init__()
        self.scale = scale
        branch_features = in_channels // 4

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, 1, 1, 0),
            Mish(),
            nn.Conv2d(branch_features, branch_features, 3, 1, 3, dilation=3)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, 1, 1, 0),
            Mish(),
            nn.Conv2d(branch_features, branch_features, (3, 1), 1, (1, 0)),
            Mish(),
            nn.Conv2d(branch_features, branch_features, 3, 1, 3, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, 1, 1, 0),
            Mish(),
            nn.Conv2d(branch_features, branch_features, (1, 3), 1, (0, 1)),
            Mish(),
            nn.Conv2d(branch_features, branch_features, 3, 1, 3, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features // 2, 1, 1, 0),
            Mish(),
            nn.Conv2d(branch_features // 2, (branch_features // 4) * 3, (1, 3), 1, (0, 1)),
            Mish(),
            nn.Conv2d((branch_features // 4) * 3, branch_features, (3, 1), 1, (1, 0)),
            Mish(),
            nn.Conv2d(branch_features, branch_features, 3, 1, 3, dilation=3),
        )

        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.mish = Mish()

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

        out = torch.cat([branch1, branch2, branch3, branch4], 1)
        out = self.conv(out)
        out = out * self.scale + shortcut
        out = self.mish(out)

        return out


# Source code reference `https://arxiv.org/pdf/2005.12597v1.pdf` paper.
class SubpixelConvolutionLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        """
        Args:
            channels (int): Number of channels in the input image.
        """
        super(SubpixelConvolutionLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.inception = InceptionX(channels, channels, 0.1)
        self.conv = nn.Conv2d(channels, channels * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
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

    def __init__(self, upscale_factor: int) -> None:
        r"""

        Args:
            upscale_factor (int): How many times to upscale the picture.
        """
        super(Generator, self).__init__()
        # Calculating the number of subpixel convolution layers.
        num_subpixel_convolution_layers = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)

        self.trunk_a = nn.Sequential(
            DepthWise(32, 32),
            DepthWise(32, 32)
        )

        self.trunk_b = nn.Sequential(
            Symmetric(32, 32),
            Symmetric(32, 32)
        )

        self.trunk_c = nn.Sequential(
            DepthWise(32, 32),
            DepthWise(32, 32)
        )

        self.trunk_d = nn.Sequential(
            InceptionX(32, 32, 0.1),
            InceptionX(32, 32, 0.1),
        )

        self.symmetric_conv = Symmetric(32, 32)

        # Sub-pixel convolution layers.
        subpixel_conv_layers = []
        for _ in range(num_subpixel_convolution_layers):
            subpixel_conv_layers.append(SubpixelConvolutionLayer(32))
        self.subpixel_conv = nn.Sequential(*subpixel_conv_layers)

        # Next conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            Mish()
        )

        # Final output layer.
        self.conv3 = nn.Conv2d(32, 3, 3, 1, 1)

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
