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

from ssrgan.activation import Mish

__all__ = ["BioNet", "bionet"]

model_urls = {
    "bionet": ""
}


class SymmetricBlock(nn.Module):
    r""" U-shaped network.

    `"U-Net: Convolutional Networks for Biomedical
    Image Segmentation" <https://arxiv.org/abs/1505.04597>`_ paper

    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        r""" Modules introduced in U-Net paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """
        super(SymmetricBlock, self).__init__()

        # Down sampling.
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            Mish(),
            nn.Conv2d(in_channels, in_channels // 2, 1, 1, 0, bias=False),
            Mish(),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2, bias=False),
            Mish(),
            nn.Conv2d(in_channels // 2, in_channels // 4, 1, 1, 0, bias=False)
        )

        # Up sampling.
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, 1, 1, groups=in_channels // 4, bias=False),
            Mish(),
            nn.Conv2d(in_channels // 4, in_channels // 2, 1, 1, 0, bias=False),
            Mish(),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2, bias=False),
            Mish(),
            nn.Conv2d(in_channels // 2, out_channels, 1, 1, 0, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Down sampling.
        out = self.down(input)
        # Up sampling.
        out = self.up(out)

        return out + input


class DepthwiseBlock(nn.Module):
    r""" Improved convolution method based on MobileNet-v2 version.

    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_ paper.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        r""" Modules introduced in MobileNetV2 paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        """
        super(DepthwiseBlock, self).__init__()

        # pw
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            Mish()
        )

        # dw
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            Mish()
        )

        # pw-linear
        self.pointwise_linear = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Expansion convolution
        out = self.pointwise(input)
        # DepthWise convolution
        out = self.depthwise(out)
        # Projection convolution
        out = self.pointwise_linear(out)

        return out + input


class InceptionBlock(nn.Module):
    r""" Base on InceptionV4

    `"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    " <https://arxiv.org/pdf/1602.07261.pdf>`_ paper.

    """

    def __init__(self, in_channels: int, out_channels: int, non_linearity: bool = True) -> None:
        r""" Modules introduced in InceptionX paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            non_linearity (optional, bool): Does the last layer use nonlinear activation. (Default: ``True``).
        """
        super(InceptionBlock, self).__init__()
        branch_features = int(in_channels // 4)

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, 1, 1, 0, bias=False),
            Mish(),
            nn.Conv2d(branch_features, branch_features, 3, 1, 3, dilation=3, bias=False),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, 1, 1, 0, bias=False),
            Mish(),
            nn.Conv2d(branch_features, branch_features, (1, 3), 1, (0, 1), bias=False),
            Mish(),
            nn.Conv2d(branch_features, branch_features, 3, 1, 3, dilation=3, bias=False),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, 1, 1, 0, bias=False),
            Mish(),
            nn.Conv2d(branch_features, branch_features, (3, 1), 1, (1, 0), bias=False),
            Mish(),
            nn.Conv2d(branch_features, branch_features, 3, 1, 3, dilation=3, bias=False),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_features, 1, 1, 0, bias=False),
            Mish(),
            nn.Conv2d(branch_features, branch_features, (1, 3), 1, (0, 1), bias=False),
            Mish(),
            nn.Conv2d(branch_features, branch_features, (3, 1), 1, (1, 0), bias=False),
            Mish(),
            nn.Conv2d(branch_features, branch_features, 3, 1, 3, dilation=3, bias=False),
        )

        self.conv1x1 = nn.Conv2d(branch_features * 4, branch_features * 4, 1, 1, 0, bias=False)
        self.Mish = Mish() if non_linearity else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(input)

        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.conv1x1(out)

        out = out + shortcut
        if self.Mish is not None:
            out = self.Mish(out)

        return out


#
#
# class InceptionDenseBlock(nn.Module):
#     r""" Inception dense network.
#     """
#
#     def __init__(self, channels: int, growth_channels: int = 32, scale_ratio: float = 0.2) -> None:
#         r""" Modules introduced in InceptionV4 paper.
#
#         Args:
#             channels (int): Number of channels in the input image.
#             growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
#             scale_ratio (float): Residual channel scaling column. (Default: 0.2)
#         """
#         super(InceptionDenseBlock, self).__init__()
#         self.IB1 = InceptionBlock(channels + 0 * growth_channels, growth_channels)
#         self.IB2 = InceptionBlock(channels + 1 * growth_channels, growth_channels)
#         self.IB3 = InceptionBlock(channels + 2 * growth_channels, growth_channels)
#         self.IB4 = InceptionBlock(channels + 3 * growth_channels, growth_channels)
#         self.IB5 = InceptionBlock(channels + 4 * growth_channels, channels, non_linearity=False)
#
#         self.scale_ratio = scale_ratio
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         ib1 = self.IB1(input)
#         ib2 = self.IB2(torch.cat((input, ib1), dim=1))
#         ib3 = self.IB3(torch.cat((input, ib1, ib2), dim=1))
#         ib4 = self.IB4(torch.cat((input, ib1, ib2, ib3), dim=1))
#         ib5 = self.IB5(torch.cat((input, ib1, ib2, ib3, ib4), dim=1))
#
#         return ib5.mul(self.scale_ratio) + input


class BioNet(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        super(BioNet, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)

        self.trunk_a = nn.Sequential(
            DepthwiseBlock(32, 32),
            DepthwiseBlock(32, 32)
        )
        self.trunk_b = nn.Sequential(
            SymmetricBlock(32, 32),
            SymmetricBlock(32, 32)
        )
        self.trunk_c = nn.Sequential(
            DepthwiseBlock(32, 32),
            DepthwiseBlock(32, 32)
        )
        self.trunk_d = nn.Sequential(
            InceptionBlock(32, 32),
            InceptionBlock(32, 32)
        )

        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                Mish(),
                nn.Conv2d(32, 128, 3, 1, 1, bias=False),
                Mish(),
                nn.PixelShuffle(upscale_factor=2),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                Mish()
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next conv layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            Mish()
        )

        # Final output layer.
        self.conv4 = nn.Conv2d(32, 3, 3, 1, 1, bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # First conv layer.
        conv1 = self.conv1(input)

        # MobileNet trunk.
        trunk_a = self.trunk_a(conv1)
        # Concat conv1 and trunk a.
        out1 = torch.add(conv1, trunk_a)

        # Symmetric trunk.
        trunk_b = self.trunk_b(out1)
        # Concat conv1 and trunk b.
        out2 = torch.add(conv1, trunk_b)

        # MobileNet trunk.
        trunk_c = self.trunk_c(out2)
        # Concat conv1 and trunk c.
        out3 = torch.add(conv1, trunk_c)

        # Inception trunk.
        trunk_d = self.trunk_b(out3)
        # Concat conv1 and trunk d.
        out4 = torch.add(conv1, trunk_d)

        # conv2 layer.
        conv2 = self.conv2(out4)
        # Concat conv1 and conv2 layer.
        out = torch.add(conv1, conv2)

        # Upsampling layers.
        out = self.upsampling(out)
        # Next conv layer.
        out = self.conv3(out)
        # Final output layer.
        out = self.conv4(out)

        return torch.tanh(out)


def bionet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> BioNet:
    r"""BioNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/2020.00000>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = BioNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["bionet"], progress=progress)
        model.load_state_dict(state_dict)
    return model
