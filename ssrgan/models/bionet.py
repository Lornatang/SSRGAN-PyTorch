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

from ssrgan.activation import FReLU
from ssrgan.models.utils import Conv
from ssrgan.models.utils import dw_conv

__all__ = ["SymmetricBlock",
           "DepthwiseBlock",
           "InceptionBlock", "InceptionDenseBlock",
           "BioNet", "bionet"]

model_urls = {
    "bionet": ""
}


class SymmetricBlock(nn.Module):
    r""" U-shaped network.

    `"U-Net: Convolutional Networks for Biomedical
    Image Segmentation" <https://arxiv.org/abs/1505.04597>`_ paper

    """

    def __init__(self, channels: int) -> None:
        r""" Modules introduced in U-Net paper.

        Args:
            channels (int): Number of channels in the input/output image.
        """
        super(SymmetricBlock, self).__init__()
        self.shortcut = Conv(channels, channels, kernel_size=1, stride=1, padding=0, act=False)

        # Down sampling.
        self.down = nn.Sequential(
            Conv(channels, channels // 2, kernel_size=3, stride=2, padding=1),
            dw_conv(channels // 2, channels // 2, kernel_size=3, stride=1, padding=3, dilation=3),
            Conv(channels // 2, channels // 4, kernel_size=1, stride=1, padding=0),
            dw_conv(channels // 4, channels // 4, kernel_size=3, stride=1, padding=3, dilation=3),
            Conv(channels // 4, channels // 8, kernel_size=1, stride=1, padding=0, act=False)
        )

        # Up sampling.
        self.up = nn.Sequential(
            nn.ConvTranspose2d(channels // 8, channels // 4, kernel_size=2, stride=2),
            dw_conv(channels // 4, channels // 4, kernel_size=3, stride=1, padding=3, dilation=3),
            Conv(channels // 4, channels // 2, kernel_size=1, stride=1, padding=0),
            dw_conv(channels // 2, channels // 2, kernel_size=3, stride=1, padding=3, dilation=3),
            Conv(channels // 2, channels, kernel_size=1, stride=1, padding=0, act=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1x1 nonlinear characteristic output.
        shortcut = self.shortcut(input)

        # Down sampling.
        out = self.down(input)
        # Up sampling.
        out = self.up(out)

        return out + shortcut


class DepthwiseBlock(nn.Module):
    r""" Improved convolution method based on MobileNet-v2 version.

    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_ paper.
    """

    def __init__(self, channels: int) -> None:
        r""" Modules introduced in MobileNetV2 paper.

        Args:
            channels (int): Number of channels in the input/output image.
        """
        super(DepthwiseBlock, self).__init__()
        self.shortcut = Conv(channels, channels, kernel_size=1, stride=1, padding=0, act=False)

        # pw
        self.pointwise = Conv(channels, channels // 2, kernel_size=1, stride=1, padding=0)

        # dw
        self.depthwise = dw_conv(channels // 2, channels // 2, kernel_size=3, stride=1, padding=3, dilation=3)

        # pw-linear
        self.pointwise_linear = Conv(channels // 2, channels, kernel_size=1, stride=1, padding=0, act=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1x1 nonlinear characteristic output.
        shortcut = self.shortcut(input)

        # Expansion convolution
        out = self.pointwise(input)
        # DepthWise convolution
        out = self.depthwise(out)
        # Projection convolution
        out = self.pointwise_linear(out)

        return out + shortcut


class InceptionBlock(nn.Module):
    r""" Base on InceptionV4

    `"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    " <https://arxiv.org/pdf/1602.07261.pdf>`_ paper.

    """

    def __init__(self, in_channels: int, out_channels: int, scale_ratio: float = 0.2,
                 non_linearity: bool = True) -> None:
        r""" Modules introduced in InceptionX paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            scale_ratio (optional, float): Residual channel scaling column. (Default: 0.5).
            non_linearity (optional, bool): Does the last layer use nonlinear activation. (Default: ``True``).
        """
        super(InceptionBlock, self).__init__()
        branch_features = int(in_channels // 4)

        self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=False)

        self.branch1 = nn.Sequential(
            Conv(in_channels, branch_features, kernel_size=1, stride=1, padding=0),
            dw_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            dw_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            Conv(branch_features, branch_features, kernel_size=3, stride=1, padding=3, dilation=3, act=False)
        )
        self.branch2 = nn.Sequential(
            Conv(in_channels, branch_features, kernel_size=1, stride=1, padding=0),
            dw_conv(branch_features, branch_features, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            dw_conv(branch_features, branch_features, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            Conv(branch_features, branch_features, kernel_size=3, stride=1, padding=3, dilation=3, act=False)
        )
        self.branch3 = nn.Sequential(
            Conv(in_channels, branch_features, kernel_size=1, stride=1, padding=0),
            dw_conv(branch_features, branch_features, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            dw_conv(branch_features, branch_features, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            Conv(branch_features, branch_features, kernel_size=3, stride=1, padding=3, dilation=3, act=False)
        )
        self.branch4 = nn.Sequential(
            Conv(in_channels, branch_features, kernel_size=1, stride=1, padding=0),
            dw_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            dw_conv(branch_features, branch_features, kernel_size=3, stride=1, padding=1),
            Conv(branch_features, branch_features, kernel_size=3, stride=1, padding=3, dilation=3, act=False)
        )

        self.branch_concat = Conv(branch_features * 2, branch_features * 2, kernel_size=1, stride=1, padding=0)

        self.conv1x1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=False)
        self.frelu = FReLU(out_channels) if non_linearity else None

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(input)

        branch1 = self.branch1(input)
        branch4 = self.branch4(input)
        branch_concat1_4 = torch.cat([branch1, branch4], dim=1)
        branch_out1 = self.branch_concat(branch_concat1_4)

        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch_concat2_3 = torch.cat([branch2, branch3], dim=1)
        branch_out2 = self.branch_concat(branch_concat2_3)

        # Concat layer
        out = torch.cat([branch_out1, branch_out2], dim=1)
        out = self.conv1x1(out)

        out = out + shortcut.mul(self.scale_ratio)
        if self.frelu is not None:
            out = self.frelu(out)

        return out


class InceptionDenseBlock(nn.Module):
    r""" Inception dense network.
    """

    def __init__(self, channels: int, growth_channels: int = 32, scale_ratio: float = 0.2) -> None:
        r""" Modules introduced in InceptionV4 paper.

        Args:
            channels (int): Number of channels in the input image.
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32).
            scale_ratio (float): Residual channel scaling column. (Default: 0.2)
        """
        super(InceptionDenseBlock, self).__init__()
        self.IB1 = InceptionBlock(channels + 0 * growth_channels, growth_channels)
        self.IB2 = InceptionBlock(channels + 1 * growth_channels, growth_channels)
        self.IB3 = InceptionBlock(channels + 2 * growth_channels, growth_channels)
        self.IB4 = InceptionBlock(channels + 3 * growth_channels, growth_channels)
        self.IB5 = InceptionBlock(channels + 4 * growth_channels, channels, non_linearity=False)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ib1 = self.IB1(input)
        ib2 = self.IB2(torch.cat((input, ib1), dim=1))
        ib3 = self.IB3(torch.cat((input, ib1, ib2), dim=1))
        ib4 = self.IB4(torch.cat((input, ib1, ib2, ib3), dim=1))
        ib5 = self.IB5(torch.cat((input, ib1, ib2, ib3, ib4), dim=1))

        return ib5.mul(self.scale_ratio) + input


class BioNet(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        super(BioNet, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.trunk_a = nn.Sequential(
            SymmetricBlock(64),
            DepthwiseBlock(64),
            InceptionDenseBlock(64),
            DepthwiseBlock(64),
            SymmetricBlock(64)
        )

        self.trunk_b = nn.Sequential(
            InceptionDenseBlock(64),
            DepthwiseBlock(64),
            SymmetricBlock(64),
            DepthwiseBlock(64),
            InceptionDenseBlock(64),
        )

        self.bionet = InceptionBlock(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                SymmetricBlock(64),
                Conv(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                SymmetricBlock(64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next conv layer
        self.conv2 = Conv(64, 64, kernel_size=3, stride=1, padding=1)

        # Final output layer.
        self.conv3 = Conv(64, 3, kernel_size=3, stride=1, padding=1, act=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # First conv layer.
        conv1 = self.conv1(input)

        # U-Net+MobileNet+IDB trunk.
        trunk_a = self.trunk_a(conv1)
        # Concat conv1 and trunk a.
        out1 = torch.add(conv1, trunk_a)

        # Inception+MobileNet+IDB trunk.
        trunk_b = self.trunk_b(out1)
        # Concat conv1 and trunk b.
        out2 = torch.add(conv1, trunk_b)

        # InceptionX layer.
        bionet = self.bionet(out2)
        # Concat conv1 and bionet layer.
        out = torch.add(conv1, bionet)

        # Upsampling layers.
        out = self.upsampling(out)
        # Next conv layer.
        out = self.conv2(out)
        # Final output layer.
        out = self.conv3(out)

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
