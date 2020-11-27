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
from ssrgan.models.mobilenetv3 import SEModule
from .utils import conv1x1
from .utils import conv3x3
from .utils import conv5x5

__all__ = ["SymmetricBlock", "SymmetricDenseBlock",
           "DepthwiseBlock", "DepthwiseDenseBlock",
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

    def __init__(self, in_channels: int, out_channels: int, expand_factor=0.5, scale_ratio: float = 0.5,
                 non_linearity: bool = True) -> None:
        r""" Modules introduced in U-Net paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, float): Number of channels produced by the expand convolution. (Default: 0.5).
            scale_ratio (optional, float): Residual channel scaling column. (Default: 0.5).
            non_linearity (optional, bool): Does the last layer use nonlinear activation. (Default: ``True``).
        """
        super(SymmetricBlock, self).__init__()
        hidden_channels = int(out_channels * expand_factor)

        # shortcut layer
        self.shortcut = conv1x1(in_channels, out_channels)

        # Down sampling.
        self.down = nn.Sequential(
            conv3x3(in_channels, in_channels, stride=2, padding=1, dilation=1),
            FReLU(in_channels),
            conv1x1(in_channels, in_channels),
            FReLU(in_channels),

            # Residual block.
            conv3x3(in_channels, in_channels, padding=3, dilation=3),
            FReLU(in_channels),
            conv1x1(in_channels, hidden_channels),
            FReLU(hidden_channels),

            conv3x3(hidden_channels, hidden_channels, padding=5, dilation=5),
            FReLU(hidden_channels),
            conv1x1(hidden_channels, in_channels),
            FReLU(in_channels)
        )

        # Up sampling.
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),

            # Residual block.
            conv3x3(in_channels, in_channels, padding=3, dilation=3),
            FReLU(in_channels),
            conv1x1(in_channels, hidden_channels),
            FReLU(hidden_channels),

            conv3x3(hidden_channels, hidden_channels, padding=5, dilation=5),
            FReLU(hidden_channels),
            conv1x1(hidden_channels, out_channels)
        )

        self.frelu = FReLU(out_channels) if non_linearity else None
        self.scale_ratio = scale_ratio

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

        # Out and input fusion.
        out = out + shortcut.mul(self.scale_ratio)
        if self.frelu is not None:
            out = self.frelu(out)

        return out


class SymmetricDenseBlock(nn.Module):
    r""" U-shaped dense network.
    """

    def __init__(self, in_channels: int, out_channels: int, expand_factor=0.5, scale_ratio: float = 0.2) -> None:
        r""" Modules introduced in U-Net paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, float): Number of channels produced by the expand convolution. (Default: 0.5).
            scale_ratio (optional, float): Residual channel scaling column. (Default: 0.2).
        """
        super(SymmetricDenseBlock, self).__init__()
        hidden_channels = int(out_channels * expand_factor)
        self.SB1 = SymmetricBlock(in_channels + 0 * hidden_channels, hidden_channels, expand_factor)
        self.SB2 = SymmetricBlock(in_channels + 1 * hidden_channels, hidden_channels, expand_factor)
        self.SB3 = SymmetricBlock(in_channels + 2 * hidden_channels, hidden_channels, expand_factor)
        self.SB4 = SymmetricBlock(in_channels + 3 * hidden_channels, hidden_channels, expand_factor)
        self.SB5 = SymmetricBlock(in_channels + 4 * hidden_channels, out_channels, expand_factor, False)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        sb1 = self.SB1(input)
        sb2 = self.SB2(torch.cat((input, sb1), dim=1))
        sb3 = self.SB3(torch.cat((input, sb1, sb2), dim=1))
        sb4 = self.SB4(torch.cat((input, sb1, sb2, sb3), dim=1))
        sb5 = self.SB5(torch.cat((input, sb1, sb2, sb3, sb4), dim=1))

        return sb5.mul(self.scale_ratio) + input


class DepthwiseBlock(nn.Module):
    r""" Base on MobileNetV2 + Squeeze-and-Excite.

    `"Searching for MobileNetV3" <https://arxiv.org/pdf/1905.02244.pdf>`_ paper.

    """

    def __init__(self, in_channels: int, out_channels: int, expand_factor=0.5, scale_ratio: float = 0.5,
                 non_linearity: bool = True) -> None:
        r""" Modules introduced in MobileNetV3 paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, float): Number of channels produced by the expand convolution. (Default: 0.5).
            scale_ratio (optional, float): Residual channel scaling column. (Default: 0.5).
            non_linearity (optional, bool): Does the last layer use nonlinear activation. (Default: ``True``).
        """
        super(DepthwiseBlock, self).__init__()
        hidden_channels = int(out_channels * expand_factor)

        self.shortcut = conv1x1(in_channels, out_channels)

        # pw
        self.pointwise = nn.Sequential(
            conv1x1(in_channels, hidden_channels),
            FReLU(hidden_channels)
        )

        # dw
        self.depthwise = conv5x5(hidden_channels, hidden_channels, groups=hidden_channels)

        # squeeze and excitation module.
        self.SEModule = nn.Sequential(
            SEModule(hidden_channels),
            FReLU(hidden_channels)
        )

        # pw-linear
        self.pointwise_linear = conv1x1(hidden_channels, out_channels)

        self.frelu = FReLU(out_channels) if non_linearity else None
        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1x1 nonlinear characteristic output.
        shortcut = self.shortcut(input)

        # Expansion convolution
        out = self.pointwise(input)
        # DepthWise convolution
        out = self.depthwise(out)
        # Squeeze-and-Excite
        out = self.SEModule(out)
        # Projection convolution
        out = self.pointwise_linear(out)

        # Out and input fusion.
        out = out + shortcut.mul(self.scale_ratio)
        if self.frelu is not None:
            out = self.frelu(out)

        return out


class DepthwiseDenseBlock(nn.Module):
    r""" MobileNetV3 dense network.
    """

    def __init__(self, in_channels: int, out_channels: int, expand_factor=0.5, scale_ratio: float = 0.2) -> None:
        r""" Modules introduced in MobileNetV3 paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, float): Number of channels produced by the expand convolution. (Default: 0.5).
            scale_ratio (optional, float): Residual channel scaling column. (Default: 0.2).
        """
        super(DepthwiseDenseBlock, self).__init__()
        hidden_channels = int(out_channels * expand_factor)
        self.DB1 = DepthwiseBlock(in_channels + 0 * hidden_channels, hidden_channels, expand_factor)
        self.DB2 = DepthwiseBlock(in_channels + 1 * hidden_channels, hidden_channels, expand_factor)
        self.DB3 = DepthwiseBlock(in_channels + 2 * hidden_channels, hidden_channels, expand_factor)
        self.DB4 = DepthwiseBlock(in_channels + 3 * hidden_channels, hidden_channels, expand_factor)
        self.DB5 = DepthwiseBlock(in_channels + 4 * hidden_channels, out_channels, expand_factor, False)

        self.scale_ratio = scale_ratio

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        db1 = self.DB1(input)
        db2 = self.DB2(torch.cat((input, db1), dim=1))
        db3 = self.DB3(torch.cat((input, db1, db2), dim=1))
        db4 = self.DB4(torch.cat((input, db1, db2, db3), dim=1))
        db5 = self.DB5(torch.cat((input, db1, db2, db3, db4), dim=1))

        return db5.mul(self.scale_ratio) + input


class InceptionBlock(nn.Module):
    r""" Base on InceptionV4

    `"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    " <https://arxiv.org/pdf/1602.07261.pdf>`_ paper.

    """

    def __init__(self, in_channels: int, out_channels: int, expand_factor=0.25, scale_ratio: float = 0.5,
                 non_linearity: bool = True) -> None:
        r""" Modules introduced in InceptionX paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, float): Number of channels produced by the expand convolution. (Default: 0.25).
            scale_ratio (optional, float): Residual channel scaling column. (Default: 0.5).
            non_linearity (optional, bool): Does the last layer use nonlinear activation. (Default: ``True``).
        """
        super(InceptionBlock, self).__init__()
        branch_features = int(in_channels * expand_factor)

        self.shortcut = conv1x1(in_channels, out_channels)

        # Squeeze style layer
        self.branch1_1 = nn.Sequential(
            conv1x1(in_channels, branch_features // 4),
            FReLU(branch_features // 4)
        )
        self.branch1_2 = nn.Sequential(
            conv1x1(branch_features // 4, branch_features // 2),
            FReLU(branch_features // 2)
        )
        self.branch1_3 = nn.Sequential(
            conv3x3(branch_features // 4, branch_features // 2),
            FReLU(branch_features // 2)
        )

        # InvertedResidual style layer
        self.branch2_1 = nn.Sequential(
            conv1x1(in_channels, branch_features * 2),
            FReLU(branch_features * 2)
        )
        self.branch2_2 = nn.Sequential(
            conv3x3(branch_features * 2, branch_features * 2),
            FReLU(branch_features * 2)
        )
        self.branch2_3 = nn.Sequential(
            conv1x1(branch_features * 2, branch_features),
            FReLU(branch_features)
        )

        # Inception style layer 1
        self.branch3 = nn.Sequential(
            conv3x3(in_channels, branch_features, kernel_size=(1, 3), padding=(0, 1)),
            FReLU(branch_features),
            conv3x3(branch_features, branch_features, kernel_size=(3, 1), padding=(1, 0)),
            FReLU(branch_features),
            conv1x1(branch_features, branch_features),
            FReLU(branch_features)
        )

        # Inception style layer 2
        self.branch4 = nn.Sequential(
            conv3x3(in_channels, branch_features, kernel_size=(3, 1), padding=(1, 0)),
            FReLU(branch_features),
            conv3x3(branch_features, branch_features, kernel_size=(1, 3), padding=(0, 1)),
            FReLU(branch_features),
            conv1x1(branch_features, branch_features),
            FReLU(branch_features)
        )

        self.branch_concat = conv1x1(branch_features * 2, branch_features * 2)

        self.conv1x1 = conv1x1(in_channels, out_channels)

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

        # Squeeze layer.
        branch1_1 = self.branch1_1(input)  # Squeeze convolution
        branch1_2 = self.branch1_2(branch1_1)  # Expand convolution 1x1
        branch1_3 = self.branch1_3(branch1_1)  # Expand convolution 3x3
        squeeze_out = torch.cat([branch1_2, branch1_3], dim=1)

        # InvertedResidual layer.
        branch2_1 = self.branch2_1(input)
        branch2_2 = self.branch2_2(branch2_1)
        mobile_out = self.branch2_3(branch2_2)

        # Concat Squeeze and InvertedResidual layer.
        branch_concat1_2 = torch.cat([squeeze_out, mobile_out], dim=1)
        branch_out1 = self.branch_concat(branch_concat1_2)

        # Inception layer
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)
        branch_concat3_4 = torch.cat([branch3, branch4], dim=1)
        branch_out2 = self.branch_concat(branch_concat3_4)

        # Concat layer
        out = torch.cat([branch_out1, branch_out2], dim=1)
        out = self.conv1x1(out)

        # Out and input fusion.
        out = out + shortcut.mul(self.scale_ratio)
        if self.frelu is not None:
            out = self.frelu(out)

        return out


class InceptionDenseBlock(nn.Module):
    r""" Inception dense network.
    """

    def __init__(self, in_channels: int, out_channels: int, expand_factor=0.25, scale_ratio: float = 0.2) -> None:
        r""" Modules introduced in InceptionV4 paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            expand_factor (optional, float): Number of channels produced by the expand convolution. (Default: 0.5).
            scale_ratio (optional, float): Residual channel scaling column. (Default: 0.2).
        """
        super(InceptionDenseBlock, self).__init__()
        hidden_channels = int(in_channels * expand_factor)
        self.IB1 = InceptionBlock(in_channels + 0 * hidden_channels, hidden_channels, expand_factor)
        self.IB2 = InceptionBlock(in_channels + 1 * hidden_channels, hidden_channels, expand_factor)
        self.IB3 = InceptionBlock(in_channels + 2 * hidden_channels, hidden_channels, expand_factor)
        self.IB4 = InceptionBlock(in_channels + 3 * hidden_channels, hidden_channels, expand_factor)
        self.IB5 = InceptionBlock(in_channels + 4 * hidden_channels, out_channels, expand_factor, False)

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
        self.conv1 = conv3x3(3, 64)

        self.trunk_a = nn.Sequential(
            SymmetricDenseBlock(64, 64),
            SymmetricDenseBlock(64, 64)
        )
        self.trunk_b = nn.Sequential(
            DepthwiseDenseBlock(64, 64),
            DepthwiseDenseBlock(64, 64)
        )
        self.trunk_c = nn.Sequential(
            InceptionDenseBlock(64, 64),
            InceptionDenseBlock(64, 64)
        )
        self.trunk_d = nn.Sequential(
            DepthwiseDenseBlock(64, 64),
            DepthwiseDenseBlock(64, 64)
        )

        self.bionet = InceptionBlock(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                InceptionBlock(64, 64),
                conv3x3(64, 64, groups=64),
                FReLU(64),
                conv1x1(64, 256),
                FReLU(256),
                nn.PixelShuffle(upscale_factor=2),
                InceptionBlock(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next conv layer
        self.conv2 = nn.Sequential(
            conv3x3(64, 64, groups=64),
            FReLU(64),
            conv1x1(64, 64),
            FReLU(64)
        )

        # Final output layer
        self.conv3 = conv3x3(64, 3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # First conv layer.
        conv1 = self.conv1(input)

        # U-Net trunk.
        trunk = self.trunk_a(conv1)
        # Concat conv1 and trunk a.
        out1 = torch.add(conv1, trunk)

        # MobileNet trunk.
        trunk_b = self.trunk_b(out1)
        # Concat conv1 and trunk b.
        out2 = torch.add(conv1, trunk_b)

        # InceptionX trunk.
        trunk_c = self.trunk_c(out2)
        # Concat conv1 and trunk-c.
        out3 = torch.add(conv1, trunk_c)

        # MobileNet trunk.
        trunk_d = self.trunk_d(out3)
        # Concat conv1 and trunk-d.
        out4 = torch.add(conv1, trunk_d)

        # InceptionX layer.
        bionet = self.bionet(out4)
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
