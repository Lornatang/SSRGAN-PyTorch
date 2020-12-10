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
"""General convolution layer"""
import torch
import torch.nn as nn

from ssrgan.activation import HSigmoid
from ssrgan.activation import Mish

__all__ = ["channel_shuffle", "SqueezeExcite",
           "GhostConv", "GhostBottleneck",
           "SPConv"]


# Source from `https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py`
def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    r""" Random shuffle channel.

    Args:
        x (torch.Tensor): PyTorch format data stream.
        groups (int): Number of blocked connections from input channels to output channels.

    Examples:
        >>> x = torch.randn(1, 64, 128, 128)
        >>> out = channel_shuffle(x, 4)
    """
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class SqueezeExcite(nn.Module):
    r""" Squeeze-and-Excite module.

    `"MnasNet: Platform-Aware Neural Architecture Search for Mobile" <https://arxiv.org/pdf/1807.11626.pdf>`_

    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        r""" Modules introduced in MnasNet paper.
        Args:
            channels (int): Number of channels in the input image.
            reduction (optional, int): Reduce the number of channels by several times. (Default: 4).
        """
        super(SqueezeExcite, self).__init__()
        reduce_channels = int(channels // reduction)
        self.HSigmoid = HSigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(channels, reduce_channels, 1, 1, 0, bias=True)
        self.Mish = Mish()
        self.conv_expand = nn.Conv2d(reduce_channels, channels, 1, 1, 0, bias=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.avgpool(input)
        out = self.conv_reduce(out)
        out = self.Mish(out)
        out = self.conv_expand(out)
        return input * self.HSigmoid(out)


class GhostConv(nn.Module):
    r""" Ghost convolution.

    `"GhostNet: More Features from Cheap Operations" <https://arxiv.org/pdf/1911.11907.pdf>`_ paper.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0,
                 dw_kernel_size: int = 5, act: bool = True):
        """

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel. (Default: 1).
            stride (optional, int or tuple): Stride of the convolution. (Default: 1).
            padding (optional, int or tuple): Zero-padding added to both sides of the input. (Default: 0).
            dw_kernel_size (int or tuple): Size of the depthwise convolving kernel. (Default: 5).
            act (optional, bool): Whether to use activation function. (Default: ``True``).
        """
        super(GhostConv, self).__init__()
        mid_channels = out_channels // 2

        # Point-wise expansion.
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding),
            Mish() if act else nn.Identity()
        )
        # Depth-wise convolution.
        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, 1, (dw_kernel_size - 1) // 2, groups=mid_channels),
            Mish() if act else nn.Identity()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.pointwise(input)
        return torch.cat([out, self.depthwise(out)], 1)


class GhostBottleneck(nn.Module):
    r""" Ghost bottleneck.

    `"GhostNet: More Features from Cheap Operations" <https://arxiv.org/pdf/1911.11907.pdf>`_ paper.
    """

    def __init__(self, in_channels: int, out_channels: int, dw_kernel_size: int = 3, stride: int = 1):
        """

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            dw_kernel_size (int or tuple): Size of the depthwise convolving kernel. (Default: 5).
            stride (optional, int or tuple): Stride of the convolution. (Default: 1).
        """
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        mid_channels = in_channels // 2
        dw_padding = (dw_kernel_size - 1) // 2

        # Shortcut layer
        if in_channels == out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel_size, stride, dw_padding, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False) if self.stride == 2 else nn.Identity()
            )
        else:
            self.shortcut = nn.Sequential()

        # Point-wise expansion.
        self.pointwise = GhostConv(in_channels, mid_channels, 1, 1, 0)

        # Depth-wise convolution.
        if self.stride == 2:
            self.depthwise = nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, stride, dw_padding,
                                       groups=mid_channels, bias=False)
        else:
            self.depthwise = nn.Identity()

        # Squeeze-and-excitation
        self.se = SqueezeExcite(mid_channels, 4)

        # Point-wise linear projection.
        self.pointwise_linear = GhostConv(mid_channels, out_channels, 1, 1, 0, act=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Nonlinear residual convolution link layer
        shortcut = self.shortcut(input)

        # 1st ghost bottleneck.
        out = self.pointwise(input)

        # Depth-wise convolution.
        if self.stride == 2:
            out = self.depthwise(out)

        # Squeeze-and-excitation
        out = self.se(out)

        # 2nd ghost bottleneck.
        out = self.pointwise_linear(out)

        return out + shortcut


class SPConv(nn.Module):
    r""" Split convolution.

    `"Split to Be Slim: An Overlooked Redundancy in Vanilla Convolution" <https://arxiv.org/pdf/2006.12085.pdf>`_ paper.
    """

    def __init__(self, in_channels, out_channels, stride=1, scale_ratio=2):
        """
        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            stride (optional, int or tuple): Stride of the convolution. (Default: 1).
            scale_ratio (optional, int): Channel number scaling size. (Default: 2).
        """
        super(SPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.scale_ratio = scale_ratio

        self.in_channels_3x3 = int(self.in_channels // self.scale_ratio)
        self.in_channels_1x1 = self.in_channels - self.in_channels_3x3
        self.out_channels_3x3 = int(self.out_channels // self.scale_ratio)
        self.out_channels_1x1 = self.out_channels - self.out_channels_3x3

        self.depthwise = nn.Conv2d(self.in_channels_3x3, self.out_channels, 3, stride, 1, groups=2, bias=False)
        self.pointwise = nn.Conv2d(self.in_channels_3x3, self.out_channels, 1, 1, 0, bias=False)

        self.conv1x1 = nn.Conv2d(self.in_channels_1x1, self.out_channels, 1, 1, 0, bias=False)
        self.groups = int(1 * self.scale_ratio)
        self.avgpool_stride = nn.AvgPool2d(2, 2)
        self.avgpool_add = nn.AdaptiveAvgPool2d(1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, channel, _, _ = input.size()

        # Split conv3x3
        input_3x3 = input[:, :int(channel // self.scale_ratio), :, :]
        depthwise_out_3x3 = self.depthwise(input_3x3)
        if self.stride == 2:
            input_3x3 = self.avgpool_stride(input_3x3)
        pointwise_out_3x3 = self.pointwise(input_3x3)
        out_3x3 = depthwise_out_3x3 + pointwise_out_3x3
        out_3x3_ratio = self.avgpool_add(out_3x3).squeeze(dim=3).squeeze(dim=2)

        # Split conv1x1.
        input_1x1 = input[:, int(channel // self.scale_ratio):, :, :]
        # use avgpool first to reduce information lost.
        if self.stride == 2:
            input_1x1 = self.avgpool_stride(input_1x1)
        out_1x1 = self.conv1x1(input_1x1)
        out_1x1_ratio = self.avgpool_add(out_1x1).squeeze(dim=3).squeeze(dim=2)

        out_31_ratio = torch.stack((out_3x3_ratio, out_1x1_ratio), 2)
        out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)

        out_3x3 = out_3x3 * (out_31_ratio[:, :, 0].view(batch_size, self.out_channels, 1, 1).expand_as(out_3x3))
        out_1x1 = out_1x1 * (out_31_ratio[:, :, 1].view(batch_size, self.out_channels, 1, 1).expand_as(out_1x1))

        return out_3x3 + out_1x1
