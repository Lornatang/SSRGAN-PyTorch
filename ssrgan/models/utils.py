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
import math

import torch
import torch.nn as nn

from ssrgan.activation import FReLU

__all__ = ["auto_padding", "dw_conv", "channel_shuffle", "Conv", "SPConv"]


# Reference from `https://github.com/ultralytics/yolov5/blob/master/models/common.py`
def auto_padding(kernel_size, padding=None):  # kernel, padding
    r""" Edge filling 0 operation.

    Args:
        kernel_size (int or tuple): Size of the convolving kernel.
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: ``None``.
    """
    # Pad to 'same'.
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]  # auto-pad
    return padding


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


def dw_conv(i, o, kernel_size=1, stride=1, padding=None, dilation=1, act=True):
    r""" Depthwise convolution

    Args:
        i (int): Number of channels in the input image.
        o (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel. (Default: 1).
        stride (optional, int or tuple): Stride of the convolution. (Default: 1).
        padding (optional, int or tuple): Zero-padding added to both sides of
            the input. Default: ``None``.
        dilation (int or tuple, optional): Spacing between kernel elements. (Default: 1).
        act (optional, bool): Whether to use activation function. (Default: ``True``).
    """
    return Conv(i, o, kernel_size, stride, padding, dilation, groups=math.gcd(i, o), act=act)


class Conv(nn.Module):
    r""" Standard convolution.
    """

    def __init__(self, i, o, kernel_size=1, stride=1, padding=None, dilation=1, groups=1, act=True) -> None:
        """
        Args:
            i (int): Number of channels in the input image.
            o (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel. (Default: 1).
            stride (optional, int or tuple): Stride of the convolution. (Default: 1).
            padding (optional, int or tuple): Zero-padding added to both sides of
                the input. Default: ``None``.
            dilation (int or tuple, optional): Spacing between kernel elements. (Default: 1).
            groups (optional, int): Number of blocked connections from input channels to output channels. (Default: 1).
            act (optional, bool): Whether to use activation function. (Default: ``True``).
        """
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(i, o, kernel_size, stride, auto_padding(kernel_size, padding), dilation=dilation,
                              groups=groups, bias=False)
        self.act = FReLU(o) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class SPConv(nn.Module):
    r""" Split convolution.
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
        self.i = in_channels
        self.o = out_channels
        self.stride = stride
        self.scale_ratio = scale_ratio

        self.i_3x3 = int(self.i // self.scale_ratio)
        self.o_3x3 = int(self.o // self.scale_ratio)
        self.i_1x1 = self.i - self.i_3x3
        self.o_1x1 = self.o - self.o_3x3

        self.depthwise_conv = nn.Conv2d(self.i_3x3, self.o, 3, self.stride, 1, groups=2, bias=False)
        self.pointwise_conv = nn.Conv2d(self.i_3x3, self.o, 1, 1, 0, bias=False)

        self.conv1x1 = nn.Conv2d(self.i_1x1, self.o, kernel_size=1)
        self.groups = int(1 * self.scale_ratio)
        self.avgpool_stride = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avgpool_add = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channel, _, _ = x.size()

        # Split conv3x3
        x_3x3 = x[:, :int(channel // self.scale_ratio), :, :]
        depthwise_out_3x3 = self.depthwise_conv(x_3x3)
        if self.stride == 2:
            x_3x3 = self.avgpool_stride(x_3x3)
        pointwise_out_3x3 = self.pointwise_conv(x_3x3)
        out_3x3 = depthwise_out_3x3 + pointwise_out_3x3
        out_3x3_ratio = self.avgpool_add(out_3x3).squeeze(dim=3).squeeze(dim=2)

        # Split conv1x1.
        x_1x1 = x[:, int(channel // self.scale_ratio):, :, :]
        # use avgpool first to reduce information lost.
        if self.stride == 2:
            x_1x1 = self.avgpool_stride(x_1x1)
        out_1x1 = self.conv1x1(x_1x1)
        out_1x1_ratio = self.avgpool_add(out_1x1).squeeze(dim=3).squeeze(dim=2)

        out_31_ratio = torch.stack((out_3x3_ratio, out_1x1_ratio), 2)
        out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)
        out_3x3 = out_3x3 * (out_31_ratio[:, :, 0].view(batch_size, self.o, 1, 1).expand_as(out_3x3))
        out_1x1 = out_1x1 * (out_31_ratio[:, :, 1].view(batch_size, self.o, 1, 1).expand_as(out_1x1))
        out = out_3x3 + out_1x1

        return out
