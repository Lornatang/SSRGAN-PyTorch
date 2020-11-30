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

__all__ = ["auto_padding", "dw_conv", "channel_shuffle", "Conv"]


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
    r""" Standard convolution

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

    def __init__(self, i, o, kernel_size=1, stride=1, padding=None, dilation=1, groups=1, act=True) -> None:
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(i, o, kernel_size, stride, auto_padding(kernel_size, padding), dilation=dilation,
                              groups=groups, bias=False)
        self.act = FReLU(o) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))
