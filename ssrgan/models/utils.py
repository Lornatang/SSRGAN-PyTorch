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

__all__ = ["conv1x1", "conv3x3", "conv5x5", "channel_shuffle"]


def conv1x1(i, o, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False):
    return nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                     bias=bias)


def conv3x3(i, o, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
    if i == o and groups != 1:
        groups = i
    return nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                     bias=bias)


def conv5x5(i, o, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, bias=False):
    if i == o and groups != 1:
        groups = i
    return nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                     bias=bias)


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
