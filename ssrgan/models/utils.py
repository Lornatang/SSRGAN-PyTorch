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
import torch.nn as nn

__all__ = ["conv1x1", "conv3x3", "conv5x5"]


def conv1x1(i, o, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
    return nn.Conv2d(i, o, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)


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
