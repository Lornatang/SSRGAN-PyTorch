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
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

__all__ = [
    "FReLU", "HSigmoid", "HSwish"
]


class FReLU(nn.Module):
    r""" FReLU formulation. The funnel condition has a window size of kxk. (k=3 by default)"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, input: Tensor):
        out = self.conv(input)
        out = self.bn(out)
        return torch.max(input, out)


class HSigmoid(nn.Module):
    @staticmethod
    def forward(input: Tensor) -> Tensor:
        return F.relu6(input + 3, inplace=True) / 6.


class HSwish(nn.Module):
    @staticmethod
    def forward(input: Tensor) -> Tensor:
        return input * F.relu6(input + 3, inplace=True) / 6.
