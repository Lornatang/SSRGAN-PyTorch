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

from ssrgan.models.utils import Conv
from ssrgan.models.utils import dw_conv

__all__ = [
    "InvertedResidual", "MobileNetV2", "mobilenetv2"
]

model_urls = {
    "mobilenetv2": ""
}


class InvertedResidual(nn.Module):
    r""" Improved convolution method based on MobileNet-v2 version.

    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_ paper.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, expand_factor: int = 6) -> None:
        r""" Modules introduced in MobileNetV2 paper.
        Args:
            in_channels (int): Number of channels in the input image. (Default: 64).
            out_channels (int): Number of channels produced by the convolution. (Default: 64).
            expand_factor (optional, int): Number of channels produced by the expand convolution. (Default: 6).
        """
        super(InvertedResidual, self).__init__()
        hidden_channels = int(round(in_channels * expand_factor))

        # pw
        self.pointwise = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)

        # dw
        self.depthwise = dw_conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)

        # pw-linear
        self.pointwise_linear = Conv(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, act=False)

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


class MobileNetV2(nn.Module):
    r""" It is mainly based on the MobileNetV1 network as the backbone network generator"""

    def __init__(self, upscale_factor: int = 4) -> None:
        r""" This is made up of SRGAN network structure."""
        super(MobileNetV2, self).__init__()
        num_upsample_block = int(math.log(upscale_factor, 4))

        # First layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        # Twenty-three structures similar to InvertedResidual network.
        trunk = []
        for _ in range(23):
            trunk.append(InvertedResidual(64, 64))
        self.trunk = nn.Sequential(*trunk)

        self.mobilenet = InvertedResidual(64, 64)

        # Upsampling layers
        upsampling = []
        for _ in range(num_upsample_block):
            upsampling += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                InvertedResidual(64, 64),
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=2),
                InvertedResidual(64, 64)
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Next conv layer
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # First conv layer.
        conv1 = self.conv1(input)

        # MobileNet trunk.
        trunk = self.trunk(conv1)
        # Concat conv1 and mobilenet trunk.
        out = torch.add(conv1, trunk)

        # MobileNet layer.
        mobilenet = self.mobilenet(out)
        # Concat conv1 and mobilenet layer.
        out = torch.add(conv1, mobilenet)

        # Upsampling layers.
        out = self.upsampling(out)
        # Next conv layer.
        out = self.conv2(out)
        # Final output layer.
        out = self.conv3(out)

        return torch.tanh(out)


def mobilenetv2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    r"""MobileNetV2 model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1801.04381>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["mobilenetv2"], progress=progress)
        model.load_state_dict(state_dict)
    return model
