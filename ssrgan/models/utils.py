# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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

__all__ = ["channel_shuffle", "SqueezeExcite",
           "GhostModule", "GhostBottleneck",
           "SPConv"]


# Source code reference from `https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py`
def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    r""" PyTorch implementation ShuffleNet module.

    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" <https://arxiv.org/pdf/1807.11164v1.pdf>` paper.

    Args:
        x (torch.Tensor): PyTorch format data stream.
        groups (int): Number of blocked connections from input channels to output channels.

    Examples:
        >>> inputs = torch.randn(1, 64, 128, 128)
        >>> output = channel_shuffle(inputs, 4)
    """
    batch_size, num_channels, height, width = x.data.image_size()
    channels_per_group = num_channels // groups

    # reshape
    out = x.view(batch_size, groups, channels_per_group, height, width)

    out = torch.transpose(out, 1, 2).contiguous()

    # flatten
    out = out.view(batch_size, -1, height, width)

    return out


class SqueezeExcite(nn.Module):
    r""" PyTorch implementation Squeeze-and-Excite module.

    `"MSqueeze-and-Excitation Networks" <https://arxiv.org/pdf/1709.01507v4.pdf>` paper.
    """

    def __init__(self, channels: int, reduction: int) -> None:
        r""" Modules introduced in SENet paper.

        Args:
            channels (int): Number of channels in the input image.
            reduction (int): Reduce the number of channels by several times.
        """
        super(SqueezeExcite, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()
        out = self.global_pooling(x)
        # Squeeze layer.
        out = out.view(batch_size, channels)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        # Excite layer.
        out = out.view(batch_size, channels, 1, 1)

        return x * out.expand_as(x)


# Source code reference from `https://github.com/huawei-noah/CV-backbones/blob/master/ghostnet_pytorch/ghostnet.py`
class GhostModule(nn.Module):
    r""" PyTorch implementation GhostNet module.

    `"GhostNet: More Features from Cheap Operations" <https://arxiv.org/pdf/1911.11907v2.pdf>` paper.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dw_size: int, stride: int, ratio: int, relu: bool) -> None:
        r""" Modules introduced in GhostNet paper.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel.
            dw_size (int): Size of the depth-wise convolving kernel.
            stride (int): Stride of the convolution.
            ratio (int): Reduce the number of channels ratio.
            relu (bool): Use activation function.
        """
        super(GhostModule, self).__init__()
        self.out_channels = out_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // ratio, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels // ratio),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(out_channels // ratio, out_channels // ratio * (ratio - 1), dw_size, 1, dw_size // 2, groups=out_channels // ratio, bias=False),
            nn.BatchNorm2d(out_channels // ratio * (ratio - 1)),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.primary_conv(x)
        out2 = self.cheap_operation(out1)
        out = torch.cat([out1, out2], dim=1)
        return out[:, :self.out_channels, :, :]


# TODO: implementation GhostNet module.
# Source code reference from `https://github.com/huawei-noah/CV-backbones/blob/master/ghostnet_pytorch/ghostnet.py`
class GhostBottleneck(nn.Module):
    r""" PyTorch implementation GhostNet module.

    `"GhostNet: More Features from Cheap Operations" <https://arxiv.org/pdf/1911.11907v2.pdf>` paper.
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, dw_size: int, stride: int, reduction: int) -> None:
        r""" Modules introduced in GhostNet paper.

        Args:
            in_channels (int): Number of channels in the input image.
            mid_channels (int): Number of channels midden by the convolution.
            out_channels (int): Number of channels produced by the convolution.
            dw_size (int): Size of the depth-wise convolving kernel.
            stride (int): Stride of the convolution.
            reduction (int): Reduce the number of SE channels ratio.
        """
        super(GhostBottleneck, self).__init__()
        pass

    #         self.stride = stride
    #         mid_channels = in_channels // 2
    #         dw_padding = (dw_size - 1) // 2
    #
    #         # Shortcut layer
    #         if in_channels == out_channels:
    #             self.shortcut = nn.Sequential(
    #                 nn.Conv2d(in_channels, in_channels, dw_size, stride, dw_padding, groups=in_channels, bias=False),
    #                 nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False) if self.stride == 2 else nn.Identity()
    #             )
    #         else:
    #             self.shortcut = nn.Sequential()
    #
    #         # Point-wise expansion.
    #         self.pointwise = GhostModule(in_channels, mid_channels, 1, 1, 0)
    #
    #         # Depth-wise convolution.
    #         if self.stride == 2:
    #             self.depthwise = nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, stride, dw_padding,
    #                                        groups=mid_channels, bias=False)
    #         else:
    #             self.depthwise = nn.Identity()
    #
    #         # Squeeze-and-excitation
    #         self.se = SqueezeExcite(mid_channels, 4)
    #
    #         # Point-wise linear projection.
    #         self.pointwise_linear = GhostConv(mid_channels, out_channels, 1, 1, 0, act=False)
    #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


#         # Nonlinear residual convolution link layer
#         shortcut = self.shortcut(x)
#
#         # 1st ghost bottleneck.
#         out = self.pointwise(x)
#
#         # Depth-wise convolution.
#         if self.stride == 2:
#             out = self.depthwise(out)
#
#         # Squeeze-and-excitation
#         out = self.se(out)
#
#         # 2nd ghost bottleneck.
#         out = self.pointwise_linear(out)
#
#         return out + shortcut


class SPConv(nn.Module):
    r""" Split convolution.

    `"Split to Be Slim: An Overlooked Redundancy in Vanilla Convolution" <https://arxiv.org/pdf/2006.12085.pdf>`_ paper.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, scale_ratio: int):
        r"""

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            stride (int): Stride of the convolution.
            scale_ratio (int): Channel number scaling size.
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

        self.depthwise = nn.Conv2d(self.in_channels_3x3, self.out_channels, 3, stride, 1, groups=2)
        self.pointwise = nn.Conv2d(self.in_channels_3x3, self.out_channels, 1, 1, 0)

        self.conv1x1 = nn.Conv2d(self.in_channels_1x1, self.out_channels, 1, 1, 0)
        self.groups = int(1 * self.scale_ratio)
        self.avg_pool_stride = nn.AvgPool2d(2, 2)
        self.avg_pool_add = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()

        # Split conv3x3
        input_3x3 = x[:, :int(channels // self.scale_ratio), :, :]
        depthwise_out_3x3 = self.depthwise(input_3x3)
        if self.stride == 2:
            input_3x3 = self.avg_pool_stride(input_3x3)
        pointwise_out_3x3 = self.pointwise(input_3x3)
        out_3x3 = depthwise_out_3x3 + pointwise_out_3x3
        out_3x3_ratio = self.avg_pool_add(out_3x3).squeeze(dim=3).squeeze(dim=2)

        # Split conv1x1.
        input_1x1 = x[:, int(channels // self.scale_ratio):, :, :]
        # use avgpool first to reduce information lost.
        if self.stride == 2:
            input_1x1 = self.avg_pool_stride(input_1x1)
        out_1x1 = self.conv1x1(input_1x1)
        out_1x1_ratio = self.avg_pool_add(out_1x1).squeeze(dim=3).squeeze(dim=2)

        out_31_ratio = torch.stack((out_3x3_ratio, out_1x1_ratio), 2)
        out_31_ratio = nn.Softmax(dim=2)(out_31_ratio)

        out_3x3 = out_3x3 * (out_31_ratio[:, :, 0].view(batch_size, self.out_channels, 1, 1).expand_as(out_3x3))
        out_1x1 = out_1x1 * (out_31_ratio[:, :, 1].view(batch_size, self.out_channels, 1, 1).expand_as(out_1x1))

        return out_3x3 + out_1x1
