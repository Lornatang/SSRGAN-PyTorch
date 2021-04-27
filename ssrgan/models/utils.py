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

__all__ = [
    "channel_shuffle",  # ShuffleNet
    "SqueezeExcitation",  # SENet
    "GhostModule", "GhostBottleneck",  # GhostNet
    "SPConv",  # SPConv
]


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


# Source code reference from `https://github.com/hujie-frank/SENet`.
class SqueezeExcitation(nn.Module):
    r""" PyTorch implementation Squeeze-and-Excite module.

    `"Squeeze-and-Excitation Networks" <https://arxiv.org/pdf/1709.01507v4.pdf>` paper.
    """

    def __init__(self, channels: int, squeeze_factor: int = 4) -> None:
        r""" Modules introduced in SENet paper.

        Args:
            channels (int): Number of channels in the input image.
            squeeze_factor (int): Channel compression ratio. (Default: 4)
        """
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = int(channels // squeeze_factor)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeeze_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, channels, kernel_size=1, stride=1, padding=0)
        self.hard_sigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.global_pooling(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.hard_sigmoid(out)

        out = out.mul(x)

        return out


# TODO: implementation GhostBottleneck module.
# Source code reference from `https://github.com/huawei-noah/CV-backbones/blob/master/ghostnet_pytorch/ghostnet.py`.
class GhostModule(nn.Module):
    pass


# TODO: implementation GhostBottleneck module.
# Source code reference from `https://github.com/huawei-noah/CV-backbones/blob/master/ghostnet_pytorch/ghostnet.py`.
class GhostBottleneck(nn.Module):
    pass


# Source code reference from `https://github.com/ruinmessi/RFBNet/blob/master/models/RFB_Net_vgg.py`.
class ReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64):
        r""" Modules introduced in RFBNet paper.
        Args:
            in_channels (int): Number of channels in the input image. (Default: 64)
            out_channels (int): Number of channels produced by the convolution. (Default: 64)
        """
        super(ReceptiveFieldBlock, self).__init__()
        branch_channels = in_channels // 4

        # shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, 1, 3, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels // 2, (branch_channels // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d((branch_channels // 4) * 3, branch_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=5, dilation=5)
        )

        self.conv_linear = nn.Conv2d(4 * branch_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        out = self.conv_linear(out)

        out = self.leaky_relu(torch.add(out * 0.1, shortcut))

        return out


# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ReceptiveFieldDenseBlock(nn.Module):
    r""" Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
        RFB-SSD proposed Receptive Fields Block (RFB) for object detection
    """

    def __init__(self, channels: int = 64, growth_channels: int = 32):
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        """
        super(ReceptiveFieldDenseBlock, self).__init__()
        self.rfb1 = nn.Sequential(
            ReceptiveFieldBlock(channels + 0 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb2 = nn.Sequential(
            ReceptiveFieldBlock(channels + 1 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb3 = nn.Sequential(
            ReceptiveFieldBlock(channels + 2 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb4 = nn.Sequential(
            ReceptiveFieldBlock(channels + 3 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb5 = ReceptiveFieldBlock(channels + 4 * growth_channels, channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rfb1 = self.rfb1(x)
        rfb2 = self.rfb2(torch.cat((x, rfb1), dim=1))
        rfb3 = self.rfb3(torch.cat((x, rfb1, rfb2), dim=1))
        rfb4 = self.rfb4(torch.cat((x, rfb1, rfb2, rfb3), dim=1))
        rfb5 = self.rfb5(torch.cat((x, rfb1, rfb2, rfb3, rfb4), dim=1))

        out = torch.add(rfb5 * 0.1, x)

        return out


# Source code reference from `https://arxiv.org/pdf/2005.12597.pdf`.
class ResidualOfReceptiveFieldDenseBlock(nn.Module):
    r"""The residual block structure of traditional RFB-ESRGAN is defined"""

    def __init__(self, channels: int = 64, growth_channels: int = 32):
        r"""
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        """
        super(ResidualOfReceptiveFieldDenseBlock, self).__init__()
        self.RFDB1 = ReceptiveFieldDenseBlock(channels, growth_channels)
        self.RFDB2 = ReceptiveFieldDenseBlock(channels, growth_channels)
        self.RFDB3 = ReceptiveFieldDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RFDB1(x)
        out = self.RFDB1(out)
        out = self.RFDB1(out)

        out = torch.add(out * 0.1, x)

        return out


# Source code reference from `https://arxiv.org/pdf/2006.12085.pdf`.
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

        out = torch.add(out_3x3, out_1x1)

        return out
