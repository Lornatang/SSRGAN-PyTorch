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
from torch.hub import load_state_dict_from_url

from ssrgan.activation import Mish

model_urls = {
    "dsgan": "https://github.com/Lornatang/SSRGAN-PyTorch/releases/download/0.1.0/RRDBNet_4x4_16_DF2K-e31a1b2e.pth"
}


class Symmetric(nn.Module):
    r""" U-shaped network.

    `"U-Net: Convolutional Networks for Biomedical
    Image Segmentation" <https://arxiv.org/abs/1505.04597>`_ paper

    """

    def __init__(self, channels: int) -> None:
        r""" Modules introduced in U-Net paper.

        Args:
            channels (int): Number of channels in the input image.
        """
        super(Symmetric, self).__init__()

        # Down sampling.
        self.down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            Mish(),
            nn.Conv2d(channels, channels // 2, kernel_size=1, stride=1, padding=0),
            Mish(),
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, stride=1, padding=1, groups=channels // 2),
            Mish(),
            nn.Conv2d(channels // 2, channels // 4, kernel_size=1, stride=1, padding=0)
        )

        # Up sampling.
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(channels // 4, channels // 4, kernel_size=3, stride=1, padding=1, groups=channels // 4),
            Mish(),
            nn.Conv2d(channels // 4, channels // 2, kernel_size=1, stride=1, padding=0),
            Mish(),
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, stride=1, padding=1, groups=channels // 2),
            Mish(),
            nn.Conv2d(channels // 2, channels, kernel_size=1, stride=1, padding=0)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Down sampling.
        out = self.down(x)
        # Up sampling.
        out = self.up(out)

        return out + x


class DepthWise(nn.Module):
    r""" Improved convolution method based on MobileNet-v2 version.

    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_ paper.
    """

    def __init__(self, channels: int) -> None:
        r""" Modules introduced in MobileNetV2 paper.

        Args:
            channels (int): Number of channels in the input image.
        """
        super(DepthWise, self).__init__()

        # pw
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            Mish(),
        )

        # dw
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            Mish(),
        )

        # pw-linear
        self.pointwise_linear = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expansion convolution
        out = self.pointwise(x)
        # DepthWise convolution
        out = self.depthwise(out)
        # Projection convolution
        out = self.pointwise_linear(out)

        return out + x


class InceptionX(nn.Module):
    r""" Base on InceptionV4

    `"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    " <https://arxiv.org/pdf/1602.07261.pdf>`_ paper.
    """

    def __init__(self, channels: int) -> None:
        r""" Modules introduced in InceptionX paper.

        Args:
            channels (int): Number of channels in the input image.
        """
        super(InceptionX, self).__init__()
        branch_features = int(channels // 4)

        self.shortcut = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, branch_features, kernel_size=1, stride=1, padding=0),
            Mish(),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=1, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, branch_features, kernel_size=1, stride=1, padding=0),
            Mish(),
            nn.Conv2d(branch_features, branch_features, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            Mish(),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=1, padding=3, dilation=3),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, branch_features, kernel_size=1, stride=1, padding=0),
            Mish(),
            nn.Conv2d(branch_features, branch_features, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            Mish(),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=1, padding=3, dilation=3),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(channels, branch_features, kernel_size=1, stride=1, padding=0),
            Mish(),
            nn.Conv2d(branch_features, branch_features, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            Mish(),
            nn.Conv2d(branch_features, branch_features, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            Mish(),
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=1, padding=3, dilation=3),
        )

        self.conv1x1 = nn.Conv2d(branch_features * 4, branch_features * 4, kernel_size=1, stride=1, padding=0)

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

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.conv1x1(out)

        return out + shortcut


class Generator(nn.Module):
    r""" It is mainly based on the mobile net network as the backbone network generator"""

    def __init__(self) -> None:
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        self.trunk_a = nn.Sequential(
            DepthWise(32),
            DepthWise(32)
        )
        self.trunk_b = nn.Sequential(
            Symmetric(32),
            Symmetric(32)
        )
        self.trunk_c = nn.Sequential(
            DepthWise(32),
            DepthWise(32)
        )
        self.trunk_d = nn.Sequential(
            InceptionX(32),
            InceptionX(32)
        )

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        self.upsampling = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            Symmetric(32),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            Mish(),
            nn.PixelShuffle(upscale_factor=2),
            Symmetric(32)
        )

        # Next conv layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            Mish()
        )

        # Final output layer.
        self.conv4 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv layer.
        conv1 = self.conv1(x)

        # MobileNet trunk.
        trunk_a = self.trunk_a(conv1)
        # Concat conv1 and trunk a.
        out1 = torch.add(conv1, trunk_a)

        # Symmetric trunk.
        trunk_b = self.trunk_b(out1)
        # Concat conv1 and trunk b.
        out2 = torch.add(conv1, trunk_b)

        # MobileNet trunk.
        trunk_c = self.trunk_c(out2)
        # Concat conv1 and trunk c.
        out3 = torch.add(conv1, trunk_c)

        # Inception trunk.
        trunk_d = self.trunk_b(out3)
        # Concat conv1 and trunk d.
        out4 = torch.add(conv1, trunk_d)

        # conv2 layer.
        conv2 = self.conv2(out4)
        # Concat conv1 and conv2 layer.
        out = torch.add(conv1, conv2)

        # Upsampling layers.
        out = self.upsampling(out)
        # Next conv layer.
        out = self.conv3(out)
        # Final output layer.
        out = self.conv4(out)

        return torch.tanh(out)


def _gan(arch, pretrained, progress):
    model = Generator()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def dsgan(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/2021.00000>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("dsgan", pretrained, progress)
