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
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from ssrgan.models import UNet
from ssrgan.utils import select_device


# 中间特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


if __name__ == "__main__":
    # get device
    device = select_device("0")
    # feature = FeatureVisualization("data/4x/train/input/1_1_1.bmp", 0)
    # 插入维度
    img = Image.open("lr.bmp")
    input = transforms.ToTensor()(img)
    input = input.unsqueeze(0)
    # Convert to define device.
    input = input.to(device)

    net = UNet().to(device)
    exact_list = ['conv3']
    myexactor = FeatureExtractor(net, exact_list)
    x = myexactor(input)

    plt.figure(figsize=(25, 25))
    for i in range(3):
        ax = plt.subplot(2, 2, i + 1)
        ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        ax.set_title(f'Channel {i}')

        plt.imshow(x[0].data.cpu().numpy()[0, i, :, :], cmap='jet')

    # plt.show()  # 图像每次都不一样，是因为模型每次都需要前向传播一次，不是加载的与训练模型
    plt.savefig("6.png")
