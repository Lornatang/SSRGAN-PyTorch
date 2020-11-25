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
import os

import torch.utils.data.dataset
import torchvision.transforms as transforms
from PIL import Image

__all__ = [
    "BaseDataset", "CustomDataset", "check_image_file"
]


class BaseDataset(torch.utils.data.dataset.Dataset):
    """An abstract class representing a :class:`Dataset`."""

    def __init__(self, dir_path):
        """
        Args:
            dir_path (str): The directory address where the data image is stored.
        """
        super(BaseDataset, self).__init__()
        self.filenames = [os.path.join(dir_path, x) for x in os.listdir(dir_path) if check_image_file(x)]

        self.input_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((54, 54), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.target_transforms = transforms.Compose([
            transforms.RandomCrop((216, 216)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        target = self.target_transforms(Image.open(self.filenames[index]))
        input = self.input_transforms(target)

        return input, target

    def __len__(self):
        return len(self.filenames)


class CustomDataset(torch.utils.data.dataset.Dataset):
    r"""An abstract class representing a :class:`Dataset`."""

    def __init__(self, input_dir, target_dir):
        """

        Args:
            input_dir (str): The directory address where the data image is stored.
            target_dir (str): The directory address where the target image is stored.
        """
        super(CustomDataset, self).__init__()
        self.input_filenames = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if check_image_file(x)]
        self.target_filenames = [os.path.join(target_dir, x) for x in os.listdir(target_dir) if check_image_file(x)]
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        r""" Get image source file.

        Args:
            index (int): Index position in image list.

        Returns:
            Low resolution image, high resolution image.
        """
        input = self.transforms(Image.open(self.input_filenames[index]))
        target = self.transforms(Image.open(self.target_filenames[index]))

        return input, target

    def __len__(self):
        return len(self.input_filenames)


def check_image_file(filename):
    r"""Filter non image files in directory.

    Args:
        filename (str): File name under path.

    Returns:
        Return True if bool(x) is True for any x in the iterable.
    """
    return any(filename.endswith(extension) for extension in [".jpg", ".JPG",
                                                              ".jpeg", ".jpeg",
                                                              ".png", ".PNG",
                                                              ".bmp", ".BMP"])
