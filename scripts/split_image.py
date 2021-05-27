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
import os

from PIL import Image


def cut_image(image):
    width, height = image.size
    item_width = int(width / 9)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 9):
        for j in range(0, 9):
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
            box_list.append(box)

    image_list = [image.crop(box) for box in box_list]
    return image_list


def save_images(raw_filename, image_list):
    index = 1
    for image in image_list:
        image.save(raw_filename.split(".")[0] + "_" + str(index) + ".png")
        index += 1


if __name__ == "__main__":
    DIR = "target"
    files = os.listdir(DIR)
    for file in files:
        print(f"Process: `{os.path.join(DIR, file)}`.")
        image = Image.open(os.path.join(DIR, file))
        image_list = cut_image(image)
        save_images(os.path.join(DIR, file), image_list)
        os.remove(os.path.join(DIR, file))
