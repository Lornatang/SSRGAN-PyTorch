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
import logging
import os
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from ssrgan.dataset import check_image_file
from ssrgan.utils import select_device
from torchvision import transforms

from model import bionet

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

# Image resolution. [width, height, width block num, height block num]
resolution_dict = {
    "240p": [320, 240, 4, 3],
    "360p": [480, 360, 8, 6],
    "720p": [1080, 720, 12, 9],
    "1080p": [1920, 1080, 2, 2],  # Only support it is.
    "2k": [2560, 1440, 20, 15],
    "4k": [4096, 2160, 32, 20],
    "8k": [7680, 4320, 60, 30]
}


class Inference(object):
    def __init__(self, args, resolution_ratio: str = "1080p"):
        self.file_path = args.input
        self.model_path = args.model
        self.device_id = args.device
        self.resolution_ratio = resolution_ratio
        self.over_length = 128  # Edge overlap length.
        self.ratio = 0.05  # Fusion calculation weight parameters.

        logger.info(f"Inference engine information:\n"
                    f"\tImage path is `{os.getcwd()}/{self.file_path}`\n"
                    f"\tModel path is `{os.getcwd()}/{self.model_path}`\n"
                    f"\tDevice id is `{self.device_id}`\n"
                    f"\tResolution ratio is `{self.resolution_ratio}`\n"
                    f"\tEdge overlap length is {int(self.over_length)}\n"
                    f"\tEdge overlap length is {float(self.ratio)}")

        # Model of configuration super-resolution algorithm.
        self.device = select_device(self.device_id)
        logger.info(f"Creating model...")
        self.model = bionet().to(self.device)
        logger.info(f"Loading model weights...")
        self.model.load_state_dict(torch.load(self.model_path))
        logger.info(f"Set model to eval mode.")
        self.model.eval()

    def inference(self, model: torch.nn.Module, device: torch.device):
        r""" Super-resolution of low resolution image.

        Args:
            model (torch.nn.Module): Base class for all neural network modules..
            device (torch.device): CPU or GPU.
        """
        # Step 1: Read image.
        image = Image.open(self.file_path)

        # Step 2: Check whether the image resolution meets the standard.
        width, height = image.image_size
        correct_width, correct_height, width_blocks, height_blocks = resolution_dict[self.resolution_ratio]

        if image.image_size[0] != correct_width or image.image_size[1] != correct_height:
            warnings.warn("Current image resolution is not supported! Auto adjust...")
            image = image.resize((correct_width, correct_height))
            width, height = image.image_size

        # Step 3: Get low-resolution image area.
        lr_patch_width_size, lr_patch_height_size = int(width // width_blocks), int(height // height_blocks)

        # Step 4: The low resolution sub regions are processed in turn.
        for w in range(width_blocks):
            for h in range(height_blocks):
                # Step 5: Get the sub block image area of low-resolution image.
                lr_box = (lr_patch_width_size * w, lr_patch_height_size * h,
                          lr_patch_width_size * (w + 1), lr_patch_height_size * (h + 1))
                # Step 6: Crop specified area.
                region = image.crop(lr_box)
                # PIL image format convert to Tensor format.
                lr = transforms.ToTensor()(region).unsqueeze(0).to(device)
                with torch.no_grad():
                    sr = model(lr)
                # Step 7: Save the image area after super-resolution.
                sr = sr.cpu().squeeze()
                sr = sr.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
                sr = Image.fromarray(sr)
                sr.save(f"tmp_{w}_{h}.png")

    def fusion(self, src: str, dst: str, left_right: bool = True):
        # Step 1:
        img1 = cv2.imread(src, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(dst, cv2.COLOR_BGR2RGB)

        img1 = (img1 - img1.min()) / img1.ptp()
        img2 = (img2 - img2.min()) / img2.ptp()

        # H*W*C for OpenCV. W*H for Pillow.
        height, width, channels = img1.shape

        # Calculate the image edge weight when splicing.
        weight = 1 / (1 + np.exp(-self.ratio * np.arange(-self.over_length / 2, self.over_length / 2)))

        if left_right:  # Left and right integrated.
            # Create new white image.
            image = np.zeros((height, 2 * width - self.over_length, 3))

            # Get the weight of the image edge.
            edge_weight = np.tile(weight, (height, 1))

            # The RGB channel is processed in turn.
            for c in range(channels):
                # Copy the left and right parts of the image to the new image.
                image[:, :width, c] = img1[:, :, c]
                image[:, width:, c] = img2[:, self.over_length:, c]

                # The fusion region image of two images is obtained.
                src_fusion = (1 - edge_weight) * img1[:, width - self.over_length:width, c]
                dst_fusion = edge_weight * img2[:, :self.over_length, c]
                image[:, width - self.over_length: width, c] = src_fusion + dst_fusion

        else:  # Down and up integrated.
            # Create new white image.
            image = np.zeros((2 * height - self.over_length, width, 3))

            # Get the weight of the image edge.
            weight = np.reshape(weight, (self.over_length, 1))
            edge_weight = np.tile(weight, (1, width))

            # The RGB channel is processed in turn.
            for c in range(channels):
                # Copy the upper and lower parts of the image to the new image.
                image[:height:, :, c] = img1[:, :, c]
                image[height:, :, c] = img2[self.over_length:, :, c]

                # The fusion region image of two images is obtained.
                src_fusion = (1 - edge_weight) * img1[height - self.over_length:height, :, c]
                dst_fusion = edge_weight * img2[:self.over_length, :, c]
                image[height - self.over_length:height, :, c] = src_fusion + dst_fusion

        return np.uint16(image * 65535)

    def run(self):
        # Super resolution image generation.
        self.inference(self.model, self.device)

        logger.info("Staring fusion image...")
        # Merge the two upper image above.
        cv2.imwrite("tmp1.png", self.fusion("tmp_0_0.png", "tmp_1_0.png", left_right=True))
        # Merge the two downer image above.
        cv2.imwrite("tmp2.png", self.fusion("tmp_0_1.png", "tmp_1_1.png", left_right=True))
        # Merge the two upper-downer above.
        cv2.imwrite(f"sr_{os.path.basename(self.file_path).split('.')[0]}.png",
                    self.fusion("tmp1.png", "tmp2.png", left_right=False))

        logger.info(f"Cleaning temp image...")
        files = os.listdir(".")
        for _, filename in enumerate(files):
            if filename.find("tmp") >= 0:
                os.remove(filename)


class SR(object):
    def __init__(self, filename: str, model: torch.nn.Module, device: torch.device, resolution_ratio: str = "1080p"):
        self.filename = filename
        self.model = model
        self.device = device
        self.resolution_ratio = resolution_ratio
        self.over_length = 128  # Edge overlap length.
        self.ratio = 0.05  # Fusion calculation weight parameters.

    def inference(self):
        r""" Super-resolution of low resolution image."""
        # Step 1: Read image.
        image = Image.open(self.filename)

        # Step 2: Check whether the image resolution meets the standard.
        width, height = image.image_size
        correct_width, correct_height, width_blocks, height_blocks = resolution_dict[self.resolution_ratio]

        if image.image_size[0] != correct_width or image.image_size[1] != correct_height:
            warnings.warn("Current image resolution is not supported! Auto adjust...")
            image = image.resize((correct_width, correct_height))
            width, height = image.image_size

        # Step 3: Get low-resolution image area.
        lr_patch_width_size, lr_patch_height_size = int(width // width_blocks), int(height // height_blocks)

        # Step 4: The low resolution sub regions are processed in turn.
        for w in range(width_blocks):
            for h in range(height_blocks):
                # Step 5: Get the sub block image area of low-resolution image.
                lr_box = (lr_patch_width_size * w, lr_patch_height_size * h,
                          lr_patch_width_size * (w + 1), lr_patch_height_size * (h + 1))
                # Step 6: Crop specified area.
                region = image.crop(lr_box)
                # PIL image format convert to Tensor format.
                lr = transforms.ToTensor()(region).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    sr = self.model(lr)
                # Step 7: Save the image area after super-resolution.
                sr = sr.cpu().squeeze()
                sr = sr.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
                sr = Image.fromarray(sr)
                sr.save(f"tmp_{w}_{h}.png")

    def fusion(self, src: str, dst: str, left_right: bool = True):
        # Step 1:
        img1 = cv2.imread(src, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(dst, cv2.COLOR_BGR2RGB)

        img1 = (img1 - img1.min()) / img1.ptp()
        img2 = (img2 - img2.min()) / img2.ptp()

        # H*W*C for OpenCV. W*H for Pillow.
        height, width, channels = img1.shape

        # Calculate the image edge weight when splicing.
        weight = 1 / (1 + np.exp(-self.ratio * np.arange(-self.over_length / 2, self.over_length / 2)))

        if left_right:  # Left and right integrated.
            # Create new white image.
            image = np.zeros((height, 2 * width - self.over_length, 3))

            # Get the weight of the image edge.
            edge_weight = np.tile(weight, (height, 1))

            # The RGB channel is processed in turn.
            for c in range(channels):
                # Copy the left and right parts of the image to the new image.
                image[:, :width, c] = img1[:, :, c]
                image[:, width:, c] = img2[:, self.over_length:, c]

                # The fusion region image of two images is obtained.
                src_fusion = (1 - edge_weight) * img1[:, width - self.over_length:width, c]
                dst_fusion = edge_weight * img2[:, :self.over_length, c]
                image[:, width - self.over_length: width, c] = src_fusion + dst_fusion

        else:  # Down and up integrated.
            # Create new white image.
            image = np.zeros((2 * height - self.over_length, width, 3))

            # Get the weight of the image edge.
            weight = np.reshape(weight, (self.over_length, 1))
            edge_weight = np.tile(weight, (1, width))

            # The RGB channel is processed in turn.
            for c in range(channels):
                # Copy the upper and lower parts of the image to the new image.
                image[:height:, :, c] = img1[:, :, c]
                image[height:, :, c] = img2[self.over_length:, :, c]

                # The fusion region image of two images is obtained.
                src_fusion = (1 - edge_weight) * img1[height - self.over_length:height, :, c]
                dst_fusion = edge_weight * img2[:self.over_length, :, c]
                image[height - self.over_length:height, :, c] = src_fusion + dst_fusion

        os.remove(src)
        os.remove(dst)

        return np.uint16(image * 65535)

    def run(self):
        # Super resolution image generation.
        self.inference()

        logger.info("Staring fusion image...")
        # Merge the two upper image above.
        cv2.imwrite("tmp1.png", self.fusion("tmp_0_0.png", "tmp_1_0.png", left_right=True))
        # Merge the two downer image above.
        cv2.imwrite("tmp2.png", self.fusion("tmp_0_1.png", "tmp_1_1.png", left_right=True))
        # Merge the two upper-downer above.

        return self.fusion("tmp1.png", "tmp2.png", left_right=False)


class COS(object):
    def __init__(self):
        self.Region = "Your Region"
        self.SecretId = "Your SecretId"
        self.SecretKey = "Your SecretKey"
        self.Scheme = "https"
        self.Bucket = "Your Bucket"

        self.config = CosConfig(Region=self.Region,
                                SecretId=self.SecretId,
                                SecretKey=self.SecretKey,
                                Scheme=self.Scheme)
        self.client = CosS3Client(self.config)

    def get_all_urls(self):
        client = self.client  # init client API.
        Bucket = self.Bucket
        url_lists = []  # Save all url address.
        marker = ""  # start index from 0.

        while True:
            response = client.list_objects(Bucket=Bucket, Marker=marker)
            # Get corresponding image files.
            for index in range(len(response["Contents"])):
                Key = response["Contents"][index]["Key"]
                # Filtering long url and non image url.
                if len(Key) <= 50 and check_image_file(Key):
                    url_lists.append(Key)

            if response["IsTruncated"] == "false":
                break
            marker = response["NextMarker"]

        # Sort by the latest time.
        url_lists.sort(reverse=True)

        return url_lists

    def download_file(self, file_path, cos_path):
        """Download file from cos url address.
        Files less than or equal to 20MB are simply downloaded, and files larger than 20MB are downloaded continuously.

        Args:
            file_path (string): Download the file to save the address.
            cos_path (string): COS url address.
        """
        client = self.client  # init client API.
        response = client.get_object(
            Bucket=self.Bucket,
            Key=cos_path,
        )
        response["Body"].get_stream_to_file(file_path)

    def upload_file(self, stream, cos_path):
        """Upload file to COS URL.
        Files less than or equal to 20MB are uploaded simply, and files larger than 20MB are uploaded in blocks.

        Args:
            stream (bytes): The uploaded file content is of file stream or byte stream type.
            cos_path (string): COS url address.
        """
        client = self.client  # init client API.
        client.put_object(
            Bucket=self.Bucket,  # Bucket storage name.
            Body=stream,
            Key=cos_path  # Block upload path name.
        )
