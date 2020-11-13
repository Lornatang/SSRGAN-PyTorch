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

import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask
from flask import jsonify
from flask import request
from gevent import pywsgi
from ssrgan.models import bionet
from ssrgan.utils import create_folder
from ssrgan.utils import select_device

from .cos_server import COS

# Flask main start
app = Flask(__name__)

# Image resolution. [width, height, width block num, height block num]
resolution_dict = {
    "240p": [320, 240, 4, 3],
    "360p": [480, 360, 8, 6],
    "720p": [1080, 720, 12, 9],
    "1080p": [1920, 1080, 16, 9],
    "2k": [2560, 1440, 20, 15],
    "4k": [4096, 2160, 32, 20],
    "8k": [7680, 4320, 60, 30]
}

# Define image params.
upscale_factor = 4  # Ony support 4 expand factor.

# Configure super-resolution image parameters.
lr_width, lr_height, width_blocks, height_blocks = resolution_dict["240p"]
sr_width, sr_height = lr_width * upscale_factor, lr_height * upscale_factor

# Configure super-resolution image block size.
lr_patch_width_size = int(lr_width // width_blocks)
lr_patch_height_size = int(lr_height // height_blocks)
sr_patch_width_size = int(sr_width // width_blocks)
sr_patch_height_size = int(sr_height // height_blocks)

# Configure download low-resolution image directory and super-resolution image directory.
data_path = "static"
lr_path = os.path.join(data_path, "lr")
sr_path = os.path.join(data_path, "sr")


def inference(image):
    """Super-resolution of low resolution image.

    Args:
        image (PIL.JpegImagePlugin.JpegImageFile): Image read by PIL.

    Returns:
        PIL.JpegImagePlugin.JpegImageFile.
    """
    # Step 1: How many rows and columns will an image be divided into.
    for w in range(width_blocks):
        for h in range(height_blocks):
            # Step 2: Get the sub block image area of low-resolution image.
            lr_box = (lr_patch_width_size * w, lr_patch_height_size * h,
                      lr_patch_width_size * (w + 1), lr_patch_height_size * (h + 1))
            # Step 3: Get the sub block image area of super-resolution image.
            sr_box = (sr_patch_width_size * w, sr_patch_height_size * h,
                      sr_patch_width_size * (w + 1), sr_patch_height_size * (h + 1))
            # Step 4: Crop specified area.
            region = image.crop(lr_box)
            # PIL image format convert to Tensor format.
            patch_lr = transforms.ToTensor()(region).unsqueeze(0).to(device)
            with torch.no_grad():
                patch_sr = model(patch_lr)
            # Step 5: Save the image area after super-resolution.
            patch_sr = patch_sr.cpu().squeeze()
            patch_sr = patch_sr.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            patch_sr = Image.fromarray(patch_sr)
            sr.paste(patch_sr, sr_box)

    return sr


@app.route("/run", methods=["POST"])
def run():
    if request.method == "POST":
        for index in range(len(url_lists)):
            cos_path = url_lists[index]
            filename = os.path.basename(cos_path)
            lr_file_path = os.path.join(os.path.join(lr_path, filename))
            sr_file_path = os.path.join(os.path.join(sr_path, filename))

            # Step 1: If the file exists, it is considered that it has been processed by default.
            if not os.path.exists(lr_file_path):
                # Step 2: Download image to `static/lr`.
                print(f"Download `{filename}`.")
                cos.download_file(lr_file_path, cos_path)

                # Step 3: Start super-resolution.
                print(f"Process `{filename}`.")
                sr = inference(Image.open(lr_file_path))
                sr.save(sr_file_path)

                # Step 4: Read the super-resolution image into bytes.
                sr_image_bytes = open(sr_file_path, "rb").read()

                # Step 5: Upload image to COS.
                print(f"Upload `{filename}`.")
                cos.upload_file(sr_image_bytes, cos_path)
            else:
                print(f"Filter `{filename}`.")
            torch.cuda.empty_cache()  # Clear CUDA cache.
        return jsonify({"code": 20000, "msg": "SR complete!"})


if __name__ == "__main__":
    # Create all process directory.
    create_folder(data_path)
    create_folder(lr_path)
    create_folder(sr_path)

    # Model of configuration super-resolution algorithm.
    device = select_device()
    model = bionet(pretrained=True, upscale_factor=4).to(device)
    model.eval()

    # Create SR image.
    sr = Image.new("RGB", (sr_width, sr_height))

    # Start Tencent COS server.
    cos = COS()
    url_lists = cos.get_all_urls()  # Get COS all image file.
    base_url = "Your COS URL"

    # Flask server run.
    server = pywsgi.WSGIServer(("0.0.0.0", 10086), app)
    server.serve_forever()
