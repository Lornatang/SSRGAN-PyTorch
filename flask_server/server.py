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

import cv2
import torch
from flask import Flask
from flask import jsonify
from flask import request
from gevent import pywsgi

from engine import COS
from engine import SR
from model import bionet
from ssrgan.utils import create_folder
from ssrgan.utils import select_device

# Flask main start
app = Flask(__name__)


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
                sr = SR(lr_file_path, model, device).run()
                cv2.imwrite(sr_file_path, sr)

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
    # Configure download low-resolution image directory and super-resolution image directory.
    data_path = "static"
    lr_path = os.path.join(data_path, "lr")
    sr_path = os.path.join(data_path, "sr")

    # Step 1: Create all process directory.
    create_folder(data_path)
    create_folder(lr_path)
    create_folder(sr_path)

    # Step 2: Model of configuration super-resolution algorithm.
    device = select_device()
    model = bionet().to(device)
    model.eval()

    # Step 3: Start Tencent COS server.
    cos = COS()
    url_lists = cos.get_all_urls()  # Get COS all image file.
    base_url = "Your COS url"

    # Step 4: Flask server run.
    server = pywsgi.WSGIServer(("0.0.0.0", 10086), app)
    server.serve_forever()
