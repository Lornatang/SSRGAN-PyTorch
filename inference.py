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
import time

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sewar.full_ref import mse
from sewar.full_ref import msssim
from sewar.full_ref import psnr
from sewar.full_ref import rmse
from sewar.full_ref import sam
from sewar.full_ref import ssim
from sewar.full_ref import vifp

from ssrgan_pytorch import cal_niqe
from ssrgan_pytorch import opencv2pil
from ssrgan_pytorch import pil2opencv
from ssrgan_pytorch import select_device
from ssrgan_pytorch.models import BioNet

# Define image params
UPSCALE_FACTOR = 4  # Ony support 4 expand factor.
LR_WIDTH, LR_HEIGHT = 486, 486
SR_WIDTH = LR_WIDTH * UPSCALE_FACTOR  # For SR image
SR_HEIGHT = LR_HEIGHT * UPSCALE_FACTOR  # For SR image
NUM_WIDTH, NUM_HEIGHT = 9, 9  # For our patch size (width=64 height=54)
sr = Image.new("RGB", (SR_WIDTH, SR_HEIGHT))

# Get LR patch size.
PATCH_LR_WIDTH_SIZE = int(LR_WIDTH // NUM_WIDTH)
PATCH_LR_HEIGHT_SIZE = int(LR_HEIGHT // NUM_HEIGHT)
# Get HR patch size.
PATCH_HR_WIDTH_SIZE = int(SR_WIDTH // NUM_WIDTH)
PATCH_HR_HEIGHT_SIZE = int(SR_HEIGHT // NUM_HEIGHT)


# Super divide image reasoning entry, return the picture in pillow format.
def inference(model: nn.Module, img, device):
    """

    Args:
        model
        img:
        device:

    Returns:

    """
    # OpenCV to PIL.Image format
    lr = opencv2pil(img)

    for w in range(NUM_WIDTH):
        for h in range(NUM_HEIGHT):
            lr_box = (PATCH_LR_WIDTH_SIZE * w, PATCH_LR_HEIGHT_SIZE * h,
                      PATCH_LR_WIDTH_SIZE * (w + 1), PATCH_LR_HEIGHT_SIZE * (h + 1))
            hr_box = (PATCH_HR_WIDTH_SIZE * w, PATCH_HR_HEIGHT_SIZE * h,
                      PATCH_HR_WIDTH_SIZE * (w + 1), PATCH_HR_HEIGHT_SIZE * (h + 1))
            region = lr.crop(lr_box)
            patch_lr = pil2tensor(region).unsqueeze(0)
            patch_lr = patch_lr.to(device)
            with torch.no_grad():
                patch_hr = model(patch_lr)
            patch_hr = patch_hr.cpu().squeeze()
            patch_hr = patch_hr.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            patch_hr = Image.fromarray(patch_hr)
            sr.paste(patch_hr, hr_box)

    # PIL.Image Convert to OpenCV format
    img = pil2opencv(sr)
    return img


if __name__ == "__main__":
    # Selection of appropriate treatment equipment. default set CUDA:0
    device = select_device("0", batch_size=1)

    # Conversion between PIL format and tensor format.
    pil2tensor = transforms.ToTensor()
    tensor2pil = transforms.ToPILImage()

    # Construct GAN model.
    model = BioNet().to(device)
    model.load_state_dict(torch.load("weight/ResNet_4x.pth", map_location=device))

    # Set model eval mode
    model.eval()

    lr = cv2.imread("lr.bmp")

    start_time = time.time()
    sr = inference(model, lr, device)
    print(f"Use time: {time.time() - start_time:.2}s")

    cv2.imwrite("sr.bmp", sr)

    # Evaluate performance
    src_img = cv2.imread("sr.bmp")
    dst_img = cv2.imread("hr.bmp")

    mse_value = mse(src_img, dst_img)
    rmse_value = rmse(src_img, dst_img)
    psnr_value = psnr(src_img, dst_img)
    ssim_value = ssim(src_img, dst_img)
    ms_ssim_value = msssim(src_img, dst_img)  # 30.00+000j
    niqe_value = cal_niqe("sr.bmp")
    sam_value = sam(src_img, dst_img)
    vif_value = vifp(src_img, dst_img)

    print("====================== Performance summary ======================")
    print(f"MSE: {mse_value:.2f}\n"
          f"RMSE: {rmse_value:.2f}\n"
          f"PSNR: {psnr_value:.2f}\n"
          f"SSIM: {ssim_value[0]:.4f}\n"
          f"MS-SSIM: {ms_ssim_value.real:.4f}\n"
          f"NIQE: {niqe_value:.2f}\n"
          f"SAM: {sam_value:.4f}\n"
          f"VIF: {vif_value:.4f}")
    print("============================== End ==============================")
