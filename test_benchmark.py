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
import argparse
import os

import cv2
import torch.utils.data
import torchvision.utils as vutils
from sewar.full_ref import psnr
from sewar.full_ref import ssim
from tqdm import tqdm

from ssrgan import DatasetFromFolder
from ssrgan import select_device
from ssrgan.models import BioNet

parser = argparse.ArgumentParser(description="Research and application of GAN based super resolution "
                                             "technology for pathological microscopic images.")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="Path to datasets. (default:`./data`)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default:4)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[4],
                    help="Low to high resolution scaling factor. (default:4).")
parser.add_argument("--model-path", default="./weight/GAN_4x.pth", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (default: ``./weight/GAN_4x.pth``).")
parser.add_argument("--device", default="0",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``CUDA:0``).")

args = parser.parse_args()

try:
    os.makedirs("benchmark")
except OSError:
    pass

# Selection of appropriate treatment equipment
device = select_device(args.device, batch_size=1)

dataset = DatasetFromFolder(input_dir=f"{args.dataroot}/{args.upscale_factor}x/test/input",
                            target_dir=f"{args.dataroot}/{args.upscale_factor}x/test/target")

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         pin_memory=True,
                                         num_workers=int(args.workers))

# Construct GAN model.
model = BioNet().to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))

# Set model eval mode
model.eval()

# Evaluate algorithm performance
total_psnr_value = 0.0
total_ssim_value = 0.0

# Start evaluate model performance
progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
for i, (input, target) in progress_bar:
    # Set model gradients to zero
    lr = input.to(device)
    hr = target.to(device)

    with torch.no_grad():
        sr = model(lr)

    vutils.save_image(lr, f"./benchmark/lr_{i}.bmp")
    vutils.save_image(sr, f"./benchmark/sr_{i}.bmp")
    vutils.save_image(hr, f"./benchmark/hr_{i}.bmp")

    # Evaluate performance
    src_img = cv2.imread(f"./benchmark/sr_{i}.bmp")
    dst_img = cv2.imread(f"./benchmark/hr_{i}.bmp")

    psnr_value = psnr(src_img, dst_img)
    ssim_value = ssim(src_img, dst_img)

    total_psnr_value += psnr_value
    total_ssim_value += ssim_value[0]

    progress_bar.set_description(f"[{i + 1}/{len(dataloader)}] "
                                 f"PSNR: {psnr_value:.2f}dB "
                                 f"SSIM: {ssim_value[0]:.4f}")

avg_psnr_value = total_psnr_value / len(dataloader)
avg_ssim_value = total_ssim_value / len(dataloader)

print("\n")
print("====================== Performance summary ======================")
print(f"Avg PSNR: {avg_psnr_value:.2f}\n"
      f"Avg SSIM: {avg_ssim_value:.4f}\n")
print("============================== End ==============================")
print("\n")
