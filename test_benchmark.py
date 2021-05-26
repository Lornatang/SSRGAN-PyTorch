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
import argparse
import logging
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm

import ssrgan.models as models
from ssrgan.dataset import CustomTestDataset
from ssrgan.utils.common import configure
from ssrgan.utils.common import create_folder
from ssrgan.utils.estimate import iqa

# Find all available models.
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main(args):
    # Initialize all evaluation criteria.
    total_mse_value, total_rmse_value, total_psnr_value, total_ssim_value, total_gmsd_value = 0.0, 0.0, 0.0, 0.0, 0.0
    # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set convolution algorithm.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build a super-resolution model, if model_ If path is defined, the specified model weight will be loaded.
    model = configure(args)
    # Switch model to eval mode.
    model.eval()

    # If the GPU is available, load the model into the GPU memory. This speed.
    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # Setting this flag allows the built-in auto tuner of cudnn to automatically find the most efficient algorithm suitable
        # for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = True
        # Ensure that every time the same input returns the same result.
        cudnn.deterministic = True

    logger.info("Load testing dataset.")
    # Selection of appropriate treatment equipment.
    dataset = CustomTestDataset(os.path.join(args.data, "test"), args.image_size, args.sampler_frequency)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers)
    logger.info(f"Dataset information:\n"
                f"\tPath:              {os.getcwd()}/{args.data}/test\n"
                f"\tNumber of samples: {len(dataset)}\n"
                f"\tNumber of batches: {len(dataloader)}\n"
                f"\tShuffle:           False\n"
                f"\tSampler:           None\n"
                f"\tWorkers:           {args.workers}")

    # It only needs to reconstruct the low resolution image without the gradient information of the reconstructed image.
    with torch.no_grad():
        # Start evaluate model performance.
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (lr, bicubic, hr) in progress_bar:
            # Move data to special device.
            if args.gpu is not None:
                lr = lr.cuda(args.gpu, non_blocking=True)
                bicubic = bicubic.cuda(args.gpu, non_blocking=True)
                hr = hr.cuda(args.gpu, non_blocking=True)

            # The low resolution image is reconstructed to the super resolution image.
            sr = model(lr)

            # The reconstructed image and the reference image are evaluated once.
            value = iqa(sr, hr, args.gpu)

            # The values of various evaluation indexes are accumulated.
            total_mse_value += value[0]
            total_rmse_value += value[1]
            total_psnr_value += value[2]
            total_ssim_value += value[3]
            total_gmsd_value += value[4]

            # Output as scrollbar style.
            progress_bar.set_description(f"[{i + 1}/{len(dataloader)}] "
                                         f"PSNR: {total_psnr_value / (i + 1):6.2f} "
                                         f"SSIM: {total_ssim_value / (i + 1):6.4f}")

            # Merge three images into one line for visualization.
            # Save a series of reconstruction results.
            vutils.save_image(torch.cat([bicubic, sr, hr], dim=-1),
                              os.path.join("benchmarks", f"{i + 1}.bmp"),
                              padding=10,
                              normalize=True)

    print(f"Performance average results:\n")
    print(f"Indicator score\n")
    print(f"--------- -----\n")
    print(f"MSE       {total_mse_value / len(dataloader):6.4f}\n"
          f"RMSE      {total_rmse_value / len(dataloader):6.4f}\n"
          f"PSNR      {total_psnr_value / len(dataloader):6.2f}\n"
          f"SSIM      {total_ssim_value / len(dataloader):6.4f}\n"
          f"GMSD      {total_gmsd_value / len(dataloader):6.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", metavar="DIR",
                        help="Path to dataset.")
    parser.add_argument("-a", "--arch", metavar="ARCH", default="pmigan",
                        choices=model_names,
                        help="Model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `pmigan`)")
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                        help="Number of data loading workers. (Default: 8)")
    parser.add_argument("-b", "--batch-size", default=32, type=int,
                        metavar="N",
                        help="mini-batch size (default: 32), this is the total "
                             "batch size of all GPUs on the current node when "
                             "using Data Parallel or Distributed Data Parallel")
    parser.add_argument("--sampler-frequency", default=1, type=int, metavar="N",
                        help="If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)")
    parser.add_argument("--image-size", type=int, default=216,
                        help="Image size of high resolution image. (Default: 216)")
    parser.add_argument("--upscale-factor", type=int, default=4, choices=[4],
                        help="Low to high resolution scaling factor. Optional: [4]. (Default: 4)")
    parser.add_argument("--model-path", default="", type=str, metavar="PATH",
                        help="Path to latest checkpoint for model. (Default: ``)")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--seed", default=666, type=int,
                        help="Seed for initializing training. (Default: 666)")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    args = parser.parse_args()

    print("##################################################\n")
    print("Run Testing Engine.\n")

    create_folder("tests")

    logger.info("TestEngine:")
    print("\tAPI version .......... 0.1.4")
    print("\tBuild ................ 2021.05.26")
    print("##################################################\n")
    main(args)

    logger.info("Test dataset performance evaluation completed successfully.")
