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
import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from ssrgan.dataset import CustomDataset
from ssrgan.utils import configure
from ssrgan.utils import get_time
from ssrgan.utils import image_quality_evaluation
from ssrgan.utils import inference
from ssrgan.utils import process_image


class Test(object):
    def __init__(self, args):
        self.args = args
        self.model, self.device = configure(args)

        print(f"[*]({get_time()})Loading dataset...")
        self.dataloader = torch.utils.data.DataLoader(
            CustomDataset(input_dir=f"{args.dataroot}/{args.upscale_factor}x/train/input",
                          target_dir=f"{args.dataroot}/{args.upscale_factor}x/train/target"),
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=int(args.workers))
        print(f"[*]({get_time()})Loaded dataset done!")

    def run(self):
        # Evaluate algorithm performance
        total_psnr_value = 0.0
        total_ssim_value = 0.0

        # Start evaluate model performance
        progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        for i, (input, target) in progress_bar:
            # Set model gradients to zero
            lr = input.to(self.device)
            hr = target.to(self.device)

            sr = inference(self.model, lr)
            vutils.save_image(sr, f"./{self.args.outf}/sr_{i}.bmp")  # Save super resolution image.
            vutils.save_image(hr, f"./{self.args.outf}/hr_{i}.bmp")  # Save high resolution image.

            # Evaluate performance
            psnr_value, ssim_value = image_quality_evaluation(f"./{self.args.outf}/sr_{i}.bmp",
                                                              f"./{self.args.outf}/hr_{i}.bmp",
                                                              self.device)

            total_psnr_value += psnr_value
            total_ssim_value += ssim_value[0]

            progress_bar.set_description(f"[{i + 1}/{len(self.dataloader)}] "
                                         f"PSNR: {psnr_value:.2f}dB "
                                         f"SSIM: {ssim_value[0]:.4f}")

        print("====================== Performance summary ======================")
        print(f"Avg PSNR: {total_psnr_value / len(self.dataloader):.2f}\n"
              f"Avg SSIM: {total_ssim_value / len(self.dataloader):.4f}\n")
        print("============================== End ==============================")


class Estimate(object):
    def __init__(self, args, detail=False):
        self.args = args
        self.model, self.device = configure(args)
        self.detail = detail

    def run(self):
        # Read img to tensor and transfer to the specified device for processing.
        img = Image.open(self.args.lr)
        lr = process_image(img, self.device)

        sr, use_time = inference(self.model, lr, statistical_time=True)
        vutils.save_image(sr, f"./{self.args.outf}/{self.args.lr}")  # Save super resolution image.

        value = image_quality_evaluation(f"./{self.args.outf}/{self.args.lr}", self.args.hr, self.detail, self.device)

        if self.detail:
            print("====================== Performance summary ======================")
            print(f"MSE: {value[0]:.2f}\n"
                  f"RMSE: {value[1]:.2f}\n"
                  f"PSNR: {value[2]:.2f}dB\n"
                  f"SSIM: {value[3][0]:.4f}\n"
                  f"MS-SSIM: {value[4]:.4f}\n"
                  f"NIQE: {value[5]:.2f}\n"
                  f"SAM: {value[6]:.4f}\n"
                  f"VIF: {value[7]:.4f}\n"
                  f"LPIPS: {value[8]:.4f}\n"
                  f"Use time: {use_time * 1000:.2f}ms/{use_time:.4f}s.")
            print("============================== End ==============================")
        else:
            print("====================== Performance summary ======================")
            print(f"PSNR: {value[0]:.2f}dB\n"
                  f"SSIM: {value[1][0]:.4f}\n"
                  f"Use time: {use_time * 1000:.2f}ms/{use_time:.4f}s.")
            print("============================== End ==============================")


class Video(object):
    def __init__(self, args):
        self.args = args
        self.model, self.device = configure(args)
        # Image preprocessing operation
        self.tensor2pil = transforms.ToPILImage()

        self.video_capture = cv2.VideoCapture(args.file)
        # Prepare to write the processed image into the video.
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Set video size
        self.size = (int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.sr_size = (self.size[0] * args.upscale_factor, self.size[1] * args.upscale_factor)
        self.pare_size = (self.sr_size[0] * 2 + 10, self.sr_size[1] + 10 + self.sr_size[0] // 5 - 9)
        # Video write loader.
        self.sr_writer = cv2.VideoWriter(f"./video/sr_{args.scale_factor}x_{os.path.basename(args.file)}",
                                         cv2.VideoWriter_fourcc(*"MPEG"), self.fps, self.sr_size)
        self.compare_writer = cv2.VideoWriter(f"./video/compare_{args.scale_factor}x_{os.path.basename(args.file)}",
                                              cv2.VideoWriter_fourcc(*"MPEG"), self.fps, self.pare_size)

    def run(self):
        # Set eval model.
        self.model.eval()

        # read frame
        success, raw_frame = self.video_capture.read()
        progress_bar = tqdm(range(self.total_frames), desc="[processing video and saving/view result videos]")
        for _ in progress_bar:
            if success:
                # Read img to tensor and transfer to the specified device for processing.
                img = Image.open(self.args.lr)
                lr = process_image(img, self.device)

                sr = inference(self.model, lr)

                sr = sr.cpu()
                sr = sr.data[0].numpy()
                sr *= 255.0
                sr = (np.uint8(sr)).transpose((1, 2, 0))
                # save sr video
                self.sr_writer.write(sr)

                # make compared video and crop shot of left top\right top\center\left bottom\right bottom
                sr = self.tensor2pil(sr)
                # Five areas are selected as the bottom contrast map.
                crop_sr_imgs = transforms.FiveCrop(size=sr.width // 5 - 9)(sr)
                crop_sr_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_sr_imgs]
                sr = transforms.Pad(padding=(5, 0, 0, 5))(sr)
                # Five areas in the contrast map are selected as the bottom contrast map
                compare_img = transforms.Resize((self.sr_size[1], self.sr_size[0]),
                                                interpolation=Image.BICUBIC)(self.tensor2pil(raw_frame))
                crop_compare_imgs = transforms.FiveCrop(size=compare_img.width // 5 - 9)(compare_img)
                crop_compare_imgs = [np.asarray(transforms.Pad(padding=(0, 5, 10, 0))(img)) for img in
                                     crop_compare_imgs]
                compare_img = transforms.Pad(padding=(0, 0, 5, 5))(compare_img)
                # concatenate all the pictures to one single picture
                # 1. Mosaic the left and right images of the video.
                top_img = np.concatenate((np.asarray(compare_img), np.asarray(sr)), axis=1)
                # 2. Mosaic the bottom left and bottom right images of the video.
                bottom_img = np.concatenate(crop_compare_imgs + crop_sr_imgs, axis=1)
                bottom_img_height = int(top_img.shape[1] / bottom_img.shape[1] * bottom_img.shape[0])
                bottom_img_width = top_img.shape[1]
                # 3. Adjust to the right size.
                bottom_img = np.asarray(
                    transforms.Resize((bottom_img_height, bottom_img_width))(self.tensor2pil(bottom_img)))
                # 4. Combine the bottom zone with the upper zone.
                final_image = np.concatenate((top_img, bottom_img))

                # save compare video
                self.compare_writer.write(final_image)

                if self.args.view:
                    # display video
                    cv2.imshow("LR video convert HR video ", final_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # next frame
                success, raw_frame = self.video_capture.read()
