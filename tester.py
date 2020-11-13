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

from ssrgan.dataset import BaseDataset
from ssrgan.utils import configure
from ssrgan.utils import get_time
from ssrgan.utils import image_quality_evaluation
from ssrgan.utils import inference
from ssrgan.utils import process_image


class Test(object):
    def __init__(self, args):
        self.args = args
        print(f"[*]({get_time()})Loading model architecture[{args.arch}]...")
        self.model, self.device = configure(args)
        print(f"[*]({get_time()})Loaded [{args.arch}] model done!")

        print(f"[*]({get_time()})Loading dataset...")
        self.dataloader = torch.utils.data.DataLoader(
            dataset=BaseDataset(dir_path=f"{args.dataroot}/test"),
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=int(args.workers))
        print(f"[*]({get_time()})Loaded dataset done!")

    def run(self):
        args = self.args
        model = self.model
        device = self.device
        dataloader = self.dataloader

        # Evaluate algorithm performance
        total_psnr_value = 0.0
        total_ssim_value = 0.0
        total_lpips_value = 0.0

        # Start evaluate model performance
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (input, target) in progress_bar:
            # Set model gradients to zero
            lr = input.to(device)
            hr = target.to(device)

            sr = inference(model, lr)
            vutils.save_image(sr, f"./{args.outf}/sr_{i}.bmp")  # Save super resolution image.
            vutils.save_image(hr, f"./{args.outf}/hr_{i}.bmp")  # Save high resolution image.

            # Evaluate performance
            psnr_value, ssim_value, lpips_value = image_quality_evaluation(f"./{args.outf}/sr_{i}.bmp",
                                                                           f"./{args.outf}/hr_{i}.bmp",
                                                                           device)

            total_psnr_value += psnr_value
            total_ssim_value += ssim_value[0]
            total_lpips_value += lpips_value.item()

            progress_bar.set_description(f"[{i + 1}/{len(dataloader)}] "
                                         f"PSNR: {psnr_value:.2f}dB "
                                         f"SSIM: {ssim_value[0]:.4f} "
                                         f"LPIPS: {lpips_value.item():.4f}")

        print("====================== Performance summary ======================")
        print(f"Avg PSNR: {total_psnr_value / len(dataloader):.2f}\n"
              f"Avg SSIM: {total_ssim_value / len(dataloader):.4f}\n"
              f"Avg SSIM: {total_lpips_value / len(dataloader):.4f}\n")
        print("============================== End ==============================")


class Estimate(object):
    def __init__(self, args):
        self.args = args
        print(f"[*]({get_time()})Loading model architecture[{args.arch}]...")
        self.model, self.device = configure(args)
        print(f"[*]({get_time()})Loaded [{args.arch}] model done!")

    def run(self):
        args = self.args
        model = self.model
        device = self.device

        # Read img to tensor and transfer to the specified device for processing.
        img = Image.open(args.lr)
        lr = process_image(img, device)

        sr, use_time = inference(model, lr, statistical_time=True)
        vutils.save_image(sr, f"./{args.outf}/{args.lr}")  # Save super resolution image.

        psnr_value, ssim_value, lpips_value = image_quality_evaluation(f"./{args.outf}/{args.lr}", args.hr, device)
        print("====================== Performance summary ======================")
        print(f"PSNR: {psnr_value:.2f}dB\n"
              f"SSIM: {ssim_value[0]:.4f}\n"
              f"LPIPS: {lpips_value.item():.4f}."
              "Use time: {use_time * 1000:.2f}ms/{use_time:.4f}s.")
        print("============================== End ==============================")


class Video(object):
    def __init__(self, args):
        self.args = args

        print(f"[*]({get_time()})Loading model architecture[{args.arch}]...")
        self.model, self.device = configure(args)
        print(f"[*]({get_time()})Loaded [{args.arch}] model done!")

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
        self.sr_writer = cv2.VideoWriter(f"sr_{args.scale_factor}x_{os.path.basename(args.file)}",
                                         cv2.VideoWriter_fourcc(*"MPEG"), self.fps, self.sr_size)
        self.compare_writer = cv2.VideoWriter(f"compare_{args.scale_factor}x_{os.path.basename(args.file)}",
                                              cv2.VideoWriter_fourcc(*"MPEG"), self.fps, self.pare_size)

    def run(self):
        args = self.args
        model = self.model
        device = self.device

        tensor2pil = self.tensor2pil
        video_capture = self.video_capture
        total_frames = self.total_frames
        sr_size = self.sr_size
        sr_writer = self.sr_writer
        compare_writer = self.compare_writer

        # Set eval model.
        model.eval()

        # read frame
        success, raw_frame = video_capture.read()
        progress_bar = tqdm(range(total_frames), desc="[processing video and saving/view result videos]")
        for _ in progress_bar:
            if success:
                # Read img to tensor and transfer to the specified device for processing.
                img = Image.open(args.lr)
                lr = process_image(img, device)

                sr = inference(model, lr)

                sr = sr.cpu()
                sr = sr.data[0].numpy()
                sr *= 255.0
                sr = (np.uint8(sr)).transpose((1, 2, 0))
                # save sr video
                sr_writer.write(sr)

                # make compared video and crop shot of left top\right top\center\left bottom\right bottom
                sr = tensor2pil(sr)
                # Five areas are selected as the bottom contrast map.
                crop_sr_imgs = transforms.FiveCrop(size=sr.width // 5 - 9)(sr)
                crop_sr_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_sr_imgs]
                sr = transforms.Pad(padding=(5, 0, 0, 5))(sr)
                # Five areas in the contrast map are selected as the bottom contrast map
                compare_img = transforms.Resize((sr_size[1], sr_size[0]),
                                                interpolation=Image.BICUBIC)(tensor2pil(raw_frame))
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
                    transforms.Resize((bottom_img_height, bottom_img_width))(tensor2pil(bottom_img)))
                # 4. Combine the bottom zone with the upper zone.
                final_image = np.concatenate((top_img, bottom_img))

                # save compare video
                compare_writer.write(final_image)

                if args.view:
                    # display video
                    cv2.imshow("LR video convert HR video ", final_image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # next frame
                success, raw_frame = video_capture.read()
