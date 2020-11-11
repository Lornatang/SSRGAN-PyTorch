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
import csv
import os

import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm

import ssrgan.models as models
from ssrgan import DatasetFromFolder
from ssrgan import VGGLoss
from ssrgan.models import BioNet
from ssrgan.models import DiscriminatorForVGG
from ssrgan.utils import Logger
from ssrgan.utils import init_torch_seeds
from ssrgan.utils import load_checkpoint
from ssrgan.utils import select_device

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class Trainer:
    def __init__(self, args):
        args.tensorboard_dir = args.log_dir if args.tensorboard_dir is None else args.tensorboard_dir
        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.manualSeed)
        logger = Logger(args)

        # Selection of appropriate treatment equipment
        device = select_device(args.device, batch_size=args.batch_size)

        # Selection of appropriate treatment equipment
        dataset = DatasetFromFolder(input_dir=f"{args.dataroot}/{args.upscale_factor}x/train/input",
                                    target_dir=f"{args.dataroot}/{args.upscale_factor}x/train/target")
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=args.batch_size,
                                                 pin_memory=True,
                                                 num_workers=int(args.workers))
        # Construct network architecture model of generator and discriminator.
        discriminator = DiscriminatorForVGG().to(device)
        generator = BioNet().to(device)

        # Parameters of pre training model.
        psnr_epochs = int(args.psnr_iters // len(dataloader))
        psnr_epoch_indices = int(psnr_epochs // 4)
        psnr_optimizer = torch.optim.Adam(generator.parameters(), lr=args.psnr_lr, betas=(0.9, 0.999))
        psnr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(psnr_optimizer,
                                                                              T_0=psnr_epoch_indices,
                                                                              T_mult=1,
                                                                              eta_min=1e-7)

        # Parameters of GAN training model.
        epochs = int(args.iters // len(dataloader))
        base_epoch = int(epochs // 8)
        indices = [base_epoch, base_epoch * 2, base_epoch * 4, base_epoch * 6]
        optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
        optimizerG = torch.optim.Adam(generator.parameters(), lr=args.lr)
        schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizerD, milestones=indices, gamma=0.5)
        schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, milestones=indices, gamma=0.5)

        # We use VGG5.4 as our feature extraction method by default.
        vgg_criterion = VGGLoss().to(device)
        # Loss = 10 * l1 loss + vgg loss + 5e-3 * adversarial loss
        pix_criterion = nn.L1Loss().to(device)
        adversarial_criterion = nn.BCEWithLogitsLoss().to(device)

        self.args = args
        # Selection of appropriate treatment equipment.
        self.device = device

        self.dataloader = dataloader
        # Build all training models.
        self.discriminator = discriminator
        self.generator = generator
        # Define the parameters of the pre training model.
        self.psnr_epochs = psnr_epochs
        self.psnr_optimizer = psnr_optimizer
        self.psnr_scheduler = psnr_scheduler
        # Define the parameters of the GAN training model.
        self.epochs = epochs
        self.optimizerD = optimizerD
        self.optimizerG = optimizerG
        self.schedulerD = schedulerD
        self.schedulerG = schedulerG

        # Define all loss function.
        self.vgg_criterion = vgg_criterion
        self.pix_criterion = pix_criterion
        self.adversarial_criterion = adversarial_criterion

        # Print log.
        self.logger = logger

    # Loading PSNR pre training model
    def resume_resnet(self):
        args = self.args
        generator = self.generator
        psnr_optimizer = self.psnr_optimizer
        args.start_epoch = load_checkpoint(generator, psnr_optimizer,
                                           f"./weight/ResNet_{args.upscale_factor}x_checkpoint.pth")

    # Loading GAN checkpoint
    def resume_gan(self):
        args = self.args
        discriminator = self.discriminator
        generator = self.generator
        optimizerD = self.optimizerD
        optimizerG = self.optimizerG
        args.start_epoch = load_checkpoint(discriminator, optimizerD,
                                           f"./weight/netD_{args.upscale_factor}x_checkpoint.pth")
        args.start_epoch = load_checkpoint(generator, optimizerG,
                                           f"./weight/netG_{args.upscale_factor}x_checkpoint.pth")

    def run(self):
        args = self.args
        # Selection of appropriate treatment equipment.
        device = self.device

        dataloader = self.dataloader
        # Build all training models.
        discriminator = self.discriminator
        generator = self.generator
        # Define the parameters of the pre training model.
        psnr_epochs = self.psnr_epochs
        psnr_optimizer = self.psnr_optimizer
        psnr_scheduler = self.psnr_scheduler
        # Define the parameters of the GAN training model.
        epochs = self.epochs
        optimizerD = self.optimizerD
        optimizerG = self.optimizerG
        schedulerD = self.schedulerD
        schedulerG = self.schedulerG

        # Define all loss function.
        vgg_criterion = self.vgg_criterion
        pix_criterion = self.pix_criterion
        adversarial_criterion = self.adversarial_criterion

        # Print log.
        logger = self.logger

        # Set the all model to training mode
        discriminator.train()
        generator.train()

        # Pre-train generator using raw l1 loss
        logger.print_info("[*] Start training PSNR model based on L1 loss.")
        # Save the generator model based on MSE pre training to speed up the training time
        if os.path.exists(f"./weight/ResNet_{args.upscale_factor}x.pth"):
            logger.print_info("[*] Found PSNR pretrained model weights. Skip pre-train.")
            generator.load_state_dict(torch.load(f"./weight/ResNet_{args.upscale_factor}x.pth", map_location=device))
        else:
            # Writer train PSNR model log.
            if args.start_epoch == 0:
                with open(f"ResNet_{args.upscale_factor}x_Loss.csv", "w+") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Epoch", "Loss"])
            logger.print_info("[!] Not found pretrained weights. Start training PSNR model.")
            for epoch in range(args.start_epoch, psnr_epochs):
                progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
                avg_loss = 0.
                for i, (input, target) in progress_bar:
                    # Set generator gradients to zero
                    generator.zero_grad()
                    # Generate data
                    lr = input.to(device)
                    hr = target.to(device)

                    # Generating fake high resolution images from real low resolution images.
                    sr = generator(lr)
                    # The MSE of the generated fake high-resolution image and real high-resolution image is calculated.
                    loss = pix_criterion(sr, hr)
                    # Calculate gradients for generator
                    loss.backward()
                    # Update generator weights
                    psnr_optimizer.step()

                    # Dynamic adjustment of learning rate
                    psnr_scheduler.step()

                    avg_loss += loss.item()

                    progress_bar.set_description(f"[{epoch + 1}/{psnr_epochs}][{i + 1}/{len(dataloader)}] "
                                                 f"Loss: {loss.item():.6f}")

                    # record iter.
                    total_iter = len(dataloader) * epoch + i

                    # The image is saved every 5000 iterations.
                    if (total_iter + 1) % args.save_freq == 0:
                        vutils.save_image(hr, os.path.join("./output/hr", f"ResNet_{total_iter + 1}.bmp"))
                        vutils.save_image(sr, os.path.join("./output/sr", f"ResNet_{total_iter + 1}.bmp"))

                # The model is saved every 1 epoch.
                torch.save({"epoch": epoch + 1,
                            "optimizer": psnr_optimizer.state_dict(),
                            "state_dict": generator.state_dict()
                            }, f"./weight/ResNet_{args.upscale_factor}x_checkpoint.pth")

                # Writer training log
                with open(f"ResNet_{args.upscale_factor}x_Loss.csv", "a+") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, avg_loss / len(dataloader)])

            torch.save(generator.state_dict(), f"./weight/ResNet_{args.upscale_factor}x.pth")
            print(f"[*] Training PSNR model done! Saving PSNR model weight to "
                  f"`./weight/ResNet_{args.upscale_factor}x.pth`.")
