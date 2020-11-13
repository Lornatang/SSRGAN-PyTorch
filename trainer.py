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

import lpips
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm

import ssrgan.models as models
from ssrgan import DatasetFromFolder
from ssrgan import VGGLoss
from ssrgan.models import DiscriminatorForVGG
from ssrgan.utils import init_torch_seeds
from ssrgan.utils import load_checkpoint
from ssrgan.utils import select_device

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class Trainer(object):
    def __init__(self, args):
        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.manualSeed)
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
        generator = models.__dict__[args.arch](upscale_factor=args.upscale_factor).to(device)

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
        # Reference sources from `https://github.com/richzhang/PerceptualSimilarity`
        self.lpips_loss = lpips.LPIPS(net="vgg").to(device)

    # Loading PSNR pre training model
    def resume_resnet(self):
        args = self.args
        generator = self.generator
        psnr_optimizer = self.psnr_optimizer
        args.start_epoch = load_checkpoint(generator, psnr_optimizer,
                                           f"./weights/ResNet_{args.upscale_factor}x_checkpoint.pth")

    # Loading GAN checkpoint
    def resume_gan(self):
        args = self.args
        discriminator = self.discriminator
        generator = self.generator
        optimizerD = self.optimizerD
        optimizerG = self.optimizerG
        args.start_epoch = load_checkpoint(discriminator, optimizerD,
                                           f"./weights/netD_{args.upscale_factor}x_checkpoint.pth")
        args.start_epoch = load_checkpoint(generator, optimizerG,
                                           f"./weights/netG_{args.upscale_factor}x_checkpoint.pth")

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

        # Set the all model to training mode
        discriminator.train()
        generator.train()

        # Loading PSNR pre training model
        if args.resume_PSNR:
            self.resume_resnet()

        # Pre-train generator using raw l1 loss
        print("[*] Start training PSNR model based on L1 loss.")
        # Save the generator model based on MSE pre training to speed up the training time
        if os.path.exists(f"./weights/ResNet_{args.upscale_factor}x.pth"):
            print("[*] Found PSNR pretrained model weights. Skip pre-train.")
            generator.load_state_dict(torch.load(f"./weights/ResNet_{args.upscale_factor}x.pth", map_location=device))
        else:
            # Writer train PSNR model log.
            if args.start_epoch == 0:
                with open(f"ResNet_{args.upscale_factor}x_Loss.csv", "w+") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Epoch", "Loss"])

            print("[!] Not found pretrained weights. Start training PSNR model.")
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
                            }, f"./weights/ResNet_{args.upscale_factor}x_checkpoint.pth")

                # Writer training log
                with open(f"ResNet_{args.upscale_factor}x_Loss.csv", "a+") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, avg_loss / len(dataloader)])

            torch.save(generator.state_dict(), f"./weights/ResNet_{args.upscale_factor}x.pth")
            print(f"[*] Training PSNR model done! Saving PSNR model weight to "
                  f"`./weights/ResNet_{args.upscale_factor}x.pth`.")

            # After training the PSNR model, set the initial iteration to 0.
            args.start_epoch = 0

            # Loading GAN checkpoint
            if args.resume:
                self.resume_gan()

            # Train GAN model.
            print(f"[*] Staring training GAN model!")
            print(f"[*] Training for {epochs} epochs.")
            # Writer train GAN model log.
            if args.start_epoch == 0:
                with open(f"GAN_{args.upscale_factor}x_Loss.csv", "w+") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Epoch", "D Loss", "G Loss"])

            for epoch in range(args.start_epoch, epochs):
                progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
                g_avg_loss = 0.
                d_avg_loss = 0.
                for i, (input, target) in progress_bar:
                    lr = input.to(device)
                    hr = target.to(device)
                    batch_size = lr.size(0)
                    real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype, device=device)
                    fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype, device=device)

                    ##############################################
                    # (1) Update D network: maximize - E(lr)[1- log(D(hr, sr))] - E(sr)[log(D(sr, hr))]
                    ##############################################
                    # Set discriminator gradients to zero.
                    discriminator.zero_grad()

                    # Generate a super-resolution image
                    sr = generator(lr)

                    # Train with real high resolution image.
                    hr_output = discriminator(hr)  # Train real image.
                    sr_output = discriminator(sr.detach())  # No train fake image.
                    # Adversarial loss for real and fake images (relativistic average GAN)
                    errD_hr = adversarial_criterion(hr_output - torch.mean(sr_output), real_label)
                    errD_sr = adversarial_criterion(sr_output - torch.mean(hr_output), fake_label)
                    errD = (errD_sr + errD_hr) / 2
                    errD.backward()
                    D_x = hr_output.mean().item()
                    D_G_z1 = sr_output.mean().item()
                    optimizerD.step()

                    ##############################################
                    # (2) Update G network: maximize - E(lr)[log(D(hr, sr))] - E(sr)[1- log(D(sr, hr))]
                    ##############################################
                    # Set generator gradients to zero
                    generator.zero_grad()

                    # According to the feature map, the root mean square error is regarded as the content loss.
                    vgg_loss = vgg_criterion(sr, hr)
                    # Train with fake high resolution image.
                    hr_output = discriminator(hr.detach())  # No train real fake image.
                    sr_output = discriminator(sr)  # Train fake image.
                    # Adversarial loss (relativistic average GAN)
                    adversarial_loss = adversarial_criterion(sr_output - torch.mean(hr_output), real_label)
                    # Pixel level loss between two images.
                    l1_loss = pix_criterion(sr, hr)
                    errG = 10 * l1_loss + vgg_loss + 5e-3 * adversarial_loss
                    errG.backward()
                    D_G_z2 = sr_output.mean().item()
                    optimizerG.step()

                    # Dynamic adjustment of learning rate
                    schedulerD.step()
                    schedulerG.step()

                    d_avg_loss += errD.item()
                    g_avg_loss += errG.item()

                    progress_bar.set_description(f"[{epoch + 1}/{epochs}][{i + 1}/{len(dataloader)}] "
                                                 f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                                                 f"D(HR): {D_x:.6f} D(G(LR)): {D_G_z1:.6f}/{D_G_z2:.6f}")

                    # record iter.
                    total_iter = len(dataloader) * epoch + i

                    # The image is saved every 5000 iterations.
                    if (total_iter + 1) % args.save_freq == 0:
                        vutils.save_image(hr, os.path.join("./output/hr", f"ResNet_{total_iter + 1}.bmp"))
                        vutils.save_image(sr, os.path.join("./output/sr", f"ResNet_{total_iter + 1}.bmp"))

                # The model is saved every 1 epoch.
                torch.save({"epoch": epoch + 1,
                            "optimizer": optimizerD.state_dict(),
                            "state_dict": discriminator.state_dict()
                            }, f"./weights/netD_{args.upscale_factor}x_checkpoint.pth")
                torch.save({"epoch": epoch + 1,
                            "optimizer": optimizerG.state_dict(),
                            "state_dict": generator.state_dict()
                            }, f"./weights/netG_{args.upscale_factor}x_checkpoint.pth")

                # Writer training log
                with open(f"GAN_{args.upscale_factor}x_Loss.csv", "a+") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1,
                                     d_avg_loss / len(dataloader),
                                     g_avg_loss / len(dataloader)])

            torch.save(generator.state_dict(), f"./weights/GAN_{args.upscale_factor}x.pth")
            print(f"[*] Training GAN model done! Saving GAN model weight "
                  f"to `./weights/GAN_{args.upscale_factor}x.pth`.")
