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
import logging
import os

import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
from tqdm import tqdm

import ssrgan.models as models
from ssrgan import BaseTrainDataset
from ssrgan import VGGLoss
from ssrgan.models import DiscriminatorForVGG
from ssrgan.utils import configure
from ssrgan.utils import init_torch_seeds
from ssrgan.utils import load_checkpoint

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.manualSeed)

        logger.info("Load training dataset")
        # Selection of appropriate treatment equipment.
        self.dataloader = torch.utils.data.DataLoader(BaseTrainDataset(args.dataroot, crop_size=216, upscale_factor=4),
                                                      batch_size=args.batch_size,
                                                      pin_memory=True,
                                                      num_workers=int(args.workers))

        logger.info(f"Train Dataset information:\n"
                    f"\tTrain Dataset dir is `{os.getcwd()}/{args.dataroot}/train`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")

        # Construct network architecture model of generator and discriminator.
        self.generator, self.device = configure(args)
        logger.info(f"Creating discriminator model")
        self.discriminator = DiscriminatorForVGG().to(self.device)

        # Parameters of pre training model.
        self.psnr_epochs = int(args.psnr_iters // len(self.dataloader))
        self.psnr_epoch_indices = int(self.psnr_epochs // 4)
        self.psnr_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.psnr_lr, betas=(0.9, 0.99))
        self.psnr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.psnr_optimizer,
                                                                                   T_0=self.psnr_epoch_indices,
                                                                                   T_mult=1,
                                                                                   eta_min=1e-7)
        logger.info(f"Pre-training model training parameters:\n"
                    f"\tIters is {args.psnr_iters}\n"
                    f"\tEpoch is {self.psnr_epochs}\n"
                    f"\tOptimizer Adam\n"
                    f"\tLearning rate {args.psnr_lr}\n"
                    f"\tBetas (0.9, 0.99)\n"
                    f"\tScheduler CosineAnnealingWarmRestarts")

        # Parameters of GAN training model.
        self.epochs = int(args.iters // len(self.dataloader))
        self.base_epoch = int(self.epochs // 8)
        self.indices = [self.base_epoch, self.base_epoch * 2, self.base_epoch * 4, self.base_epoch * 6]
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=args.lr)
        self.schedulerD = torch.optim.lr_scheduler.MultiStepLR(self.optimizerD, milestones=self.indices, gamma=0.5)
        self.schedulerG = torch.optim.lr_scheduler.MultiStepLR(self.optimizerG, milestones=self.indices, gamma=0.5)
        logger.info(f"All model training parameters:\n"
                    f"\tIters is {args.iters}\n"
                    f"\tEpoch is {self.epochs}\n"
                    f"\tOptimizer is Adam\n"
                    f"\tLearning rate is {args.lr}\n"
                    f"\tBetas is (0.9, 0.999)\n"
                    f"\tScheduler is MultiStepLR")

        # We use VGG5.4 as our feature extraction method by default.
        self.vgg_criterion = VGGLoss().to(self.device)
        # Loss = 10 * l1 loss + vgg loss + 5e-3 * adversarial loss
        self.pix_criterion = nn.L1Loss().to(self.device)
        self.adversarial_criterion = nn.BCEWithLogitsLoss().to(self.device)
        logger.info(f"Loss function:\n"
                    f"\tVGG loss is VGGLoss\n"
                    f"\tPixel loss is L1\n"
                    f"\tAdversarial loss is BCEWithLogitsLoss")

    # Loading PSNR pre training model
    def resume_resnet(self):
        self.args.start_epoch = load_checkpoint(self.generator, self.psnr_optimizer,
                                                f"./weights/ResNet_{self.args.upscale_factor}x_checkpoint.pth")

    # Loading GAN checkpoint
    def resume_gan(self):
        self.args.start_epoch = load_checkpoint(self.discriminator, self.optimizerD,
                                                f"./weights/netD_{self.args.upscale_factor}x_checkpoint.pth")
        self.args.start_epoch = load_checkpoint(self.generator, self.optimizerG,
                                                f"./weights/netG_{self.args.upscale_factor}x_checkpoint.pth")

    def run(self):
        # Loading PSNR pre training model
        if self.args.resume_PSNR:
            logger.info("Load pre-training model parameters and weights")
            self.resume_resnet()

        # Pre-train generator using raw l1 loss.
        logger.info("Start training PSNR model based on L1 loss")
        # Save the generator model based on MSE pre training to speed up the training time
        if os.path.exists(f"./weights/ResNet_{self.args.upscale_factor}x.pth"):
            logger.info("Found PSNR pretrained model weights. Skip pre-train")
            self.generator.load_state_dict(torch.load(f"./weights/ResNet_{self.args.upscale_factor}x.pth",
                                                      map_location=self.device))
        else:
            # Writer train PSNR model log.
            if self.args.start_epoch == 0:
                with open(f"ResNet_{self.args.upscale_factor}x_Loss.csv", "w+") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Epoch", "Loss"])

            logger.warning("Not found pretrained weights. Start training PSNR model")
            for epoch in range(self.args.start_epoch, self.psnr_epochs):
                progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
                avg_loss = 0.
                for i, (input, target) in progress_bar:
                    # Set generator gradients to zero
                    self.generator.zero_grad()
                    # Generate data
                    lr = input.to(self.device)
                    hr = target.to(self.device)

                    # Generating fake high resolution images from real low resolution images.
                    sr = self.generator(lr)
                    # The MSE of the generated fake high-resolution image and real high-resolution image is calculated.
                    loss = self.pix_criterion(sr, hr)
                    # Calculate gradients for generator
                    loss.backward()
                    # Update generator weights
                    self.psnr_optimizer.step()

                    # Dynamic adjustment of learning rate
                    self.psnr_scheduler.step()

                    avg_loss += loss.item()

                    progress_bar.set_description(f"[{epoch + 1}/{self.psnr_epochs}][{i + 1}/{len(self.dataloader)}] "
                                                 f"Loss: {loss.item():.6f}")

                    # record iter.
                    total_iter = len(self.dataloader) * epoch + i

                    # The image is saved every 5000 iterations.
                    if (total_iter + 1) % self.args.save_freq == 0:
                        vutils.save_image(hr, os.path.join("./output/hr", f"ResNet_{total_iter + 1}.bmp"))
                        vutils.save_image(sr, os.path.join("./output/sr", f"ResNet_{total_iter + 1}.bmp"))

                # The model is saved every 1 epoch.
                torch.save({"epoch": epoch + 1,
                            "optimizer": self.psnr_optimizer.state_dict(),
                            "state_dict": self.generator.state_dict()
                            }, f"./weights/ResNet_{self.args.upscale_factor}x_checkpoint.pth")

                # Writer training log
                with open(f"ResNet_{self.args.upscale_factor}x_Loss.csv", "a+") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, avg_loss / len(self.dataloader)])

            torch.save(self.generator.state_dict(), f"./weights/ResNet_{self.args.upscale_factor}x.pth")
            logger.info(f"Training PSNR model done! Saving PSNR model weight to "
                        f"`./weights/ResNet_{self.args.upscale_factor}x.pth`")

            # After training the PSNR model, set the initial iteration to 0.
            self.args.start_epoch = 0

            # Loading GAN checkpoint
            if self.args.resume:
                logger.info("Load GAN model parameters and weights")
                self.resume_gan()

            # Train GAN model.
            logger.info("Staring training GAN model")
            logger.info(f"Training for {self.epochs} epochs")
            # Writer train GAN model log.
            if self.args.start_epoch == 0:
                with open(f"GAN_{self.args.upscale_factor}x_Loss.csv", "w+") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Epoch", "D Loss", "G Loss"])

            for epoch in range(self.args.start_epoch, self.epochs):
                # Set the all model to training mode
                logger.info("Switch discriminator model to train mode")
                self.discriminator.train()
                logger.info("Switch generator model to train mode")
                self.generator.train()

                progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
                g_avg_loss = 0.
                d_avg_loss = 0.
                for i, (input, target) in progress_bar:
                    lr = input.to(self.device)
                    hr = target.to(self.device)
                    batch_size = lr.size(0)
                    real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype, device=self.device)
                    fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype, device=self.device)

                    ##############################################
                    # (1) Update D network: maximize - E(hr)[1- log(D(hr, sr))] - E(sr)[log(D(sr, hr))]
                    ##############################################
                    # Set discriminator gradients to zero.
                    self.discriminator.zero_grad()

                    # Generate a super-resolution image
                    sr = self.generator(lr)

                    # Train with real high resolution image.
                    hr_output = self.discriminator(hr)  # Train real image.
                    sr_output = self.discriminator(sr.detach())  # No train fake image.
                    # Adversarial loss for real and fake images (relativistic average GAN)
                    errD_hr = self.adversarial_criterion(hr_output - torch.mean(sr_output), real_label)
                    errD_sr = self.adversarial_criterion(sr_output - torch.mean(hr_output), fake_label)
                    errD = (errD_sr + errD_hr) / 2
                    errD.backward()
                    D_x = hr_output.mean().item()
                    D_G_z1 = sr_output.mean().item()
                    self.optimizerD.step()

                    ##############################################
                    # (2) Update G network: maximize - E(hr)[log(D(hr, sr))] - E(sr)[1- log(D(sr, hr))]
                    ##############################################
                    # Set generator gradients to zero
                    self.generator.zero_grad()

                    # According to the feature map, the root mean square error is regarded as the content loss.
                    vgg_loss = self.vgg_criterion(sr, hr)
                    # Train with fake high resolution image.
                    hr_output = self.discriminator(hr.detach())  # No train real fake image.
                    sr_output = self.discriminator(sr)  # Train fake image.
                    # Adversarial loss (relativistic average GAN)
                    adversarial_loss = self.adversarial_criterion(sr_output - torch.mean(hr_output), real_label)
                    # Pixel level loss between two images.
                    l1_loss = self.pix_criterion(sr, hr)
                    errG = 10 * l1_loss + vgg_loss + 5e-3 * adversarial_loss
                    errG.backward()
                    D_G_z2 = sr_output.mean().item()
                    self.optimizerG.step()

                    # Dynamic adjustment of learning rate
                    self.schedulerD.step()
                    self.schedulerG.step()

                    d_avg_loss += errD.item()
                    g_avg_loss += errG.item()

                    progress_bar.set_description(f"[{epoch + 1}/{self.epochs}][{i + 1}/{len(self.dataloader)}] "
                                                 f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                                                 f"D(HR): {D_x:.6f} D(G(LR)): {D_G_z1:.6f}/{D_G_z2:.6f}")

                    # record iter.
                    total_iter = len(self.dataloader) * epoch + i

                    # The image is saved every 5000 iterations.
                    if (total_iter + 1) % self.args.save_freq == 0:
                        vutils.save_image(hr, os.path.join("./output/hr", f"ResNet_{total_iter + 1}.bmp"))
                        vutils.save_image(sr, os.path.join("./output/sr", f"ResNet_{total_iter + 1}.bmp"))

                # The model is saved every 1 epoch.
                torch.save({"epoch": epoch + 1,
                            "optimizer": self.optimizerD.state_dict(),
                            "state_dict": self.discriminator.state_dict()
                            }, f"./weights/netD_{self.args.upscale_factor}x_checkpoint.pth")
                torch.save({"epoch": epoch + 1,
                            "optimizer": self.optimizerG.state_dict(),
                            "state_dict": self.generator.state_dict()
                            }, f"./weights/netG_{self.args.upscale_factor}x_checkpoint.pth")

                # Writer training log
                with open(f"GAN_{self.args.upscale_factor}x_Loss.csv", "a+") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1,
                                     d_avg_loss / len(self.dataloader),
                                     g_avg_loss / len(self.dataloader)])

            torch.save(self.generator.state_dict(), f"./weights/GAN_{self.args.upscale_factor}x.pth")
            logger.info(f"Training GAN model done! Saving GAN model weight to "
                        f"`./weights/GAN_{self.args.upscale_factor}x.pth`")
