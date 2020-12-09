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
import csv
import logging
import math
import os
import time
from typing import Any

import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

import ssrgan.models as models
from ssrgan import CustomTestDataset
from ssrgan import CustomTrainDataset
from ssrgan import VGGLoss
from ssrgan.models import DiscriminatorForVGG
from ssrgan.utils import AverageMeter
from ssrgan.utils import ProgressMeter
from ssrgan.utils import configure
from ssrgan.utils import init_torch_seeds
from ssrgan.utils import save_checkpoint

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def resume(model: nn.Module, optimizer: torch.optim.Adam, device: torch.device,
           args: argparse.ArgumentParser.parse_args) -> [int, float]:
    r""" Resume your last training schedule.

    Args:
        model (nn.Module): Neural network model.
        optimizer (torch.optim.Adam): Model optimizer.
        device (torch.device): Load data to specified device.
        args (argparse.ArgumentParser.parse_args): Parsing command line parameters.
    """
    # At present, it supports the simultaneous loading of two models.
    models = [args.resumeD, args.resumeG]
    start_epochs, best_values = [], []
    for index in range(len(models)):
        if os.path.isfile(models[index]):
            logger.info(f"Loading checkpoint '{os.path.basename(models[index])}'.")
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(models[index], map_location=device)
            start_epoch = checkpoint["epoch"]
            # The optimization index of discriminator is different from that of generator.
            best_value = checkpoint["best_value"]
            # best_loss may be from a checkpoint from a different GPU.
            best_value = best_value.to(device)
            # Support transfer learning.
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"Loaded checkpoint '{os.path.basename(models[index])}' (epoch {checkpoint['epoch']}).")
        else:
            start_epoch = 0
            best_value = 0.
            logger.info(f"No checkpoint found at '{os.path.basename(models[index])}'")

        start_epochs.append(start_epoch)
        best_values.append(best_value)

    return start_epochs[0], start_epochs[1], best_values[0], best_values[1]


def train_psnr(dataloader: torch.utils.data.DataLoader, epoch: int, model: nn.Module, criterion: nn.L1Loss,
               optimizer: torch.optim, scheduler: torch.optim.lr_scheduler, device: torch.device,
               args: argparse.ArgumentParser.parse_args) -> Any:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.6f")
    progress = ProgressMeter(len(dataloader), [batch_time, data_time, losses], prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Move data to special device.
        lr = images.to(device)
        hr = target.to(device)

        # Generating fake high resolution images from real low resolution images.
        sr = model(lr)
        # The L1 Loss of the generated fake high-resolution image and real high-resolution image is calculated.
        loss = criterion(sr, hr)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Dynamic adjustment of learning rate.
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def test_psnr(dataloader: torch.utils.data.DataLoader, epoch: int, model: nn.Module, criterion: nn.MSELoss,
              device: torch.device, args: argparse.ArgumentParser.parse_args) -> float:
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.6f")
    psnres = AverageMeter("PSNR", ":2.2f")
    progress = ProgressMeter(len(dataloader), [batch_time, losses, psnres], prefix="Test: ")

    # switch to evaluate mode.
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, _, target) in enumerate(dataloader):

            # Move data to special device.
            lr = images.to(device)
            hr = target.to(device)

            # Generating fake high resolution images from real low resolution images.
            sr = model(lr)
            # The MSE of the generated fake high-resolution image and real high-resolution image is calculated.
            loss = criterion(sr, hr)
            psnr = 10 * math.log10(1. / loss.item())

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            psnres.update(psnr, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    # Last saved image of test.
    vutils.save_image(hr, os.path.join("./output/hr", f"ResNet_{epoch + 1}.bmp"))
    vutils.save_image(sr, os.path.join("./output/sr", f"ResNet_{epoch + 1}.bmp"))

    # Writer evaluation log
    with open(f"ResNet_{args.upscale_factor}x_{args.arch}.csv", "a+") as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, psnres.avg])

    # TODO: this should also be done with the ProgressMeter.
    logger.info(f" * PSNR: {psnres.avg:.2f}.")

    return psnres.avg


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.manualSeed)

        logger.info("Load training dataset")
        # Selection of appropriate treatment equipment.
        self.train_dataloader = torch.utils.data.DataLoader(CustomTrainDataset(f"{args.dataroot}/train"),
                                                            batch_size=args.batch_size,
                                                            pin_memory=True,
                                                            num_workers=int(args.workers))
        self.test_dataloader = torch.utils.data.DataLoader(CustomTestDataset(f"{args.dataroot}/test"),
                                                           batch_size=args.batch_size,
                                                           pin_memory=True,
                                                           num_workers=int(args.workers))

        logger.info(f"Train Dataset information:\n"
                    f"\tTrain Dataset dir is `{os.getcwd()}/{args.dataroot}/train`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")
        logger.info(f"Test Dataset information:\n"
                    f"\tTest Dataset dir is `{os.getcwd()}/{args.dataroot}/test`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")

        # Construct network architecture model of generator and discriminator.
        self.generator, self.device = configure(args)
        logger.info(f"Creating discriminator model")
        self.discriminator = DiscriminatorForVGG().to(self.device)

        # Parameters of pre training model.
        self.psnr_epochs = int(args.psnr_iters // len(self.train_dataloader))
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
        self.epochs = int(args.iters // len(self.train_dataloader))
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
        # Evaluating the loss function of PSNR.
        self.mse_criterion = nn.MSELoss().to(self.device)
        logger.info(f"Loss function:\n"
                    f"\tVGG loss is VGGLoss\n"
                    f"\tPixel loss is L1\n"
                    f"\tAdversarial loss is BCEWithLogitsLoss")

    def run(self):
        args = self.args
        best_loss = 0.
        best_psnr = 0.

        # Loading PSNR pre training model.
        if args.resumeG or args.resumeD:
            args.start_epoch, args.start_epoch, best_loss, best_psnr = resume(model=self.generator,
                                                                              optimizer=self.psnr_optimizer,
                                                                              device=self.device,
                                                                              args=self.args)

        # Start train PSNR model.
        logger.info("Staring training PSNR model")
        logger.info(f"Training for {self.psnr_epochs} epochs")
        # Writer train PSNR model log.
        if self.args.start_epoch == 0:
            with open(f"ResNet_{self.args.upscale_factor}x_{args.arch}.csv", "w+") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "PSNR"])
        for epoch in range(args.start_epoch, self.psnr_epochs):
            train_psnr(dataloader=self.train_dataloader,
                       epoch=epoch,
                       model=self.generator,
                       criterion=self.pix_criterion,
                       optimizer=self.psnr_optimizer,
                       scheduler=self.psnr_scheduler,
                       device=self.device,
                       args=self.args)

            psnr = test_psnr(dataloader=self.test_dataloader,
                             epoch=epoch,
                             model=self.generator,
                             criterion=self.mse_criterion,
                             device=self.device,
                             args=self.args)

            # remember best psnr and save checkpoint
            is_best = psnr > best_psnr
            best_psnr = max(psnr, best_psnr)

            # The model is saved every 1 epoch.
            save_checkpoint({"epoch": epoch + 1, "state_dict": self.generator.state_dict(), "best_value": best_psnr,
                             "optimizer": self.psnr_optimizer.state_dict()}, is_best,
                            f"./weights/ResNet_{args.upscale_factor}x_{args.arch}_checkpoint.pth",
                            f"./weights/Exp/Activations/baseline.pth")

        # # Loading GAN training all model.
        # if args.resumed and args.resumeG:
        #     args.start_epoch, best_loss = self.resume_discriminator(model=self.discriminator,
        #                                                             optimizer=self.optimizerG,
        #                                                             device=self.device)
        #     args.start_epoch, best_psnr = self.resume_generator(model=self.generator,
        #                                                         optimizer=self.optimizerD,
        #                                                         device=self.device)

        # # Writer train GAN model log.
        # if self.args.start_epoch == 0:
        #     with open(f"GAN_{self.args.upscale_factor}x_Loss.csv", "w+") as f:
        #         writer = csv.writer(f)
        #         writer.writerow(["Epoch", "D Loss", "G Loss"])
        #
        # for epoch in range(self.args.start_epoch, self.epochs):
        #     progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        #     avg_d_loss = 0.
        #     avg_g_loss = 0.
        #     for i, (input, target) in progress_bar:
        #         lr = input.to(self.device)
        #         hr = target.to(self.device)
        #         batch_size = lr.size(0)
        #         real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype, device=self.device)
        #         fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype, device=self.device)
        #
        #         ##############################################
        #         # (1) Update D network: maximize - E(hr)[1- log(D(hr, sr))] - E(sr)[log(D(sr, hr))]
        #         ##############################################
        #         # Set discriminator gradients to zero.
        #         self.discriminator.zero_grad()
        #
        #         # Generate a super-resolution image
        #         sr = self.generator(lr)
        #
        #         # Train with real high resolution image.
        #         hr_output = self.discriminator(hr)  # Train real image.
        #         sr_output = self.discriminator(sr.detach())  # No train fake image.
        #         # Adversarial loss for real and fake images (relativistic average GAN)
        #         errD_hr = self.adversarial_criterion(hr_output - torch.mean(sr_output), real_label)
        #         errD_sr = self.adversarial_criterion(sr_output - torch.mean(hr_output), fake_label)
        #         errD = (errD_sr + errD_hr) / 2
        #         errD.backward()
        #         D_x = hr_output.mean().item()
        #         D_G_z1 = sr_output.mean().item()
        #         self.optimizerD.step()
        #
        #         ##############################################
        #         # (2) Update G network: maximize - E(hr)[log(D(hr, sr))] - E(sr)[1- log(D(sr, hr))]
        #         ##############################################
        #         # Set generator gradients to zero
        #         self.generator.zero_grad()
        #
        #         # According to the feature map, the root mean square error is regarded as the content loss.
        #         vgg_loss = self.vgg_criterion(sr, hr)
        #         # Train with fake high resolution image.
        #         hr_output = self.discriminator(hr.detach())  # No train real fake image.
        #         sr_output = self.discriminator(sr)  # Train fake image.
        #         # Adversarial loss (relativistic average GAN)
        #         adversarial_loss = self.adversarial_criterion(sr_output - torch.mean(hr_output), real_label)
        #         # Pixel level loss between two images.
        #         l1_loss = self.pix_criterion(sr, hr)
        #         errG = 10 * l1_loss + vgg_loss + 5e-3 * adversarial_loss
        #         errG.backward()
        #         D_G_z2 = sr_output.mean().item()
        #         self.optimizerG.step()
        #
        #         # Dynamic adjustment of learning rate
        #         self.schedulerD.step()
        #         self.schedulerG.step()
        #
        #         avg_d_loss += errD.item()
        #         avg_g_loss += errG.item()
        #
        #         progress_bar.set_description(f"[{epoch + 1}/{self.epochs}][{i + 1}/{len(self.dataloader)}] "
        #                                      f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
        #                                      f"D(HR): {D_x:.6f} D(G(LR)): {D_G_z1:.6f}/{D_G_z2:.6f}")
        #
        #         # record iter.
        #         total_iter = len(self.dataloader) * epoch + i
        #
        #         # The image is saved every 5000 iterations.
        #         if (total_iter + 1) % self.args.save_freq == 0:
        #             vutils.save_image(hr, os.path.join("./output/hr", f"GAN_{total_iter + 1}.bmp"))
        #             vutils.save_image(sr, os.path.join("./output/sr", f"GAN_{total_iter + 1}.bmp"))
        #
        #     # The model is saved every 1 epoch.
        #     torch.save({"epoch": epoch + 1,
        #                 "state_dict": self.discriminator.state_dict(),
        #                 "best_loss": avg_d_loss,
        #                 "optimizer": self.optimizerD.state_dict()
        #                 }, f"./weights/netD_{self.args.upscale_factor}x_checkpoint.pth")
        #     torch.save({"epoch": epoch + 1,
        #                 "state_dict": self.generator.state_dict(),
        #                 "best_loss": avg_g_loss,
        #                 "optimizer": self.optimizerG.state_dict()
        #                 }, f"./weights/netG_{self.args.upscale_factor}x_checkpoint.pth")
        #
        #     # Writer training log
        #     with open(f"GAN_{self.args.upscale_factor}x_Loss.csv", "a+") as f:
        #         writer = csv.writer(f)
        #         writer.writerow([epoch + 1,
        #                          avg_d_loss / len(self.dataloader),
        #                          avg_g_loss / len(self.dataloader)])
        #
        # torch.save(self.generator.state_dict(), f"./weights/GAN_{self.args.upscale_factor}x.pth")
        # logger.info(f"Training GAN model done! Saving GAN model weight to "
        #             f"`./weights/GAN_{self.args.upscale_factor}x.pth`")
