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
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

import ssrgan.models as models
from ssrgan.dataset import CustomTestDataset
from ssrgan.dataset import CustomTrainDataset
from ssrgan.loss import LPIPSLoss
from ssrgan.loss import VGGLoss
from ssrgan.models.discriminator import discriminator_for_vgg
from ssrgan.utils.common import AverageMeter
from ssrgan.utils.common import ProgressMeter
from ssrgan.utils.common import configure
from ssrgan.utils.common import create_folder
from ssrgan.utils.estimate import test

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG)

parser = argparse.ArgumentParser("Research on the technology of digital pathological image super-resolution.")
parser.add_argument("data", metavar="DIR",
                    help="Path to dataset")
parser.add_argument("-a", "--arch", metavar="ARCH", default="pmi_srgan",
                    choices=model_names,
                    help="Model architecture: " +
                         " | ".join(model_names) +
                         " (default: pmi_srgan)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="Number of data loading workers. (default: 4)")
parser.add_argument("--psnr-epochs", default=128, type=int, metavar="N",
                    help="Number of total psnr epochs to run. (default: 128)")
parser.add_argument("--start-psnr-epoch", default=0, type=int, metavar='N',
                    help="Manual psnr epoch number (useful on restarts). (default: 0)")
parser.add_argument("--gan-epochs", default=32, type=int, metavar="N",
                    help="Number of total gan epochs to run. (default: 32)")
parser.add_argument("--start-gan-epoch", default=0, type=int, metavar='N',
                    help="Manual gan epoch number (useful on restarts). (default: 0)")
parser.add_argument("-b", "--batch-size", default=4, type=int,
                    metavar="N",
                    help="Mini-batch size (default: 4), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--sampler-frequency", default=1, type=int, metavar="N",
                    help="If there are many datasets, this method can be used "
                         "to increase the number of epochs. (default:1)")
parser.add_argument("--psnr-lr", type=float, default=0.0004,
                    help="Learning rate for psnr-oral. (default: 0.0004)")
parser.add_argument("--gan-lr", type=float, default=0.0002,
                    help="Learning rate for gan-oral. (default: 0.0002)")
parser.add_argument("--image-size", type=int, default=256,
                    help="Image size of high resolution image. (default: 256)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[4],
                    help="Low to high resolution scaling factor. (default: 4)")
parser.add_argument("--model-path", default="", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model.")
parser.add_argument("--resume_psnr", default="", type=str, metavar="PATH",
                    help="Path to latest psnr-oral checkpoint.")
parser.add_argument("--resume_d", default="", type=str, metavar="PATH",
                    help="Path to latest -oral checkpoint.")
parser.add_argument("--resume_g", default="", type=str, metavar="PATH",
                    help="Path to latest psnr-oral checkpoint.")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="Use pre-trained model.")
parser.add_argument("--world-size", default=-1, type=int,
                    help="Number of nodes for distributed training.")
parser.add_argument("--rank", default=-1, type=int,
                    help="Node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://59.110.31.55:12345", type=str,
                    help="url used to set up distributed training. (default: tcp://59.110.31.55:12345)")
parser.add_argument("--dist-backend", default="nccl", type=str,
                    help="Distributed backend. (default: nccl)")
parser.add_argument("--seed", default=None, type=int,
                    help="Seed for initializing training.")
parser.add_argument("--gpu", default=None, type=int,
                    help="GPU id to use.")
parser.add_argument("--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                         "N processes per node, which has N GPUs. This is the "
                         "fastest way to use PyTorch for either single node or "
                         "multi node data parallel training.")

best_psnr = 0.0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed training. "
                      "This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! "
                      "You may see unexpected behavior when restarting "
                      "from checkpoints.")

    if args.gpu is not None:
        logger.warning("You have chosen a specific GPU. This will completely disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_psnr
    args.gpu = gpu

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for training.")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    # create model
    generator = configure(args)
    discriminator = discriminator_for_vgg(args.image_size)

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            discriminator.cuda(args.gpu)
            generator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            discriminator = nn.parallel.DistributedDataParallel(module=discriminator, device_ids=[args.gpu])
            generator = nn.parallel.DistributedDataParallel(module=generator, device_ids=[args.gpu])
        else:
            discriminator.cuda()
            generator.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            discriminator = nn.parallel.DistributedDataParallel(discriminator)
            generator = nn.parallel.DistributedDataParallel(generator)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
        generator = generator.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            discriminator.features = torch.nn.DataParallel(discriminator.features)
            generator.features = torch.nn.DataParallel(generator.features)
            discriminator.cuda()
            generator.cuda()
        else:
            discriminator = torch.nn.DataParallel(discriminator).cuda()
            generator = torch.nn.DataParallel(generator).cuda()

    # Loss = pixel loss + content loss + 0.1 * lpips loss + 0.001 * adversarial loss
    pixel_criterion = nn.L1Loss().cuda(args.gpu)
    content_criterion = VGGLoss().cuda(args.gpu)
    lpips_criterion = LPIPSLoss().cuda(args.gpu)
    adversarial_criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    logger.info(f"Losses function information:\n"
                f"\tPixel:       L1Loss\n"
                f"\tPerceptual:  VGG19_35th\n"
                f"\tAdversarial: BCEWithLogitsLoss\n"
                f"\tLPIPS:       VGG16")

    # All optimizer function and scheduler function.
    psnr_optimizer = torch.optim.Adam(generator.parameters(), lr=args.psnr_lr, betas=(0.9, 0.99))
    psnr_scheduler = torch.optim.lr_scheduler.ExponentialLR(psnr_optimizer, gamma=0.95)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.9, 0.99))
    discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=0.95)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.9, 0.99))
    generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=0.95)
    logger.info(f"Optimizer information:\n"
                f"\tPSNR learning rate:          {args.psnr_lr}\n"
                f"\tDiscriminator learning rate: {args.gan_lr}\n"
                f"\tGenerator learning rate:     {args.gan_lr}\n"
                f"\tPSNR optimizer:              Adam, [betas=(0.9,0.99)]\n"
                f"\tDiscriminator optimizer:     Adam, [betas=(0.9,0.99)]\n"
                f"\tGenerator optimizer:         Adam, [betas=(0.9,0.99)]\n"
                f"\tPSNR scheduler:              ExponentialLR, [gamma=0.95]\n"
                f"\tDiscriminator scheduler:     ExponentialLR, [gamma=0.95]\n"
                f"\tGenerator scheduler:         ExponentialLR, [gamma=0.95]")

    logger.info("Load training dataset")
    # Selection of appropriate treatment equipment.
    train_dataset = CustomTrainDataset(root=os.path.join(args.data, "train"), sampler_frequency=args.sampler_frequency)
    test_dataset = CustomTestDataset(root=os.path.join(args.data, "test"), image_size=args.image_size, sampler_frequency=args.sampler_frequency)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   pin_memory=True,
                                                   sampler=train_sampler,
                                                   num_workers=args.workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=args.workers)

    logger.info(f"Dataset information:\n"
                f"\tTrain Path:              {os.getcwd()}/{args.data}/train\n"
                f"\tTest Path:               {os.getcwd()}/{args.data}/test\n"
                f"\tNumber of train samples: {len(train_dataset)}\n"
                f"\tNumber of test samples:  {len(test_dataset)}\n"
                f"\tNumber of train batches: {len(train_dataloader)}\n"
                f"\tNumber of test batches:  {len(test_dataloader)}\n"
                f"\tShuffle of train:        True\n"
                f"\tShuffle of test:         False\n"
                f"\tSampler of train:        {bool(train_sampler)}\n"
                f"\tSampler of test:         None\n"
                f"\tWorkers of train:        {args.workers}\n"
                f"\tWorkers of test:         {args.workers}")

    # optionally resume from a checkpoint
    if args.resume_psnr:
        if os.path.isfile(args.resume_psnr):
            logger.info(f"Loading checkpoint '{args.resume_psnr}'.")
            if args.gpu is None:
                checkpoint = torch.load(args.resume_psnr)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume_psnr, map_location=f"cuda:{args.gpu}")
            args.start_psnr_epoch = checkpoint["epoch"]
            best_psnr = checkpoint["best_psnr"]
            if args.gpu is not None:
                # best_psnr may be from a checkpoint from a different GPU
                best_psnr = best_psnr.to(args.gpu)
            generator.load_state_dict(checkpoint["state_dict"])
            psnr_optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(f"Loaded checkpoint '{args.resume_psnr}' (epoch {checkpoint['epoch']}).")
        else:
            logger.info(f"No checkpoint found at '{args.resume_psnr}'.")

    if args.resume_d or args.resume_g:
        if os.path.isfile(args.resume_d) or os.path.isfile(args.resume_g):
            logger.info(f"Loading checkpoint '{args.resume_d}'.")
            logger.info(f"Loading checkpoint '{args.resume_g}'.")
            if args.gpu is None:
                checkpoint_d = torch.load(args.resume_d)
                checkpoint_g = torch.load(args.resume_g)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint_d = torch.load(args.resume_d, map_location=f"cuda:{args.gpu}")
                checkpoint_g = torch.load(args.resume_g, map_location=f"cuda:{args.gpu}")
            args.start_gan_epoch = checkpoint_g["epoch"]
            best_psnr = checkpoint_g["best_psnr"]
            if args.gpu is not None:
                # best_psnr may be from a checkpoint from a different GPU
                best_psnr = best_psnr.to(args.gpu)
            discriminator.load_state_dict(checkpoint_d["state_dict"])
            discriminator_optimizer.load_state_dict(checkpoint_d["optimizer"])
            generator.load_state_dict(checkpoint_g["state_dict"])
            generator_optimizer.load_state_dict(checkpoint_g["optimizer"])
            logger.info(f"Loaded checkpoint '{args.resume_d}' (epoch {checkpoint_d['epoch']}).")
            logger.info(f"Loaded checkpoint '{args.resume_g}' (epoch {checkpoint_g['epoch']}).")
        else:
            logger.info(f"No checkpoint found at '{args.resume_d}' or '{args.resume_g}'.")

    cudnn.benchmark = True

    # The mixed precision training is used in PSNR-oral.
    scaler = amp.GradScaler()
    logger.info("Turn on mixed precision training.")

    # Create a SummaryWriter at the beginning of training.
    psnr_writer = SummaryWriter(f"runs/{args.arch}_psnr_logs")
    gan_writer = SummaryWriter(f"runs/{args.arch}_gan_logs")

    logger.info(f"Train information:\n"
                f"\tPSNR-oral epochs: {args.psnr_epochs}\n"
                f"\tGAN-oral epochs:  {args.gan_epochs}")

    for epoch in range(args.start_psnr_epoch, args.psnr_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_psnr(train_dataloader, generator, pixel_criterion, psnr_optimizer, epoch, scaler, psnr_writer, args)

        psnr_scheduler.step()

        # Test for every epoch.
        psnr, ssim, lpips, gmsd = test(generator, test_dataloader, args.gpu)
        gan_writer.add_scalar("Test/PSNR", psnr, epoch + 1)
        gan_writer.add_scalar("Test/SSIM", ssim, epoch + 1)
        gan_writer.add_scalar("Test/LPIPS", lpips, epoch + 1)
        gan_writer.add_scalar("Test/GMSD", gmsd, epoch + 1)

        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            torch.save({"epoch": epoch + 1,
                        "arch": args.arch,
                        "best_psnr": best_psnr,
                        "state_dict": generator.state_dict(),
                        "optimizer": psnr_optimizer.state_dict(),
                        }, os.path.join("weights", f"PSNR_epoch{epoch}.pth"))
            if is_best:
                torch.save(generator.state_dict(), os.path.join("weights", f"PSNR.pth"))

    # Load best model weight.
    generator.load_state_dict(torch.load(os.path.join("weights", f"PSNR.pth"), map_location=f"cuda:{args.gpu}"))

    for epoch in range(args.start_gan_epoch, args.gan_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_gan(train_dataloader, discriminator, generator, pixel_criterion, content_criterion, lpips_criterion, adversarial_criterion,
                  discriminator_optimizer, generator_optimizer, epoch, gan_writer, scaler, args)

        discriminator_scheduler.step()
        generator_scheduler.step()

        # Test for every epoch.
        psnr, ssim, lpips, gmsd = test(generator, test_dataloader, args.gpu)
        gan_writer.add_scalar("Test/PSNR", psnr, epoch + 1)
        gan_writer.add_scalar("Test/SSIM", ssim, epoch + 1)
        gan_writer.add_scalar("Test/LPIPS", lpips, epoch + 1)
        gan_writer.add_scalar("Test/GMSD", gmsd, epoch + 1)

        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            torch.save({"epoch": epoch + 1,
                        "arch": "vgg",
                        "state_dict": discriminator.state_dict(),
                        "optimizer": discriminator_optimizer.state_dict()
                        }, os.path.join("weights", f"Discriminator_epoch{epoch}.pth"))
            torch.save({"epoch": epoch + 1,
                        "arch": args.arch,
                        "best_psnr": best_psnr,
                        "state_dict": generator.state_dict(),
                        "optimizer": generator_optimizer.state_dict()
                        }, os.path.join("weights", f"Generator_epoch{epoch}.pth"))
            if is_best:
                torch.save(generator.state_dict(), os.path.join("weights", f"GAN.pth"))


def train_psnr(train_dataloader: torch.utils.data.DataLoader,
               generator: nn.Module,
               pixel_criterion: nn.L1Loss,
               psnr_optimizer: torch.optim.Adam,
               epoch: int,
               scaler: amp.GradScaler,
               writer: SummaryWriter,
               args: argparse.ArgumentParser.parse_args):
    batch_time = AverageMeter("Time", ":6.4f")
    l1_losses = AverageMeter("L1 Loss", ":.6f")
    progress = ProgressMeter(len(train_dataloader), [batch_time, l1_losses], prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    generator.train()

    end = time.time()
    for i, (lr, hr) in enumerate(train_dataloader):
        # Move data to special device.
        if args.gpu is not None:
            lr = lr.cuda(args.gpu, non_blocking=True)
            hr = hr.cuda(args.gpu, non_blocking=True)

        psnr_optimizer.zero_grad()

        with amp.autocast():
            sr = generator(lr)
            loss = pixel_criterion(sr, hr)

        scaler.scale(loss).backward()
        scaler.step(psnr_optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        l1_losses.update(loss.item(), lr.size(0))

        iters = i + epoch * len(train_dataloader) + 1
        writer.add_scalar("Train/L1 Loss", loss.item(), iters)

        # Output results every 100 batches.
        if i % 100 == 0:
            progress.display(i)

        # Save image every 1000 batches.
        if iters % 1000 == 0:
            vutils.save_image(hr.detach(), os.path.join("runs", "hr", f"PSNR_{iters}.bmp"))
            vutils.save_image(sr.detach(), os.path.join("runs", "sr", f"PSNR_{iters}.bmp"))


def train_gan(train_dataloader: torch.utils.data.DataLoader,
              discriminator: nn.Module,
              generator: nn.Module,
              pixel_criterion: nn.L1Loss,
              content_criterion: VGGLoss,
              adversarial_criterion: nn.BCEWithLogitsLoss,
              lpips_criterion: LPIPSLoss,
              discriminator_optimizer: torch.optim.Adam,
              generator_optimizer: torch.optim.Adam,
              epoch: int,
              scaler: amp.GradScaler,
              writer: SummaryWriter,
              args: argparse.ArgumentParser.parse_args):
    batch_time = AverageMeter("Time", ":.4f")
    d_losses = AverageMeter("D Loss", ":.6f")
    g_losses = AverageMeter("G Loss", ":.6f")
    pixel_losses = AverageMeter("L1 Loss", ":6.4f")
    content_losses = AverageMeter("Content Loss", ":6.4f")
    lpips_losses = AverageMeter("LPIPS Loss", ":6.4f")
    adversarial_losses = AverageMeter("Adversarial Loss", ":6.4f")

    progress = ProgressMeter(
        len(train_dataloader),
        [batch_time,
         d_losses, g_losses,
         pixel_losses, content_losses, lpips_losses, adversarial_losses],
        prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    discriminator.train()
    generator.train()

    end = time.time()
    for i, (lr, hr) in enumerate(train_dataloader):
        # Move data to special device.
        if args.gpu is not None:
            lr = lr.cuda(args.gpu, non_blocking=True)
            hr = hr.cuda(args.gpu, non_blocking=True)
        batch_size = lr.size(0)

        # The real sample label is 1, and the generated sample label is 0.
        real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype).cuda(args.gpu, non_blocking=True)
        fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype).cuda(args.gpu, non_blocking=True)

        ##############################################
        # (1) Update D network: E(hr)[fake(C(D(hr) - E(sr)C(sr)))] + E(sr)[fake(C(fake) - E(real)C(real))]
        ##############################################
        discriminator_optimizer.zero_grad()

        with amp.autocast():
            # Generating fake high resolution images from real low resolution images.
            sr = generator(lr)

            real_output = discriminator(hr)
            fake_output = discriminator(sr.detach())
            # Adversarial loss for real and fake images (relativistic average GAN)
            d_loss_real = adversarial_criterion(real_output - torch.mean(fake_output), real_label)
            d_loss_fake = adversarial_criterion(fake_output - torch.mean(real_output), fake_label)
            # Count all discriminator losses.
            d_loss = d_loss_fake + d_loss_real

        scaler.scale(d_loss).backward()
        scaler.step(discriminator_optimizer)
        scaler.update()

        ##############################################
        # (2) Update G network: E(hr)[sr(C(D(hr) - E(sr)C(sr)))] + E(sr)[sr(C(fake) - E(real)C(real))]
        ##############################################
        # Set generator gradients to zero.
        generator_optimizer.zero_grad()

        with amp.autocast():
            pixel_loss = pixel_criterion(sr, hr.detach())
            content_loss = content_criterion(sr, hr.detach())
            real_output = discriminator(hr.detach())
            fake_output = discriminator(sr)
            lpips_loss = lpips_criterion(sr, hr.detach())
            # Adversarial loss for real and fake images (relativistic average GAN)
            adversarial_loss = adversarial_criterion(fake_output - torch.mean(real_output), real_label)
            # Count all generator losses.
            g_loss = 1 * pixel_loss + 1 * content_loss + 0.1 * lpips_loss + 0.001 * adversarial_loss

        scaler.scale(d_loss).backward()
        scaler.step(discriminator_optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        d_losses.update(d_loss.item(), lr.size(0))
        g_losses.update(g_loss.item(), lr.size(0))
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        lpips_losses.update(lpips_loss.item(), lr.size(0))

        iters = i + epoch * len(train_dataloader) + 1
        writer.add_scalar("Train/D Loss", d_loss.item(), iters)
        writer.add_scalar("Train/G Loss", g_loss.item(), iters)
        writer.add_scalar("Train/Pixel Loss", pixel_loss.item(), iters)
        writer.add_scalar("Train/Content Loss", content_loss.item(), iters)
        writer.add_scalar("Train/Adversarial Loss", adversarial_loss.item(), iters)
        writer.add_scalar("Train/LPIPS Loss", lpips_loss.item(), iters)

        # Output results every 100 batches.
        if i % 100 == 0:
            progress.display(i)

        # Save image every 1000 batches.
        if iters % 1000 == 0:
            vutils.save_image(hr.detach(), os.path.join("runs", "hr", f"PSNR_{iters}.bmp"))
            vutils.save_image(sr.detach(), os.path.join("runs", "sr", f"PSNR_{iters}.bmp"))


if __name__ == "__main__":
    print("##################################################\n")
    print("Run Training Engine.\n")

    create_folder("runs")
    create_folder("runs/hr")
    create_folder("runs/sr")
    create_folder("weights")

    logger.info("TrainingEngine:")
    print("\tAPI version .......... 0.1.0")
    print("\tBuild ................ 2021.04.02")
    print("##################################################\n")
    main()
    logger.info("All training has been completed successfully.\n")
