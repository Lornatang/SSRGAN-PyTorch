# SSRGAN

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Research and application of GAN based super resolution technology for pathological microscopic images](http://www.dakewe.com/).

### Table of contents

1. [About Research and application of GAN based super resolution technology for pathological microscopic images](#about-research-and-application-of-gan-based-super-resolution-technology-for-pathological-microscopic-images)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-pmigan)
    * [Download dataset](#download-dataset)
4. [Test](#test)
    * [Test benchmark](#test-benchmark)
    * [Test image](#test-image)
    * [Test video](#test-video)
    * [Test model performance](#test-model-performance)
5. [Train](#train-eg-div2k)
6. [Contributing](#contributing)
7. [Credit](#credit)

### About Research and application of GAN based super resolution technology for pathological microscopic images

If you're new to SSRGAN, here's an abstract straight from the paper:

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central
problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of
optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on
minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking
high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this
paper, we present SSRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable
of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an
adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is
trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by
perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily
downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SSRGAN.
The MOS scores obtained with SSRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art
method.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates
images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is
a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/SSRGAN.git
$ cd SSRGAN/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g pmigan)

```bash
$ cd weights/
$ python3 download_weights.py
```

#### Download dataset

```bash
$ cd data/
$ bash download_dataset.sh
```

### Test

#### Test benchmark

```text
usage: test_benchmark.py [-h] [-a {esrgan16,esrgan23,pmigan,rfb,rfb_4x4,srgan}] [-j WORKERS] [-b BATCH_SIZE] [--sampler-frequency SAMPLER_FREQUENCY] [--image-size IMAGE_SIZE] [--upscale-factor {4}] [--model-path MODEL_PATH] [--pretrained]
                         [--seed SEED] [--gpu GPU]
                         DIR

positional arguments:
  DIR                   Path to dataset.

optional arguments:
  -h, --help            show this help message and exit
  -a {esrgan16,esrgan23,pmigan,rfb,rfb_4x4,srgan}, --arch {esrgan16,esrgan23,pmigan,rfb,rfb_4x4,srgan}
                        Model architecture: esrgan16 | esrgan23 | pmigan | rfb | rfb_4x4 | srgan. (Default: `pmigan`)
  -j WORKERS, --workers WORKERS
                        Number of data loading workers. (Default: 4)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size of the dataset. (Default: 64)
  --sampler-frequency SAMPLER_FREQUENCY
                        If there are many datasets, this method can be used to increase the number of epochs. (Default: 1)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (Default: 216)
  --upscale-factor {4}  Low to high resolution scaling factor. Optional: [4]. (Default: 4)
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  
# Example
$ python3 test_benchmark.py --arch pmigan --pretrained --gpu 0 [image-folder with train and val folders]
```

#### Test image

```text
usage: test_image.py [-h] [--arch {esrgan16,esrgan23,pmigan,rfb,rfb_4x4,srgan}] --lr LR [--hr HR] [--upscale-factor {4}] [--model-path MODEL_PATH] [--pretrained] [--seed SEED] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --arch {esrgan16,esrgan23,pmigan,rfb,rfb_4x4,srgan}
                        Model architecture: esrgan16 | esrgan23 | pmigan | rfb | rfb_4x4 | srgan. (Default: `pmigan`)
  --lr LR               Test low resolution image name.
  --hr HR               Raw high resolution image name.
  --upscale-factor {4}  Low to high resolution scaling factor. Optional: [4]. (Default: 4)
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.

# Example
$ python3 test_image.py --arch pmigan --lr [path-to-lr-image] --hr [Optional, path-to-hr-image] --pretrained --gpu 0
```

#### Test video

```text
usage: test_video.py [-h] [--arch {esrgan16,esrgan23,pmigan,rfb,rfb_4x4,srgan}] --file FILE [--upscale-factor {4}] [--model-path MODEL_PATH] [--pretrained] [--seed SEED] [--gpu GPU] [--view]

optional arguments:
  -h, --help            show this help message and exit
  --arch {esrgan16,esrgan23,pmigan,rfb,rfb_4x4,srgan}
                        Model architecture: esrgan16 | esrgan23 | pmigan | rfb | rfb_4x4 | srgan. (Default: `pmigan`)
  --file FILE           Test low resolution video name.
  --upscale-factor {4}  Low to high resolution scaling factor. Optional: [4]. (Default: 4)
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  --view                Do you want to show SR video synchronously.
 
# Example
$ python3 test_video.py --arch pmigan --file [path-to-video] --pretrained --gpu 0 --view 
```

#### Test model performance

TODO

Low resolution / Recovered High Resolution / Ground Truth

<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g DIV2K)

```text
usage: train.py [-h] [--arch {esrgan16,esrgan23,pmigan,rfb,rfb_4x4,srgan}] [-j WORKERS] [--psnr-epochs PSNR_EPOCHS] [--start-psnr-epoch START_PSNR_EPOCH] [--gan-epochs GAN_EPOCHS] [--start-gan-epoch START_GAN_EPOCH] [-b BATCH_SIZE]
                [--sampler-frequency SAMPLER_FREQUENCY] [--psnr-lr PSNR_LR] [--gan-lr GAN_LR] [--image-size IMAGE_SIZE] [--upscale-factor {4}] [--netD NETD] [--netG NETG] [--pretrained] [--world-size WORLD_SIZE] [--rank RANK]
                [--dist-url DIST_URL] [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU] [--multiprocessing-distributed]
                DIR

positional arguments:
  DIR                   Path to dataset.

optional arguments:
  -h, --help            show this help message and exit
  --arch {esrgan16,esrgan23,pmigan,rfb,rfb_4x4,srgan}
                        Model architecture: esrgan16 | esrgan23 | pmigan | rfb | rfb_4x4 | srgan. (Default: pmigan)
  -j WORKERS, --workers WORKERS
                        Number of data loading workers. (Default: 4)
  --psnr-epochs PSNR_EPOCHS
                        Number of total psnr epochs to run. (Default: 64)
  --start-psnr-epoch START_PSNR_EPOCH
                        Manual psnr epoch number (useful on restarts). (Default: 0)
  --gan-epochs GAN_EPOCHS
                        Number of total gan epochs to run. (Default: 32)
  --start-gan-epoch START_GAN_EPOCH
                        Manual gan epoch number (useful on restarts). (Default: 0)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size of the dataset. (Default: 4)
  --sampler-frequency SAMPLER_FREQUENCY
                        If there are many datasets, this method can be used to increase the number of epochs. (Default:1)
  --psnr-lr PSNR_LR     Learning rate for psnr-oral. (Default: 0.0004)
  --gan-lr GAN_LR       Learning rate for gan-oral. (Default: 0.0002)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (Default: 216)
  --upscale-factor {4}  Low to high resolution scaling factor. Optional: [4]. (Default: 4)
  --netD NETD           Path to Discriminator checkpoint.
  --netG NETG           Path to Generator checkpoint.
  --pretrained          Use pre-trained model.
  --world-size WORLD_SIZE
                        Number of nodes for distributed training.
  --rank RANK           Node rank for distributed training. (Default: -1)
  --dist-url DIST_URL   url used to set up distributed training. (Default: `tcp://59.110.31.55:12345`)
  --dist-backend DIST_BACKEND
                        Distributed backend. (Default: `nccl`)
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training.

# Example (e.g DIV2K)
$ python3 train.py --arch pmigan --gpu 0 [image-folder with train and val folders]
# Multi-processing Distributed Data Parallel Training
$ python3 train.py --arch pmigan --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [image-folder with train and val folders]
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py --arch pmigan --start-psnr-epoch 10 --netG weights/PSNR_epoch10.pth --gpu 0 [image-folder with train and val folders] 
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Research and application of GAN based super resolution technology for pathological microscopic images

_Changyu Liu, Qiyue Yu, Bo Wang, Yang Wang, Yahong Liu, Lanjing Xiao_ <br>

**Abstract** <br>
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central
problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of
optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on
minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking
high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this
paper, we present SSRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable
of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an
adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is
trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by
perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily
downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SSRGAN.
The MOS scores obtained with SSRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art
method.

[[Paper]](http://www.dakewe.com/)

```
@InProceedings{ssrgan,
    author = {Changyu Liu, Qiyue Yu, Bo Wang, Yang Wang, Riliang Wu, Yahong Liu, Rundong Chen, Lanjing Xiao},
    title = {Research and application of GAN based super resolution technology for pathological microscopic images},
    booktitle = {-},
    year = {2021}
}
```
