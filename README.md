# SSRGAN

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Research and application of GAN based super resolution technology for pathological microscopic images](http://www.dakewe.com/).

### Table of contents
1. [About Research and application of GAN based super resolution technology for pathological microscopic images](#about-research-and-application-of-gan-based-super-resolution-technology-for-pathological-microscopic-images)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-bionet)
    * [Download dataset](#download-dataset)
4. [Script](#script)
    * [Computational model complexity](#computational-model-complexity)
4. [Test](#test)
    * [Basic test](#basic-test)
    * [Test benchmark](#test-benchmark)
    * [Test video](#test-video)
4. [Train](#train-eg-div2k)
5. [Contributing](#contributing) 
6. [Credit](#credit)

### About Research and application of GAN based super resolution technology for pathological microscopic images

If you're new to SSRGAN, here's an abstract straight from the paper:

Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and 
deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover 
the finer texture details when we super-resolve at large upscaling factors? The behavior of 
optimization-based super-resolution methods is principally driven by the choice of the objective function. 
Recent work has largely focused on minimizing the mean squared reconstruction error. 
The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and 
are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. 
In this paper, we present SSRGAN, a generative adversarial network (GAN) for image super-resolution (SR). 
To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. 
To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. 
The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained 
to differentiate between the super-resolved images and original photo-realistic images. In addition, 
we use a content loss motivated by perceptual similarity instead of similarity in pixel space. 
Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. 
An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SSRGAN. 
The MOS scores obtained with SSRGAN are closer to those of the original high-resolution images than to those obtained 
with any state-of-the-art method.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. 
It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is 
a discriminant network that discriminates whether an image is real. The input is x, x is a picture, 
and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, 
and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/SSRGAN.git
$ cd SSRGAN/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g bionet)

```bash
$ cd weights/
$ python3 download_weights.py
```

#### Download dataset

```bash
$ cd data/
$ bash download_dataset.sh
```

### Script

#### Computational model complexity

```text
$ python3 scripts/cal_model_complexity.py
                   Summary                     
-----------------------------------------------
|       Model       |    Params   |   FLOPs   |
-----------------------------------------------
|       BioNet      |     0.27M   |  2.19GMac |
|       ESRGAN      |    16.92M   | 52.35GMac |
|     Inception     |     0.88M   |  7.10GMac |
|    MobileNetV1    |     0.35M   |  4.31GMac |
|    MobileNetV2    |     1.78M   | 11.37GMac |
|    MobileNetV3    |     3.98M   | 12.82GMac |
|     RFBESRGAN     |    21.31M   | 66.49GMac |
|    ShuffleNetV1   |     0.22M   |  3.70GMac |
|    ShuffleNetV2   |     0.25M   |  3.87GMac |
|     SqueezeNet    |     0.87M   |  6.93GMac |
|       SRGAN       |     1.54M   |  6.46GMac |
|        UNet       |     2.36M   |  8.87GMac |
-----------------------------------------------
```

### Test

Using pre training model to generate pictures.

#### Basic test

```text
usage: test_image.py [-h] --lr LR --hr HR [--outf PATH] [--device DEVICE]
                     [--detail] [-a ARCH] [--upscale-factor {4}]
                     [--model-path PATH] [--pretrained]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  --lr LR               Test low resolution image name.
  --hr HR               Raw high resolution image name.
  --outf PATH           The location of the image in the evaluation process.
                        (default: ``test``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``cpu``).
  --detail              Use comprehensive assessment.
  -a ARCH, --arch ARCH  model architecture: bionet | esrgan |
                        get_upsample_filter | inception | lapsrn | mobilenetv1
                        | mobilenetv2 | mobilenetv3 | rfb_esrgan |
                        shufflenetv1 | shufflenetv2 | squeezenet | srgan |
                        unet (default: bionet)
  --upscale-factor {4}  Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default: ````).
  --pretrained          Use pre-trained model.

# Example
$ python3 test_image.py -a bionet --pretrained --lr <path>/<to>/lr.png --hr <path>/<to>/hr.png 
```

#### Test benchmark

```text
usage: test_benchmark.py [-h] [--dataroot DATAROOT] [-j N] [--outf PATH]
                         [--device DEVICE] [--detail] [-a ARCH]
                         [--upscale-factor {4}] [--model-path PATH]
                         [--pretrained] [-b N]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --outf PATH           The location of the image in the evaluation process.
                        (default: ``test``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default: ````).
  --detail              Use comprehensive assessment.
  -a ARCH, --arch ARCH  model architecture: bionet | esrgan |
                        get_upsample_filter | inception | lapsrn | mobilenetv1
                        | mobilenetv2 | mobilenetv3 | rfb_esrgan |
                        shufflenetv1 | shufflenetv2 | squeezenet | srgan |
                        unet (default: bionet)
  --upscale-factor {4}  Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default: ````).
  --pretrained          Use pre-trained model.
  -b N, --batch-size N  mini-batch size (default: 16), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.

# Example
$ python3 test_benchmark.py -a bionet --pretrained
```

#### Test video

```text
usage: test_video.py [-h] --file FILE [--outf PATH] [--device DEVICE] [--view]
                     [-a ARCH] [--upscale-factor {4}] [--model-path PATH]
                     [--pretrained]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE           Test low resolution video name.
  --outf PATH           The location of the image in the evaluation process.
                        (default: ``video``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``0``).
  --view                Super resolution real time to show.
  -a ARCH, --arch ARCH  model architecture: bionet | esrgan |
                        get_upsample_filter | inception | lapsrn | mobilenetv1
                        | mobilenetv2 | mobilenetv3 | rfb_esrgan |
                        shufflenetv1 | shufflenetv2 | squeezenet | srgan |
                        unet (default: bionet)
  --upscale-factor {4}  Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default: ````).
  --pretrained          Use pre-trained model.

# Example
$ python3 test_video.py -a bionet --pretrained --file <path>/<to>/video.mp4
```

Low resolution / Recovered High Resolution / Ground Truth
<span align="center"><img src="assets/result.png" alt="">
</span>

### Train (e.g DIV2K)

```text
usage: train.py [-h] [--dataroot DATAROOT] [-j N] [--manualSeed MANUALSEED]
                [--device DEVICE] [--save-freq SAVE_FREQ] [-a ARCH]
                [--upscale-factor {4}] [--model-path PATH] [--pretrained]
                [--resume-PSNR] [--resume] [--start-epoch N] [--psnr-iters N]
                [--iters N] [-b N] [--psnr-lr PSNR_LR] [--lr LR]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   Path to datasets. (default:`./data`)
  -j N, --workers N     Number of data loading workers. (default:4)
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:1111)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default: ````).
  --save-freq SAVE_FREQ
                        frequency of evaluating and save the model.
  -a ARCH, --arch ARCH  model architecture: bionet | esrgan |
                        get_upsample_filter | inception | lapsrn | mobilenetv1
                        | mobilenetv2 | mobilenetv3 | rfb_esrgan |
                        shufflenetv1 | shufflenetv2 | squeezenet | srgan |
                        unet (default: bionet)
  --upscale-factor {4}  Low to high resolution scaling factor. (default:4).
  --model-path PATH     Path to latest checkpoint for model. (default: ````).
  --pretrained          Use pre-trained model.
  --resume-PSNR         Path to latest checkpoint for PSNR model.
  --resume              Path to latest checkpoint for Generator.
  --start-epoch N       manual epoch number (useful on restarts)
  --psnr-iters N        The number of iterations is needed in the training of
                        PSNR model. (default:1e6)
  --iters N             The training of srgan model requires the number of
                        iterations. (default:4e5)
  -b N, --batch-size N  mini-batch size (default: 8), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --psnr-lr PSNR_LR     Learning rate for PSNR model. (default:2e-4)
  --lr LR               Learning rate. (default:1e-4)

# Example (e.g DIV2K)
$ python3 train.py -a bionet
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py --resume-PSNR
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 

### Credit

#### Research and application of GAN based super resolution technology for pathological microscopic images
_Changyu Liu, Qiyue Yu, Bo Wang, Yang Wang, Riliang Wu, Yahong Liu, Rundong Chen, Lanjing Xiao_ <br>

**Abstract** <br>
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and 
deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover 
the finer texture details when we super-resolve at large upscaling factors? The behavior of 
optimization-based super-resolution methods is principally driven by the choice of the objective function. 
Recent work has largely focused on minimizing the mean squared reconstruction error. 
The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and 
are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. 
In this paper, we present SSRGAN, a generative adversarial network (GAN) for image super-resolution (SR). 
To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. 
To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. 
The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained 
to differentiate between the super-resolved images and original photo-realistic images. In addition, 
we use a content loss motivated by perceptual similarity instead of similarity in pixel space. 
Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. 
An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SSRGAN. 
The MOS scores obtained with SSRGAN are closer to those of the original high-resolution images than to those obtained 
with any state-of-the-art method.

[[Paper]](http://www.dakewe.com/)

```
@InProceedings{ssrgan,
    author = {Changyu Liu, Qiyue Yu, Bo Wang, Yang Wang, Riliang Wu, Yahong Liu, Rundong Chen, Lanjing Xiao},
    title = {Research and application of GAN based super resolution technology for pathological microscopic images},
    booktitle = {-},
    year = {2020}
}
```
