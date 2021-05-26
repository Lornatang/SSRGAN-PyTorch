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
import random
import time

import torch
import torch.backends.cudnn as cudnn

import ssrgan.models as models

# Find all available models.
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG)


def main(args):
    # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set convolution algorithm.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build a super-resolution model, if model_ If path is defined, the specified model weight will be loaded.
    model = models.__dict__[args.arch]()
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

    # Create an image that conforms to the normal distribution.
    data = torch.randn([1, 3, args.image_size, args.image_size], requires_grad=False)

    # If there is a GPU, the data will be loaded into the GPU memory.
    if args.gpu is not None:
        data = data.cuda(args.gpu, non_blocking=True)

    # It only needs to reconstruct the low resolution image without the gradient information of the reconstructed image.
    with torch.no_grad():
        start = time.time()
        _ = model(data)
        # Waits for all kernels in all streams on a CUDA device to complete.
        torch.cuda.synchronize()
        print(f"Time:{(time.time() - start) * 1000:.2f}ms.")

    # Context manager that manages autograd profiler state and holds a summary of results.
    with torch.autograd.profiler.profile(enabled=True, use_cuda=args.gpu, record_shapes=False, profile_memory=False) as profile:
        _ = model(data)
    print(profile.table())
    # Open Chrome browser and enter in the address bar `chrome://tracing`
    profile.export_chrome_trace("profile.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", metavar="ARCH", default="pmigan",
                        choices=model_names,
                        help="Model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `pmigan`)")
    parser.add_argument("-i", "--image-size", type=int, default=54,
                        help="Image size of low-resolution. (Default: 54)")
    parser.add_argument("--seed", default=666, type=int,
                        help="Seed for initializing training. (Default: 666)")
    parser.add_argument("--gpu", default=None, type=int,
                        help="GPU id to use.")
    args = parser.parse_args()

    logger.info("ScriptEngine:")
    logger.info("\tAPI version .......... 0.1.4")
    logger.info("\tBuild ................ 2021.05.26\n")

    main(args)
