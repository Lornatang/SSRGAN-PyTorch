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
import time

import torch
from ptflops import get_model_complexity_info

import ssrgan.models as models
from ssrgan.utils import select_device

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research and application of GAN based super resolution "
                                                 "technology for pathological microscopic images.")
    # model parameters
    parser.add_argument("-a", "--arch", metavar="ARCH", default="dsgan",
                        choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             " (Default: ``dsgan``)")
    parser.add_argument("-b", "--batch-size", type=int, default=12,
                        help="When calculating full memory inference speed. (Default: 12).")
    parser.add_argument("-i", "--image-size", type=int, default=64,
                        help="Image size of sample. (Default: 64).")
    args = parser.parse_args()

    cpu_device = select_device("cpu")
    cuda_device = select_device("1")

    image_size = (3, args.image_size, args.image_size)
    batch_size = args.batch_size
    inputs = torch.randn(batch_size, 3, args.image_size, args.image_size)

    # Create cpu model and gpu model.
    model = models.__dict__[args.arch]()

    # Cal cpu forward time.
    cpu_data = inputs.to(cpu_device)
    cpu_model = model.to(cpu_device)
    start_time = time.time()
    _ = cpu_model(cpu_data)
    cpu_time = time.time() - start_time

    # Cal gpu forward time.
    cuda_data = inputs.to(cuda_device)
    cuda_model = model.to(cuda_device)
    start_time = time.time()
    _ = cuda_model(cuda_data)
    cuda_time = time.time() - start_time

    flops, params = get_model_complexity_info(model, image_size, as_strings=True, print_per_layer_stat=False)
    print(f"|---------------------------------------------------------------------|")
    print(f"|                               Summary                               |")
    print(f"|---------------------------------------------------------------------|")
    print(f"|       Model       |    Params   |   FLOPs   | CPU Speed | GPU SPeed |")
    print(f"|---------------------------------------------------------------------|")
    print(f"|{cpu_model.__class__.__name__.center(19):19}"
          f"|{params.center(13):13}"
          f"|{flops.center(11):11}"
          f"|  {int(1 / cpu_time * args.batch_size)} it/s  "
          f"|  {int(1 / cuda_time * args.batch_size)} it/s  |")
    print(f"|---------------------------------------------------------------------|")
