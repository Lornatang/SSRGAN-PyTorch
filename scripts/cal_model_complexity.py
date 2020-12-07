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
    parser.add_argument("-a", "--arch", metavar="ARCH", default="bionet",
                        choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             " (default: bionet)")
    args = parser.parse_args()

    inputs = torch.randn(16, 3, 54, 54)

    # Create cpu model and cpu data.
    cpu_device = select_device("cpu")
    cpu_data = inputs.to(cpu_device)
    cpu_model = models.__dict__[args.arch]().to(cpu_device)

    # Create gpu model and gpu data.
    cuda_device = select_device("0")
    cuda_data = inputs.to(cuda_device)
    cuda_model = models.__dict__[args.arch]().to(cuda_device)

    size = (3, 54, 54)
    flops, params = get_model_complexity_info(cpu_model, size, as_strings=True, print_per_layer_stat=True)

    # Cal cpu forward time.
    start_time = time.time()
    _ = cpu_model(cpu_data)
    cpu_time = time.time() - start_time

    # Cal gpu forward time.
    start_time = time.time()
    _ = cuda_model(cuda_data)
    cuda_time = time.time() - start_time
    print(f"                               Summary                                 ")
    print(f"-----------------------------------------------------------------------")
    print(f"|       Model       |    Params   |   FLOPs   |CPU Latency|GPU Latency|")
    print(f"-----------------------------------------------------------------------")
    print(f"|{cpu_model.__class__.__name__.center(19):19}"
          f"|{params.center(13):13}"
          f"|{flops.center(11):11}"
          f"|  {(cpu_time * 1000):.2f}ms "
          f"|  {(cuda_time * 1000):.2f}ms |")
    print(f"-----------------------------------------------------------------------")
