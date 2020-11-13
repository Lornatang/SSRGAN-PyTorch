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
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.models as models

from .device import select_device


def configure(args):
    """Global profile.

    Args:
        args (argparse.ArgumentParser.parse_args): Use argparse library parse command.
    """
    # Selection of appropriate treatment equipment
    device = select_device(args.device, batch_size=1)

    # Construct GAN model.
    model = models.__dict__[args.arch](upscale_factor=args.upscale_factor).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    return model, device


def create_folder(folder):
    try:
        os.makedirs(folder)
    except OSError:
        print(f"[!]`{os.path.join(os.getcwd(), folder)}` already exists!")
        pass


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


# Source from "https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py"
def init_torch_seeds(seed: int = 0):
    r""" Sets the seed for generating random numbers. Returns a

    Args:
        seed (int): The desired seed.
    """

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
