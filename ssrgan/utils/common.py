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
import logging
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import ssrgan.models as models
from .device import select_device

__all__ = [
    "configure", "create_folder", "get_time", "inference", "init_torch_seeds", "load_checkpoint"
]

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def configure(args):
    """Global profile.

    Args:
        args (argparse.ArgumentParser.parse_args): Use argparse library parse command.
    """
    # Selection of appropriate treatment equipment
    device = select_device(args.device, batch_size=1)

    # Create model
    if args.pretrained:
        logger.info(f"Using pre-trained model `{args.arch}`")
        model = models.__dict__[args.arch](pretrained=True, upscale_factor=args.upscale_factor).to(device)
    else:
        logger.info(f"Creating model `{args.arch}`")
        model = models.__dict__[args.arch](upscale_factor=args.upscale_factor).to(device)
        if args.model_path:
            logger.info(f"You loaded the specified weight. Load weights from `{args.model_path}`")
            model.load_state_dict(torch.load(args.model_path, map_location=device))

    return model, device


def create_folder(folder):
    try:
        os.makedirs(folder)
        logger.info(f"Create `{os.path.join(os.getcwd(), folder)}` directory successful.")
    except OSError:
        logger.warning(f"Directory `{os.path.join(os.getcwd(), folder)}` already exists!")
        pass


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def inference(model, lr, statistical_time=False):
    r"""General inference method.

    Args:
        model (nn.Module): Neural network model.
        lr (Torch.Tensor): Picture in pytorch format (N*C*H*W).
        statistical_time (optional, bool): Is reasoning time counted. (default: ``False``).

    Returns:
        super resolution image, time consumption of super resolution image (if `statistical_time` set to `True`).
    """
    # Set eval model.
    model.eval()

    if statistical_time:
        start_time = time.time()
        with torch.no_grad():
            sr = model(lr)
        use_time = time.time() - start_time
        return sr, use_time
    else:
        with torch.no_grad():
            sr = model(lr)
        return sr


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

    logger.info("Initialize random seed.")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Adam = torch.optim.Adam, file: str = None) -> int:
    r""" Quick loading model functions

    Args:
        model (nn.Module): Neural network model.
        optimizer (torch.optim): Model optimizer. (Default: torch.optim.Adam).
        file (str): Model file. (default: None).

    Returns:
        How much epoch to start training from.
    """
    if os.path.isfile(file):
        logger.info(f"Loading checkpoint `{file}`")
        checkpoint = torch.load(file)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"Loaded checkpoint `{file}` (epoch {checkpoint['epoch'] + 1})")
    else:
        logger.warning(f"No checkpoint found at `{file}`")
        epoch = 0

    return epoch
