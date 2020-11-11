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
import math
import os
import random
import time

import PIL.BmpImagePlugin
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tensorboardX import SummaryWriter

__all__ = ["Logger", "calculate_weights_indices", "create_initialization_folder", "cubic",
           "imresize", "init_torch_seeds", "load_checkpoint", "opencv2pil", "pil2opencv", "select_device"]

logger = logging.getLogger(__name__)


# Source from "https://github.com/mit-han-lab/gan-compression/blob/master/utils/logger.py"
class Logger:
    def __init__(self, args):
        self.args = args
        self.log_file = open(os.path.join(args.log_dir, "log.txt"), "a")
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(args.tensorboard_dir)
        now = time.strftime("%c")
        self.log_file.write("================ (%s) ================\n" % now)
        self.log_file.flush()
        self.progress_bar = None

    def set_progress_bar(self, progress_bar):
        self.progress_bar = progress_bar

    def plot(self, items, step):
        if len(items) == 0:
            return
        for k, v in items.items():
            self.writer.add_scalar(k, v, global_step=step)
        self.writer.flush()

    def print_current_errors(self, epoch, i, errors, t):
        message = f"(epoch: {epoch}, iters: {i}, time: {t:.3f}) "

        for k, v in errors.items():
            if "Specific" in k:
                continue
            kk = k.split("/")[-1]
            message += f"{kk}: {v:.3f} "
        if self.progress_bar is None:
            print(message, flush=True)
        else:
            self.progress_bar.write(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def print_current_metrics(self, epoch, i, metrics, t):
        message = f"###(Evaluate epoch: {epoch}, iters: {i}, time: {t:.3f}) "

        for k, v in metrics.items():
            kk = k.split("/")[-1]
            message += f"{kk}: {v:.3f} "
        if self.progress_bar is None:
            print(message, flush=True)
        else:
            self.progress_bar.write(message)
        self.log_file.write(message + "\n")

    def print_info(self, message):
        if self.progress_bar is None:
            print(message, flush=True)
        else:
            self.progress_bar.write(message)
        self.log_file.write(message + "\n")


def calculate_weights_indices(in_length, out_length, scale, kernel_width, antialiasing):
    """Some operations of making data set. Reference from `https://github.com/xinntao/BasicSR`"""

    if (scale < 1) and antialiasing:
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def create_initialization_folder():
    try:
        os.makedirs("./output/lr")
        os.makedirs("./output/hr")
        os.makedirs("./output/sr")
        os.makedirs("weights")
    except OSError:
        pass


def cubic(x):
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (((absx > 1) * (absx <= 2)).type_as(absx))


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = "cubic"

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(in_H,
                                                                             out_H,
                                                                             scale,
                                                                             kernel_width,
                                                                             antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(in_W,
                                                                             out_W,
                                                                             scale,
                                                                             kernel_width,
                                                                             antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(weights_W[i])

    return torch.clamp(out_2, 0, 1)


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


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Adam = torch.optim.Adam,
                    file: str = None) -> int:
    r""" Quick loading model functions

    Args:
        model (nn.Module): Neural network model.
        optimizer (torch.optim): Model optimizer. (Default: torch.optim.Adam)
        file (str): Model file.

    Returns:
        How much epoch to start training from.
    """
    if os.path.isfile(file):
        logger.info(f"[*] Loading checkpoint `{file}`.")
        checkpoint = torch.load(file)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"[*] Loaded checkpoint `{file}` (epoch {checkpoint['epoch']})")
    else:
        logger.info(f"[!] no checkpoint found at '{file}'")
        epoch = 0

    return epoch


def opencv2pil(img: np.ndarray) -> PIL.BmpImagePlugin.BmpImageFile:
    """ OpenCV Convert to PIL.Image format.

    Returns:
        PIL.Image.
    """

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


def pil2opencv(img: PIL.BmpImagePlugin.BmpImageFile) -> np.ndarray:
    """ PIL.Image Convert to OpenCV format.

    Returns:
        np.ndarray.
    """

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img


def select_device(device: str = None, batch_size: int = 1) -> torch.device:
    r""" Choose the right equipment.

    Args:
        device (str): Use CPU or CUDA. (Default: None)
        batch_size (int, optional): Data batch size, cannot be less than the number of devices. (Default: 1).

    Returns:
        torch.device.
    """
    # device = "cpu" or "cuda:0,1,2,3"
    only_cpu = device.lower() == "cpu"
    if device and not only_cpu:  # if device requested other than "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable
        assert torch.cuda.is_available(), f"CUDA unavailable, invalid device {device} requested"

    cuda = False if only_cpu else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % gpu_count == 0, f"batch-size {batch_size} not multiple of GPU count {gpu_count}"
        x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
        s = "Using CUDA "
        for i in range(0, gpu_count):
            if i == 1:
                s = " " * len(s)
            logger.info(f"{s}\n\t+ device:{i} (name=`{x[i].name}`, "
                        f"total_memory={int(x[i].total_memory / c)}MB)")
    else:
        logger.info("Using CPU")

    logger.info("")  # skip a line
    return torch.device("cuda:0" if cuda else "cpu")
