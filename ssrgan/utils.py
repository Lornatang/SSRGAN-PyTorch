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
import time

import torch

__all__ = [
    "inference", "load_checkpoint"
]


def inference(model, lr, statistical_time=False):
    r"""General inference method.

    Args:
        model (nn.Module): Neural network model.
        lr (Torch.Tensor): Picture in pytorch format (N*C*H*W).
        statistical_time (optional, bool): Is reasoning time counted. (Default: ``False``).

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


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Adam = torch.optim.Adam, file: str = None) -> int:
    r""" Quick loading model functions

    Args:
        model (nn.Module): Neural network model.
        optimizer (torch.optim): Model optimizer. (Default: torch.optim.Adam).
        file (str): Model file. (Default: None).

    Returns:
        How much epoch to start training from.
    """
    if os.path.isfile(file):
        print(f"[*] Loading checkpoint `{file}`.")
        checkpoint = torch.load(file)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[*] Loaded checkpoint `{file}` (epoch {checkpoint['epoch']})")
    else:
        print(f"[!] no checkpoint found at '{file}'")
        epoch = 0

    return epoch
