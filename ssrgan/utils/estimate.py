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
import cv2
import lpips
import torch
from sewar.full_ref import mse
from sewar.full_ref import msssim
from sewar.full_ref import psnr
from sewar.full_ref import rmse
from sewar.full_ref import sam
from sewar.full_ref import ssim
from sewar.full_ref import vifp

from .calculate_niqe import niqe
from .transform import opencv2tensor

__all__ = [
    "image_quality_evaluation"
]


def image_quality_evaluation(sr_filename: str, hr_filename: str, detail: bool = False, device: torch.device = "cpu"):
    """Image quality evaluation function.

    Args:
        sr_filename (str): Image file name after super resolution.
        hr_filename (str): Original high resolution image file name.
        detail (optional, bool): Is there a detailed assessment. (Default: ``False``)
        device (optional, torch.device): Selection of data processing equipment in PyTorch. (Default: ``cpu``).

    Returns:
        If the `simple` variable is set to ``False`` return `mse, rmse, psnr, ssim, msssim, niqe, sam, vifp, lpips`,
        else return `psnr, ssim`.
    """
    # Reference sources from `https://github.com/richzhang/PerceptualSimilarity`
    lpips_loss = lpips.LPIPS(net="vgg", verbose=False).to(device)
    # Evaluate performance
    sr = cv2.imread(sr_filename)
    hr = cv2.imread(hr_filename)

    # For LPIPS evaluation
    sr_tensor = opencv2tensor(sr, device)
    hr_tensor = opencv2tensor(hr, device)

    if detail:
        # Complete estimate.
        mse_value = mse(sr, hr)
        rmse_value = rmse(sr, hr)
        psnr_value = psnr(sr, hr)
        ssim_value = ssim(sr, hr)
        msssim_value = msssim(sr, hr)
        niqe_value = niqe(sr_filename)
        sam_value = sam(sr, hr)
        vifp_value = vifp(sr, hr)
        lpips_value = lpips_loss(sr_tensor, hr_tensor)
        return mse_value, rmse_value, psnr_value, ssim_value, msssim_value, niqe_value, sam_value, vifp_value, lpips_value
    else:
        psnr_value = psnr(sr, hr)
        ssim_value = ssim(sr, hr)
        return psnr_value, ssim_value
