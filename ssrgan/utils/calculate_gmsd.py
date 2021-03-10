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
import numpy as np
from numba import jit

__all__ = [
    "convolution", "gmsd"
]


# Implement convolution operation.
@jit
def convolution(image, kernal_size):
    new_arr = kernal_size.reshape(kernal_size.size)
    new_arr = new_arr[::-1]
    kernal_size = new_arr.reshape(kernal_size.shape)

    cor_height = image.shape[0] - kernal_size.shape[0] + 1
    cor_width = image.shape[1] - kernal_size.shape[1] + 1
    result = np.zeros((cor_height, cor_width), dtype=np.float64)
    for i in range(cor_height):
        for j in range(cor_width):
            result[i][j] = (image[i:i + kernal_size.shape[0], j:j + kernal_size.shape[1]] * kernal_size).sum()
    return result


def gmsd(source, target, c=170):
    # prewitt operator.
    hx = np.array([[1 / 3, 0, -1 / 3]] * 3, dtype=np.float64)
    # Mean filter kernel.
    ave_filter = np.array([[0.25, 0.25], [0.25, 0.25]])
    hy = hx.transpose()
    # Mean filter kernel.
    ave_source = convolution(source, ave_filter)
    ave_target = convolution(target, ave_filter)
    # Sample down.
    ave_source_down = ave_source[np.arange(0, ave_source.shape[0], 2), :]
    ave_source_down = ave_source_down[:, np.arange(0, ave_source_down.shape[1], 2)]
    ave_target_down = ave_target[np.arange(0, ave_target.shape[0], 2), :]
    ave_target_down = ave_target_down[:, np.arange(0, ave_target_down.shape[1], 2)]
    # Calculate the intermediate variables such as mr md
    mr_sq = convolution(ave_target_down, hx) ** 2 + convolution(ave_target_down, hy) ** 2
    md_sq = convolution(ave_source_down, hx) ** 2 + convolution(ave_source_down, hy) ** 2
    mr = np.sqrt(mr_sq)
    md = np.sqrt(md_sq)
    gms = (2 * mr * md + c) / (mr_sq + md_sq + c)
    gmsm = np.mean(gms)
    gmsd_value = np.mean((gms - gmsm) ** 2)
    return gmsd_value
