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
import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    @staticmethod
    def forward(x):
        return x * F.relu6(x + 3, inplace=True) / 6


class HSigmoid(nn.Module):
    @staticmethod
    def forward(x):
        return F.relu6(x + 3, inplace=True) / 6
