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
from ptflops import get_model_complexity_info

from ssrgan_pytorch.models import BioNet
from ssrgan_pytorch.utils import select_device

device = select_device("0")

model = BioNet().to(device)

size = (3, 54, 54)
flops, params = get_model_complexity_info(model, size, as_strings=True, print_per_layer_stat=True)
print(f"                   Summary                     ")
print(f"-----------------------------------------------")
print(f"|       Model       |    Params   |   FLOPs   |")
print(f"-----------------------------------------------")
print(f"|{model.__class__.__name__.center(19):19}|{params.center(13):13}|{flops.center(11):11}|")
print(f"-----------------------------------------------")
