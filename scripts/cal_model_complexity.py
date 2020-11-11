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

from ptflops import get_model_complexity_info

import ssrgan.models as models
from ssrgan.utils import select_device

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
print(model_names)

parser = argparse.ArgumentParser(description="Research and application of GAN based super resolution "
                                             "technology for pathological microscopic images.")
# model parameters
parser.add_argument("-a", "--arch", metavar="ARCH", default="bionet",
                    choices=model_names,
                    help="model architecture: " +
                         " | ".join(model_names) +
                         " (default: bionet)")
args = parser.parse_args()

device = select_device("cpu")
model = models.__dict__[args.arch]().to(device)

size = (3, 54, 54)
flops, params = get_model_complexity_info(model, size, as_strings=True, print_per_layer_stat=True)
print(f"                   Summary                     ")
print(f"-----------------------------------------------")
print(f"|       Model       |    Params   |   FLOPs   |")
print(f"-----------------------------------------------")
print(f"|{model.__class__.__name__.center(19):19}|{params.center(13):13}|{flops.center(11):11}|")
print(f"-----------------------------------------------")
