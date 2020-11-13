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

import ssrgan.models as models
from tester import Video

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description="Research and application of GAN based super resolution "
                                             "technology for pathological microscopic images.")
# model parameters
parser.add_argument("-a", "--arch", metavar="ARCH", default="bionet",
                    choices=model_names,
                    help="model architecture: " +
                         " | ".join(model_names) +
                         " (default: bionet)")
parser.add_argument("--upscale-factor", type=int, default=4, choices=[4],
                    help="Low to high resolution scaling factor. (default:4).")

parser.add_argument("--file", type=str, required=True,
                    help="Test low resolution video name.")
parser.add_argument("--model-path", default="./weights/GAN_4x.pth", type=str, metavar="PATH",
                    help="Path to latest checkpoint for model. (default: ``./weights/GAN_4x.pth``).")
parser.add_argument("--device", default="0",
                    help="device id i.e. `0` or `0,1` or `cpu`. (default: ``0``).")
parser.add_argument("--view", action="store_true",
                    help="Super resolution real time to show.")

args = parser.parse_args()
print(args)

if __name__ == "__main__":
    video = Video(args)
    video.run()
    print("Video super resolution complete!")
