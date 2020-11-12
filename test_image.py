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
from ssrgan.utils import create_folder
from tester import Estimate

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research and application of GAN based super resolution "
                                                 "technology for pathological microscopic images.")
    # model parameters
    parser.add_argument("-a", "--arch", metavar="ARCH", default="bionet",
                        choices=model_names,
                        help="model architecture: " +
                             " | ".join(model_names) +
                             " (default: bionet)")
    parser.add_argument("--upscale_factor", type=int, default=4, choices=[4],
                        help="Low to high resolution scaling factor. (default:4).")
    parser.add_argument("--model_path", default="./weights/GAN_4x.pth", type=str, metavar="PATH",
                        help="Path to latest checkpoint for model. (default: ``./weights/GAN_4x.pth``).")

    # test parameters
    parser.add_argument("--lr", type=str,
                        help="Test low resolution image name.")
    parser.add_argument("--hr", type=str,
                        help="Raw high resolution image name.")
    parser.add_argument("--outf", default="output", type=str, metavar="PATH",
                        help="The location of the image in the evaluation process. (default: ``output``).")
    parser.add_argument("--device", default="cpu",
                        help="device id i.e. `0` or `0,1` or `cpu`. (default: ``cpu``).")

    args = parser.parse_args()
    print(args)

    print("[*]Start test single image performance...")
    create_folder(args.outf)  # create evaluation directory.
    estimate = Estimate(args)
    estimate.run()
    print("[*]Test single image performance evaluation completed!")
