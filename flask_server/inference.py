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
import logging

from engine import Inference

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super resolution model inference engine.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Required. Super resolution image file name to be processed.")
    parser.add_argument("-m", "--model", type=str, default="resources/srgan.pth",
                        help="Optional. Super resolution weights path. (Default: ``resources/srgan.pth``).")
    parser.add_argument("-d", "--device", type=str, default="0",
                        help="device id i.e. `0` or `0,1` or `cpu`. (default: ``0``).")
    args = parser.parse_args()

    print("##################################################\n")
    print("Run Inference Engine.\n")
    print(args)

    logger.info("InferenceEngine:")
    print("\tAPI version .......... 0.1.1")
    print("\tBuild ................ 2020.11.30-1116-0c5adc7e")

    logger.info("Creating Inference Engine")
    inference = Inference(args)

    logger.info("Staring inference...")
    inference.run()
    print("##################################################\n")

    logger.info("Inference completed successfully.\n")
