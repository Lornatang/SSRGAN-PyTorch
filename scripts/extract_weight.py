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
import argparse
import logging
import random

import torch

import ssrgan.models as models

# Find all available models.
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def main(args):
    # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set convolution algorithm.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build a super-resolution model, if model_ If path is defined, the specified model weight will be loaded.
    model = models.__dict__[args.arch]()
    # Switch model to eval mode.
    model.eval()
    # Load trained model weight.
    model.load_state_dict(torch.load(args.model_path)["state_dict"])
    # Save simplified model weights.
    torch.save(model.state_dict(), "Generator.pth")
    logger.info("Model convert done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--arch", metavar="ARCH", default="pmigan",
                        choices=model_names,
                        help="Model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `pmigan`)")
    parser.add_argument("--model-path", type=str, metavar="PATH", required=True, help="Path to latest checkpoint for model.")
    parser.add_argument("--seed", default=666, type=int, help="Seed for initializing training. (Default: 666)")
    args = parser.parse_args()

    logger.info("ScriptEngine:")
    logger.info("\tAPI version .......... 0.1.4")
    logger.info("\tBuild ................ 2021.05.26\n")

    main(args)
