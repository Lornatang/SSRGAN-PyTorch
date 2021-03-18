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
import logging
import os

import torch

import ssrgan.models as models

__all__ = [
    "create_folder", "configure"
]

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def create_folder(folder):
    try:
        os.makedirs(folder)
        logger.info(f"Create `{os.path.join(os.getcwd(), folder)}` directory successful.")
    except OSError:
        logger.warning(f"Directory `{os.path.join(os.getcwd(), folder)}` already exists!")
        pass


def configure(args):
    """Global profile.

    Args:
        args (argparse.ArgumentParser.parse_args): Use argparse library parse command.
    """

    # Create model
    if args.pretrained:
        logger.info(f"Using pre-trained model `{args.arch}`.")
        model = models.__dict__[args.arch](pretrained=True)
    else:
        logger.info(f"Creating model `{args.arch}`.")
        model = models.__dict__[args.arch]()
    if args.model_path:
        logger.info(f"You loaded the specified weight. Load weights from `{os.path.abspath(args.model_path)}`.")
        model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")), strict=False)

    return model
