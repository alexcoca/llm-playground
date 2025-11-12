import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # ok if no cuda


def ensure_determinism(seed: int):

    logger.info("Setting seed")
    set_seed(seed)

    logger.info(
        "Setting debug environment for CUDA to ensure determinism and avoiding"
        " non-deterministic ops"
    )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    logger.info(f"CUDA_WORK_SPACE_CONFIG: {os.environ.get('CUDA_WORK_SPACE_CONFIG')}")
    torch.use_deterministic_algorithms(True)
    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
