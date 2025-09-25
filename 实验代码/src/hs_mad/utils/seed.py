"""Seed utilities for deterministic HS-MAD experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

__all__ = ["SeedConfig", "seed_everything"]


@dataclass
class SeedConfig:
    """Configuration for reproducibility controls.

    Attributes:
        seed: Base integer seed applied to Python, NumPy, and PyTorch.
        deterministic: Enable deterministic algorithms in PyTorch at the cost of speed.
        cudnn_benchmark: Whether to enable CuDNN benchmarking. Must be False when deterministic.
    """

    seed: int = 42
    deterministic: bool = True
    cudnn_benchmark: bool = False


def seed_everything(config: SeedConfig | int, *, extra_offset: Optional[int] = None) -> int:
    """Seed all relevant RNGs.

    Args:
        config: Seed configuration or raw integer seed.
        extra_offset: Optional offset applied to the base seed (useful in DDP ranks).

    Returns:
        The final integer seed applied to all backends.
    """

    if isinstance(config, int):
        config = SeedConfig(seed=config)

    seed = config.seed if extra_offset is None else config.seed + extra_offset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if config.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = config.cudnn_benchmark

    return seed
