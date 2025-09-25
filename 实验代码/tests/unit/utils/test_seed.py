from __future__ import annotations

import os
import random

import numpy as np
import torch

from hs_mad.utils.seed import SeedConfig, seed_everything


def test_seed_everything_reproducible():
    seed_everything(SeedConfig(seed=123, deterministic=True))
    values_1 = (
        random.random(),
        np.random.rand(),
        torch.randn(2, 3),
    )

    seed_everything(123)
    values_2 = (
        random.random(),
        np.random.rand(),
        torch.randn(2, 3),
    )

    assert values_1[0] == values_2[0]
    assert values_1[1] == values_2[1]
    assert torch.allclose(values_1[2], values_2[2])
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_seed_everything_extra_offset():
    base = SeedConfig(seed=10, deterministic=False, cudnn_benchmark=True)
    seed = seed_everything(base, extra_offset=5)
    assert seed == 15
    assert torch.backends.cudnn.benchmark
