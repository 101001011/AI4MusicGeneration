from __future__ import annotations

import torch

from hs_mad.modules.hse.conformer import ConformerConfig, ConformerStack


def test_conformer_stack_forward():
    cfg = ConformerConfig(dim=32, ffn_mult=2.0, num_heads=4, conv_kernel=7, dropout=0.0, drop_path=0.0)
    stack = ConformerStack(cfg, num_layers=2)
    x = torch.randn(3, 16, 32)
    y = stack(x)
    assert y.shape == x.shape


def test_conformer_with_drop_path():
    cfg = ConformerConfig(dim=16, drop_path=0.5)
    stack = ConformerStack(cfg, num_layers=4)
    stack.train()
    x = torch.randn(2, 10, 16)
    y = stack(x)
    assert y.shape == x.shape
