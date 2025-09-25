from __future__ import annotations

import torch

from hs_mad.modules.unet.attention import GatedTemporalCrossAttention, StandardCrossAttention


def test_standard_cross_attention_shapes():
    attn = StandardCrossAttention(query_dim=32, kv_dim=64, n_heads=4)
    q = torch.randn(2, 10, 32)
    kv = torch.randn(2, 10, 64)
    out = attn(q, kv)
    assert out.shape == q.shape


def test_gated_attention_zero_init():
    attn = GatedTemporalCrossAttention(query_dim=32, kv_dim=64, n_heads=4, zero_init=True)
    q = torch.randn(2, 10, 32)
    kv = torch.randn(2, 10, 64)
    out = attn(q, kv)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
    out.mean().backward()
    assert attn.out_proj.weight.grad is not None
