"""Skip-connection modulation (SCM) decoder block."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import GatedTemporalCrossAttention, StandardCrossAttention
from .blocks import AdaLNModulator, ResBlock1D

__all__ = ["SCMDecoderBlock"]


class SCMDecoderBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_skip: int,
        c_out: int,
        cond_dim: int,
        time_dim: int,
        style_dim: int,
        n_heads: int,
        enable_scm: bool = True,
    ) -> None:
        super().__init__()
        self.enable_scm = enable_scm
        if enable_scm:
            self.gated_tca = GatedTemporalCrossAttention(query_dim=c_skip, kv_dim=cond_dim, n_heads=n_heads)
        else:
            self.gated_tca = None
        self.fuse = nn.Conv1d(c_in + c_skip, c_out, kernel_size=1)
        self.resblock = ResBlock1D(c_out, c_out, time_dim)
        self.adaln = AdaLNModulator(c_out, style_dim, time_dim)
        self.cond_proj = nn.Linear(cond_dim, c_out)
        self.cross_attn = StandardCrossAttention(query_dim=c_out, kv_dim=c_out, n_heads=n_heads)

    def forward(
        self,
        z_in: torch.Tensor,
        z_skip: torch.Tensor,
        cond: torch.Tensor,
        time_emb: torch.Tensor,
        style_vec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # SCM modulation on skip connection
        if self.enable_scm and self.gated_tca is not None:
            z_skip_mod = z_skip + self.gated_tca(z_skip.transpose(1, 2), cond).transpose(1, 2)
        else:
            z_skip_mod = z_skip
        z = torch.cat([z_in, z_skip_mod], dim=1)
        z = self.fuse(z)
        z = self.resblock(z, time_emb)
        z = self.adaln(z, style_vec, time_emb)
        cond_proj = self.cond_proj(cond)
        z = z + self.cross_attn(z.transpose(1, 2), cond_proj).transpose(1, 2)
        return z
