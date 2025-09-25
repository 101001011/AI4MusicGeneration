"""Attention modules for MRCI and SCM."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["StandardCrossAttention", "GatedTemporalCrossAttention"]


class StandardCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if query_dim % n_heads != 0:
            raise ValueError("query_dim must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # query: [B, T_q, C_q], key_value: [B, T_k, C_k]
        b, t_q, _ = query.shape
        t_k = key_value.shape[1]
        q = self.q_proj(self.norm_q(query)).reshape(b, t_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.norm_kv(key_value)).reshape(b, t_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.norm_kv(key_value)).reshape(b, t_k, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(b, t_q, self.n_heads * self.head_dim)
        attn = self.out_proj(self.dropout(attn))
        return attn


class GatedTemporalCrossAttention(StandardCrossAttention):
    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        n_heads: int,
        dropout: float = 0.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__(query_dim, kv_dim, n_heads, dropout)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
