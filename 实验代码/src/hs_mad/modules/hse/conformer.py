"""Conformer blocks used within the hierarchical symbolic encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ConformerBlock", "ConformerStack", "ConformerConfig"]


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim * num_heads != dim:
            raise ValueError("dim must be divisible by num_heads")
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(b, t, d)
        return self.out(self.dropout(attn))


class ConvolutionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


@dataclass
class ConformerConfig:
    dim: int
    ffn_mult: float = 4.0
    num_heads: int = 8
    conv_kernel: int = 31
    dropout: float = 0.1
    drop_path: float = 0.1


class ConformerBlock(nn.Module):
    def __init__(self, cfg: ConformerConfig) -> None:
        super().__init__()
        self.ffn1 = FeedForwardModule(cfg.dim, cfg.ffn_mult, cfg.dropout)
        self.attn = MultiHeadSelfAttention(cfg.dim, cfg.num_heads, cfg.dropout)
        self.conv = ConvolutionModule(cfg.dim, cfg.conv_kernel, cfg.dropout)
        self.ffn2 = FeedForwardModule(cfg.dim, cfg.ffn_mult, cfg.dropout)
        self.norm_final = nn.LayerNorm(cfg.dim)
        self.drop_path = DropPath(cfg.drop_path)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + 0.5 * self.drop_path(self.ffn1(x))
        x = x + self.drop_path(self.attn(x, mask))
        x = x + self.drop_path(self.conv(x))
        x = x + 0.5 * self.drop_path(self.ffn2(x))
        return self.norm_final(x)


class ConformerStack(nn.Module):
    def __init__(self, cfg: ConformerConfig, num_layers: int) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            drop_path_rate = cfg.drop_path * (i + 1) / max(1, num_layers)
            layer_cfg = ConformerConfig(
                dim=cfg.dim,
                ffn_mult=cfg.ffn_mult,
                num_heads=cfg.num_heads,
                conv_kernel=cfg.conv_kernel,
                dropout=cfg.dropout,
                drop_path=drop_path_rate,
            )
            layers.append(ConformerBlock(layer_cfg))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
