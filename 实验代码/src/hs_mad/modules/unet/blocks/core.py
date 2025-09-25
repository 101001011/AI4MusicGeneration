"""Core 1D U-Net building blocks used by HS-MAD."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResBlock1D", "Downsample1D", "Upsample1D", "AdaLNModulator"]


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        time_term = self.time_proj(time_emb).unsqueeze(-1)
        h = h + time_term
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(self.conv2(h))
        return residual + h


class Downsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=2 * factor,
            stride=factor,
            padding=factor // 2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int) -> None:
        super().__init__()
        self.transposed = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=factor,
            stride=factor,
        )

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        x = self.transposed(x)
        if x.size(-1) > target_length:
            x = x[..., :target_length]
        elif x.size(-1) < target_length:
            pad_amount = target_length - x.size(-1)
            x = F.pad(x, (0, pad_amount))
        return x


class AdaLNModulator(nn.Module):
    def __init__(self, channels: int, style_dim: int, time_dim: int) -> None:
        super().__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.time_dim = time_dim
        self.norm = nn.LayerNorm(channels)
        self.linear = nn.Linear(style_dim + time_dim, channels * 2)

    def forward(
        self,
        x: torch.Tensor,
        style: Optional[torch.Tensor],
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        b, c, t = x.shape
        if style is None:
            style = torch.zeros((b, self.style_dim), device=x.device, dtype=x.dtype)
        combined = torch.cat([style, time_emb], dim=-1)
        gamma, beta = self.linear(combined).chunk(2, dim=-1)
        x_t = x.transpose(1, 2)
        x_norm = self.norm(x_t)
        x_mod = x_norm * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return x_mod.transpose(1, 2)
