"""Decoupled classifier-free guidance for structure and style."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

__all__ = ["DecoupledCFG"]


def _zero_structure(structure_cond: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: torch.zeros_like(value) for key, value in structure_cond.items()}


class DecoupledCFG(nn.Module):
    def forward(
        self,
        unet: nn.Module,
        z_t: torch.Tensor,
        t_embed: torch.Tensor,
        structure_cond: Dict[str, torch.Tensor],
        style_cond: Optional[torch.Tensor],
        w_structure: float,
        w_style: float,
    ) -> torch.Tensor:
        null_structure = _zero_structure(structure_cond)
        null_style = torch.zeros_like(style_cond) if style_cond is not None else None

        eps_null = unet(z_t, t_embed, null_structure, null_style)
        eps_structure = unet(z_t, t_embed, structure_cond, null_style)
        eps_style = unet(z_t, t_embed, null_structure, style_cond)

        return eps_null + w_structure * (eps_structure - eps_null) + w_style * (eps_style - eps_null)
