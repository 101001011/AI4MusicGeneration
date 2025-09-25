"""Auxiliary losses for hierarchical supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HierarchicalAuxiliaryLosses", "AuxiliaryTargets"]


@dataclass
class AuxiliaryTargets:
    harmony: Optional[torch.Tensor] = None  # [B, T_M]
    beat: Optional[torch.Tensor] = None  # [B, T_M]
    tempo: Optional[torch.Tensor] = None  # [B]
    key: Optional[torch.Tensor] = None  # [B]


class HierarchicalAuxiliaryLosses(nn.Module):
    def __init__(self, d_local: int, d_global: int) -> None:
        super().__init__()
        self.harmony_head = nn.Linear(d_local, 24)
        self.beat_head = nn.Linear(d_local, 2)
        self.tempo_head = nn.Linear(d_global, 1)
        self.key_head = nn.Linear(d_global, 24)

    def forward(
        self,
        local_feats: torch.Tensor,
        global_feats: torch.Tensor,
        targets: Optional[AuxiliaryTargets] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        harmony_logits = self.harmony_head(local_feats)
        beat_logits = self.beat_head(local_feats)
        tempo_pred = self.tempo_head(global_feats.mean(dim=1))
        key_logits = self.key_head(global_feats.mean(dim=1))

        device = local_feats.device
        zero = torch.tensor(0.0, device=device)

        if targets and targets.harmony is not None:
            losses["harm"] = F.cross_entropy(harmony_logits.view(-1, harmony_logits.size(-1)), targets.harmony.view(-1))
        else:
            losses["harm"] = zero

        if targets and targets.beat is not None:
            losses["beat"] = F.cross_entropy(beat_logits.view(-1, beat_logits.size(-1)), targets.beat.view(-1))
        else:
            losses["beat"] = zero

        if targets and targets.tempo is not None:
            losses["tempo"] = F.l1_loss(tempo_pred.squeeze(-1), targets.tempo)
        else:
            losses["tempo"] = zero

        if targets and targets.key is not None:
            losses["key"] = F.cross_entropy(key_logits, targets.key)
        else:
            losses["key"] = zero

        return losses
