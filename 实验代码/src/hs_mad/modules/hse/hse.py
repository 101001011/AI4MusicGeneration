"""Hierarchical Symbolic Encoder implementation."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hs_mad.data.srm.renderer import SyncRenderingModule
from hs_mad.utils.midi import MidiPerformance

from .aux_losses import AuxiliaryTargets, HierarchicalAuxiliaryLosses
from .conformer import ConformerConfig, ConformerStack

__all__ = ["HierarchicalSymbolicEncoder"]


class HierarchicalSymbolicEncoder(nn.Module):
    def __init__(
        self,
        r1: int = 4,
        r2: int = 4,
        d_in: int = 384,
        d_event: int = 512,
        d_local: int = 1024,
        d_global: int = 1536,
        n_blocks: Sequence[int] = (6, 6, 6),
        conformer_cfg: Optional[Dict[str, float]] = None,
        sr_latent: float = 137.8,
    ) -> None:
        super().__init__()
        self.r1 = r1
        self.r2 = r2
        self.srm = SyncRenderingModule(d_in=d_in, sr_latent=sr_latent)

        cfg_defaults = {
            "ffn_mult": 4.0,
            "num_heads": 8,
            "conv_kernel": 31,
            "dropout": 0.1,
            "drop_path": 0.1,
        }
        if conformer_cfg:
            cfg_defaults.update({k: float(v) for k, v in conformer_cfg.items() if k in cfg_defaults})

        def make_cfg(dim: int) -> ConformerConfig:
            return ConformerConfig(
                dim=dim,
                ffn_mult=cfg_defaults["ffn_mult"],
                num_heads=int(cfg_defaults["num_heads"]),
                conv_kernel=int(cfg_defaults["conv_kernel"]),
                dropout=cfg_defaults["dropout"],
                drop_path=cfg_defaults["drop_path"],
            )

        self.event_input = nn.Conv1d(d_in, d_event, kernel_size=1)
        self.stage1 = ConformerStack(make_cfg(d_event), n_blocks[0])
        self.downsample1 = nn.Conv1d(
            d_event,
            d_local,
            kernel_size=2 * r1,
            stride=r1,
            padding=r1 // 2,
        )
        self.stage2 = ConformerStack(make_cfg(d_local), n_blocks[1])
        self.downsample2 = nn.Conv1d(
            d_local,
            d_global,
            kernel_size=2 * r2,
            stride=r2,
            padding=r2 // 2,
        )
        self.stage3 = ConformerStack(make_cfg(d_global), n_blocks[2])

        self.aux_losses = HierarchicalAuxiliaryLosses(d_local=d_local, d_global=d_global)

    def _ensure_sequence(self, midi: Sequence[MidiPerformance]) -> Sequence[MidiPerformance]:
        if isinstance(midi, MidiPerformance):
            return [midi]
        return midi

    def forward(
        self,
        midi: Sequence[MidiPerformance],
        dur_sec: float | Sequence[float],
        aux_targets: Optional[AuxiliaryTargets] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        midi_batch = self._ensure_sequence(midi)
        if isinstance(dur_sec, torch.Tensor):
            dur_sec = dur_sec.tolist() if dur_sec.ndim > 0 else float(dur_sec.item())
        s_sync = self.srm.render(midi_batch, dur_sec)  # [B, D_in, T_H]

        event = self.event_input(s_sync).transpose(1, 2)  # [B, T_H, d_event]
        event = self.stage1(event)

        local_in = self.downsample1(event.transpose(1, 2))
        local = self.stage2(local_in.transpose(1, 2))

        global_in = self.downsample2(local.transpose(1, 2))
        global_feat = self.stage3(global_in.transpose(1, 2))

        features = {
            "event": event,
            "local": local,
            "global": global_feat,
        }

        aux_losses = self.aux_losses(local, global_feat, aux_targets)
        return features, aux_losses
