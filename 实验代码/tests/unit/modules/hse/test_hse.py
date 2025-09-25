from __future__ import annotations

import torch

from hs_mad.modules.hse.aux_losses import AuxiliaryTargets
from hs_mad.modules.hse.hse import HierarchicalSymbolicEncoder
from hs_mad.utils.midi import MidiPerformance, NoteEvent, TempoEvent, TimeSignatureEvent


def build_performance() -> MidiPerformance:
    return MidiPerformance(
        notes=[
            NoteEvent(0.1, 0.5, 60, 1.0, "piano", 0, False),
            NoteEvent(0.4, 0.9, 64, 0.8, "piano", 0, False),
        ],
        tempos=[TempoEvent(0.0, 120.0)],
        time_signatures=[TimeSignatureEvent(0.0, 4, 4)],
        duration=1.0,
    )


def test_hse_outputs_shapes():
    encoder = HierarchicalSymbolicEncoder(r1=4, r2=4, d_in=384, d_event=16, d_local=32, d_global=48, n_blocks=(1, 1, 1), sr_latent=100.0)
    feats, aux = encoder([build_performance()], dur_sec=1.0)
    assert feats["event"].shape == (1, 100, 16)
    assert feats["local"].shape[1] == 25
    assert feats["global"].shape[1] == 6
    assert set(aux.keys()) == {"harm", "beat", "tempo", "key"}


def test_hse_auxiliary_losses_with_targets():
    encoder = HierarchicalSymbolicEncoder(r1=4, r2=4, d_in=384, d_event=16, d_local=32, d_global=48, n_blocks=(1, 1, 1), sr_latent=100.0)
    feats, aux = encoder([build_performance()], dur_sec=torch.tensor([1.0]), aux_targets=AuxiliaryTargets(
        harmony=torch.zeros((1, 25), dtype=torch.long),
        beat=torch.zeros((1, 25), dtype=torch.long),
        tempo=torch.tensor([1.0]),
        key=torch.zeros((1,), dtype=torch.long),
    ))
    assert aux["harm"].requires_grad
    assert aux["tempo"].item() >= 0
