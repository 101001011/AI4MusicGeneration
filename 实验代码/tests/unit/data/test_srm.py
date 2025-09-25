from __future__ import annotations

import torch

from hs_mad.data.srm.renderer import SyncRenderingModule
from hs_mad.utils.midi import MidiPerformance, NoteEvent, TempoEvent, TimeSignatureEvent


def make_performance() -> MidiPerformance:
    return MidiPerformance(
        notes=[
            NoteEvent(start=0.1, end=0.5, pitch=60, velocity=1.0, instrument="piano", program=0, is_drum=False),
            NoteEvent(start=0.3, end=0.7, pitch=64, velocity=0.8, instrument="piano", program=0, is_drum=False),
        ],
        tempos=[TempoEvent(0.0, 120.0)],
        time_signatures=[TimeSignatureEvent(0.0, 4, 4)],
        duration=1.0,
    )


def test_sync_rendering_alignment():
    srm = SyncRenderingModule(d_in=384, sr_latent=100.0)
    perf = make_performance()
    tensor = srm.render([perf], dur_sec=1.0)
    assert tensor.shape == (1, 384, 100)
    onset_60 = tensor[0, 60]
    peak_index = torch.argmax(onset_60).item()
    assert abs(peak_index - int(0.1 * 100)) <= 1  # gaussian centered near start


def test_sync_rendering_handles_empty_notes():
    srm = SyncRenderingModule(d_in=384, sr_latent=100.0)
    empty_perf = MidiPerformance(notes=[], tempos=[], time_signatures=[], duration=1.0)
    tensor = srm.render([empty_perf], dur_sec=1.0)
    assert torch.allclose(tensor, torch.zeros_like(tensor))
