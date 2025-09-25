"""Soft piano roll rendering aligned to latent frames."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from hs_mad.utils.midi import MidiPerformance, performance_to_note_array

__all__ = ["SyncRenderingModule"]


class SyncRenderingModule(nn.Module):
    """Render MIDI performances into soft piano-roll tensors."""

    def __init__(
        self,
        d_in: int = 384,
        sigma_on_ms: float = 3.6,
        sigma_off_ms: float = 3.6,
        sr_latent: float = 137.8,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_in % 3 != 0:
            raise ValueError("d_in must be divisible by 3 (onset/sustain/velocity channels)")
        self.d_in = d_in
        self.num_pitches = d_in // 3
        self.sigma_on = sigma_on_ms / 1000.0 * sr_latent
        self.sigma_off = sigma_off_ms / 1000.0 * sr_latent
        self.sr_latent = sr_latent
        self.register_buffer("_time_axis", torch.zeros(1), persistent=False)
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._time_axis.device

    def _ensure_time_axis(self, t_h: int) -> torch.Tensor:
        if self._time_axis.numel() != t_h:
            self._time_axis = torch.arange(t_h, device=self._time_axis.device, dtype=torch.float32)
        return self._time_axis

    def render(self, midi_batch: Sequence[MidiPerformance], dur_sec: float | Sequence[float]) -> torch.Tensor:
        if isinstance(dur_sec, (list, tuple)):
            unique = {float(d) for d in dur_sec}
            if len(unique) != 1:
                raise ValueError("All durations must match for synchronized rendering")
            dur_value = unique.pop()
        else:
            dur_value = float(dur_sec)
        device = self.device if self._time_axis.numel() > 0 else torch.device(self._device or "cpu")
        t_h = int(math.floor(dur_value * self.sr_latent))
        time_axis = self._ensure_time_axis(t_h).to(device)
        batch = []
        for perf in midi_batch:
            batch.append(self._render_single(perf, t_h, time_axis))
        return torch.stack(batch, dim=0)

    def forward(self, midi_batch: Sequence[MidiPerformance], dur_sec: float | Sequence[float]) -> torch.Tensor:
        return self.render(midi_batch, dur_sec)

    def _render_single(self, perf: MidiPerformance, t_h: int, time_axis: torch.Tensor) -> torch.Tensor:
        onset = torch.zeros((self.num_pitches, t_h), device=time_axis.device, dtype=torch.float32)
        sustain = torch.zeros_like(onset)
        velocity = torch.zeros_like(onset)

        notes = performance_to_note_array(perf)
        if notes.shape[0] == 0:
            return torch.cat([onset, sustain, velocity], dim=0)

        sigma_on = max(self.sigma_on, 1e-4)
        sigma_off = max(self.sigma_off, 1e-4)
        for start, end, pitch, vel, is_drum in notes:
            pitch_idx = int(min(max(pitch, 0), self.num_pitches - 1))
            start_frame = start * self.sr_latent
            end_frame = end * self.sr_latent
            # onset gaussian
            if vel > 0.0:
                onset[pitch_idx] += torch.exp(-0.5 * ((time_axis - start_frame) / sigma_on) ** 2) * vel
            # release gaussian contributes to sustain tail
            sustain_gauss = torch.exp(-0.5 * ((time_axis - end_frame) / sigma_off) ** 2)
            sustain_range_start = max(int(math.floor(start_frame)), 0)
            sustain_range_end = min(int(math.ceil(end_frame)), t_h)
            if sustain_range_end > sustain_range_start:
                sustain[pitch_idx, sustain_range_start:sustain_range_end] += 1.0
                velocity[pitch_idx, sustain_range_start:sustain_range_end] += vel
            # Add smooth release tail (decays after end)
            release_mask = time_axis >= end_frame
            sustain[pitch_idx, release_mask] += sustain_gauss[release_mask] * 0.1
            velocity[pitch_idx, release_mask] += sustain_gauss[release_mask] * vel * 0.1

        tensor = torch.cat([onset, sustain, velocity], dim=0)
        tensor = torch.clamp(tensor, min=0.0, max=1.0)
        return tensor
