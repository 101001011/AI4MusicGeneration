"""Base dataset utilities for HS-MAD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from hs_mad.utils.audio import load_audio, segment_audio
from hs_mad.utils.midi import MidiPerformance, load_midi

__all__ = ["BaseMusicDataset", "ManifestEntry"]


@dataclass
class ManifestEntry:
    uid: str
    audio_path: Path
    midi_path: Path
    style_text: Optional[str]
    style_audio_path: Optional[Path]
    duration: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_duration: float) -> "ManifestEntry":
        return cls(
            uid=data["uid"],
            audio_path=Path(data["audio"]),
            midi_path=Path(data["midi"]),
            style_text=data.get("style_text"),
            style_audio_path=Path(data["style_audio"]) if data.get("style_audio") else None,
            duration=float(data.get("duration", default_duration)),
        )


class BaseMusicDataset(Dataset[Dict[str, Any]]):
    def __init__(
        self,
        manifest: List[Dict[str, Any]],
        sample_rate: int,
        segment_seconds: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.entries = [ManifestEntry.from_dict(item, default_duration=segment_seconds) for item in manifest]

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.entries)

    def _load_audio(self, entry: ManifestEntry) -> torch.Tensor:
        wave = load_audio(entry.audio_path, self.sample_rate)
        wave = segment_audio(wave, self.sample_rate, self.segment_seconds)
        return torch.from_numpy(wave).unsqueeze(0)  # [1, T]

    def _load_midi(self, entry: ManifestEntry) -> MidiPerformance:
        return load_midi(entry.midi_path)

    def _load_style_ref(self, entry: ManifestEntry) -> Optional[torch.Tensor]:
        if not entry.style_audio_path:
            return None
        ref_wave = load_audio(entry.style_audio_path, self.sample_rate)
        ref_wave = segment_audio(ref_wave, self.sample_rate, entry.duration)
        return torch.from_numpy(ref_wave).unsqueeze(0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        wave = self._load_audio(entry)
        midi = self._load_midi(entry)
        style_ref = self._load_style_ref(entry)
        style = {"text": entry.style_text, "ref": style_ref}
        return {
            "uid": entry.uid,
            "wave": wave,
            "midi": midi,
            "dur_sec": float(entry.duration),
            "style": style,
        }
