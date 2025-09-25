"""Audio utility functions for HS-MAD."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf

__all__ = [
    "load_audio",
    "save_audio",
    "normalize_audio",
    "segment_audio",
    "time_to_latent_frames",
    "latent_frames_to_time",
]


def load_audio(path: Path | str, sample_rate: int) -> np.ndarray:
    """Load mono audio and resample to target sample rate."""

    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1)  # mono mixdown
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="kaiser_best")
    return audio.astype(np.float32)


def save_audio(path: Path | str, audio: np.ndarray, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def normalize_audio(audio: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    peak = np.max(np.abs(audio)) + eps
    return audio / peak


def segment_audio(audio: np.ndarray, sample_rate: int, segment_seconds: float) -> np.ndarray:
    target_length = int(segment_seconds * sample_rate)
    if audio.shape[0] >= target_length:
        return audio[:target_length]
    pad = target_length - audio.shape[0]
    return np.pad(audio, (0, pad))


def time_to_latent_frames(duration_seconds: float, latent_rate: float) -> int:
    return int(np.round(duration_seconds * latent_rate))


def latent_frames_to_time(frames: int, latent_rate: float) -> float:
    return frames / latent_rate
