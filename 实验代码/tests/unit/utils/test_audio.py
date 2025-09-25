from __future__ import annotations

from pathlib import Path

import numpy as np

from hs_mad.utils.audio import (
    latent_frames_to_time,
    load_audio,
    normalize_audio,
    save_audio,
    segment_audio,
    time_to_latent_frames,
)


def test_load_and_save_audio(tmp_path: Path):
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "test.wav"
    save_audio(path, wave, sr)
    loaded = load_audio(path, sample_rate=sr)
    assert loaded.shape == wave.shape
    assert np.allclose(loaded[:100], wave[:100], atol=1e-4)


def test_normalize_and_segment():
    audio = np.array([0.2, -0.5, 0.1], dtype=np.float32)
    norm = normalize_audio(audio)
    assert np.isclose(np.max(np.abs(norm)), 1.0)

    seg = segment_audio(np.ones(10), sample_rate=10, segment_seconds=1.5)
    assert seg.shape[0] == 15
    assert np.allclose(seg[:10], 1.0)
    assert np.allclose(seg[10:], 0.0)


def test_latent_time_conversion():
    assert time_to_latent_frames(1.0, 137.8) == 138
    assert np.isclose(latent_frames_to_time(4134, 137.8), 30.0, atol=0.1)
