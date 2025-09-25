from __future__ import annotations

import numpy as np
import torch

from hs_mad.eval.metrics import compute_clap_similarity, compute_fad, compute_onset_metrics, compute_structure_metrics


def test_compute_fad_returns_non_negative():
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    ref = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    gen = np.sin(2 * np.pi * 442 * t).astype(np.float32)
    fad = compute_fad(ref, gen, sr)
    assert fad >= 0


def test_compute_onset_metrics_basic():
    ref = np.array([0.1, 0.5, 0.9])
    gen = np.array([0.11, 0.48, 0.95])
    metrics = compute_onset_metrics(ref, gen)
    assert 0 <= metrics["f1"] <= 1


def test_compute_structure_metrics():
    ref = np.array([0.0, 1.0, 2.0])
    gen = np.array([0.1, 1.1, 1.9])
    metrics = compute_structure_metrics(ref, gen)
    assert metrics["bac"] >= 0


def test_compute_clap_similarity_runs():
    audio = torch.randn(1, 44100)
    score = compute_clap_similarity(audio, ["piano"], torch.device("cpu"))
    assert isinstance(score, float)
