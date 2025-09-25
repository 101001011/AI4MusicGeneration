"""Evaluation utilities for HS-MAD."""

from __future__ import annotations

import numpy as np
import torch
from scipy import linalg

from hs_mad.modules.codecs.clap_encoder import CLAPEncoder, CLAPConfig

__all__ = [
    "compute_fad",
    "compute_onset_metrics",
    "compute_structure_metrics",
    "compute_clap_similarity",
]


def _mfcc_features(audio: np.ndarray, sample_rate: int, n_mfcc: int = 30) -> np.ndarray:
    import librosa

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc.T


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * 1e-6).dot(sigma2 + np.eye(sigma2.shape[0]) * 1e-6))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)


def compute_fad(reference: np.ndarray, generated: np.ndarray, sample_rate: int) -> float:
    ref_feats = _mfcc_features(reference, sample_rate)
    gen_feats = _mfcc_features(generated, sample_rate)
    mu_ref, sigma_ref = ref_feats.mean(axis=0), np.cov(ref_feats, rowvar=False)
    mu_gen, sigma_gen = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)
    return float(_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen))


def compute_onset_metrics(reference_onsets: np.ndarray, generated_onsets: np.ndarray, tolerance: float = 0.05) -> dict[str, float]:
    ref = list(reference_onsets)
    gen = list(generated_onsets)
    tp = 0
    used = set()
    for r in ref:
        for idx, g in enumerate(gen):
            if idx in used:
                continue
            if abs(r - g) <= tolerance:
                tp += 1
                used.add(idx)
                break
    precision = tp / max(len(gen), 1)
    recall = tp / max(len(ref), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    mae = float(np.mean([abs(r - g) for r, g in zip(sorted(ref), sorted(gen))])) if ref and gen else 0.0
    return {"f1": f1, "mae": mae}


def compute_structure_metrics(reference_beats: np.ndarray, generated_beats: np.ndarray) -> dict[str, float]:
    if reference_beats.size == 0 or generated_beats.size == 0:
        return {"bac": 0.0}
    diffs = np.abs(reference_beats - generated_beats[: reference_beats.size])
    return {"bac": float(np.mean(diffs))}


def compute_clap_similarity(audio: torch.Tensor, style_text: list[str], device: torch.device) -> float:
    encoder = CLAPEncoder(CLAPConfig(), device)
    style_vec = encoder({"text": style_text})
    audio_vec = encoder({"ref": audio})
    style_norm = style_vec / (style_vec.norm(dim=-1, keepdim=True) + 1e-8)
    audio_norm = audio_vec / (audio_vec.norm(dim=-1, keepdim=True) + 1e-8)
    return float((style_norm * audio_norm).sum(dim=-1).mean().item())
