"""Visualization helpers used for diagnostics."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_piano_roll", "plot_alignment"]


def plot_piano_roll(times: np.ndarray, pitches: np.ndarray, velocities: np.ndarray):
    """Plot a simple piano roll heatmap."""

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(times, pitches, velocities, shading="auto")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MIDI Pitch")
    fig.colorbar(mesh, ax=ax, label="Velocity")
    fig.tight_layout()
    return fig, ax


def plot_alignment(audio_onsets: Iterable[float], midi_onsets: Iterable[float]):
    """Compare audio vs MIDI onset times on a timeline."""

    fig, ax = plt.subplots(figsize=(10, 2))
    audio_onsets = np.array(list(audio_onsets))
    midi_onsets = np.array(list(midi_onsets))
    ax.vlines(audio_onsets, 0, 1, colors="tab:blue", label="Audio", linewidth=1.5)
    ax.vlines(midi_onsets, 0, 1, colors="tab:orange", label="MIDI", linewidth=1.5, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax
