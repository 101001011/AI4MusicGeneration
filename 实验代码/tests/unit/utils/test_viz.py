from __future__ import annotations

import numpy as np

from hs_mad.utils.viz import plot_alignment, plot_piano_roll


def test_plot_piano_roll_creates_figure():
    times = np.linspace(0, 1, 10)
    pitches = np.arange(60, 62)
    velocities = np.random.rand(pitches.shape[0], times.shape[0])
    fig, ax = plot_piano_roll(times, pitches, velocities)
    assert fig is not None
    assert ax is not None
    fig.clf()


def test_plot_alignment_creates_figure():
    fig, ax = plot_alignment([0.1, 0.5], [0.12, 0.52])
    assert fig is not None
    assert ax is not None
    fig.clf()
