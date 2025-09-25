from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

from hs_mad.utils.midi import load_midi, performance_to_note_array


def create_test_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="piano")
    note = pretty_midi.Note(start=0.0, end=1.0, pitch=60, velocity=100)
    inst.notes.append(note)
    midi.instruments.append(inst)
    midi.write(str(path))


def test_load_midi_and_note_array(tmp_path: Path):
    midi_path = tmp_path / "test.mid"
    create_test_midi(midi_path)
    perf = load_midi(midi_path)
    assert perf.duration == 1.0
    assert len(perf.notes) == 1
    note = perf.notes[0]
    assert note.pitch == 60
    assert np.isclose(note.velocity, 100 / 127.0)

    array = performance_to_note_array(perf)
    assert array.shape == (1, 5)
    assert np.isclose(array[0, 0], 0.0)
    assert np.isclose(array[0, 1], 1.0)
    assert array[0, 2] == 60
    assert np.isclose(array[0, 3], 100 / 127.0)
    assert array[0, 4] == 0.0
