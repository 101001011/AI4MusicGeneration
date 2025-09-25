"""MIDI parsing helpers shared by SRM and HSE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pretty_midi

__all__ = [
    "NoteEvent",
    "TempoEvent",
    "TimeSignatureEvent",
    "MidiPerformance",
    "load_midi",
    "performance_to_note_array",
]


@dataclass(slots=True)
class NoteEvent:
    start: float
    end: float
    pitch: int
    velocity: float
    instrument: str
    program: int
    is_drum: bool


@dataclass(slots=True)
class TempoEvent:
    time: float
    bpm: float


@dataclass(slots=True)
class TimeSignatureEvent:
    time: float
    numerator: int
    denominator: int


@dataclass
class MidiPerformance:
    notes: List[NoteEvent]
    tempos: List[TempoEvent]
    time_signatures: List[TimeSignatureEvent]
    duration: float

    @classmethod
    def from_pretty_midi(cls, midi: pretty_midi.PrettyMIDI) -> "MidiPerformance":
        notes: List[NoteEvent] = []
        for inst in midi.instruments:
            for note in inst.notes:
                notes.append(
                    NoteEvent(
                        start=float(note.start),
                        end=float(note.end),
                        pitch=int(note.pitch),
                        velocity=float(note.velocity) / 127.0,
                        instrument=inst.name or pretty_midi.program_to_instrument_name(inst.program),
                        program=int(inst.program),
                        is_drum=bool(inst.is_drum),
                    )
                )
        notes.sort(key=lambda n: (n.start, n.pitch))

        tempos = [TempoEvent(float(t), float(bpm)) for t, bpm in zip(*midi.get_tempo_changes())]
        if not tempos:
            tempos = [TempoEvent(0.0, 120.0)]

        time_sigs = [
            TimeSignatureEvent(float(sig.time), int(sig.numerator), int(sig.denominator))
            for sig in midi.time_signature_changes
        ]
        if not time_sigs:
            time_sigs = [TimeSignatureEvent(0.0, 4, 4)]

        duration = float(midi.get_end_time())
        return cls(notes=notes, tempos=tempos, time_signatures=time_sigs, duration=duration)


def load_midi(path: Path | str) -> MidiPerformance:
    midi = pretty_midi.PrettyMIDI(str(path))
    return MidiPerformance.from_pretty_midi(midi)


def performance_to_note_array(perf: MidiPerformance) -> np.ndarray:
    """Convert a performance to a structured NumPy array for SRM rendering.

    Returns:
        np.ndarray: shape [N, 5] with columns (start, end, pitch, velocity, is_drum).
    """

    if not perf.notes:
        return np.zeros((0, 5), dtype=np.float32)

    data = np.zeros((len(perf.notes), 5), dtype=np.float32)
    for idx, note in enumerate(perf.notes):
        data[idx] = [note.start, note.end, note.pitch, note.velocity, float(note.is_drum)]
    return data
