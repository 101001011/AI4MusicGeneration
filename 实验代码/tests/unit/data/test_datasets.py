from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import torch

from hs_mad.data.datasets.maestro import MaestroDataset
from hs_mad.data.datasets.slakh import SlakhDataset
from hs_mad.utils.io import write_json


def create_audio(path: Path, sr: int = 44100) -> None:
    t = np.linspace(0, 1, sr, endpoint=False)
    wave = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    import soundfile as sf

    sf.write(path, wave, sr)


def create_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(start=0.0, end=0.5, pitch=60, velocity=90))
    midi.instruments.append(inst)
    midi.write(str(path))


def build_manifest(tmp_path: Path, name: str) -> Path:
    audio = tmp_path / f"{name}.wav"
    midi = tmp_path / f"{name}.mid"
    create_audio(audio)
    create_midi(midi)
    manifest = [
        {
            "uid": name,
            "audio": str(audio),
            "midi": str(midi),
            "style_text": name,
        }
    ]
    manifest_path = tmp_path / f"{name}.json"
    write_json(manifest_path, manifest)
    return manifest_path


def test_maestro_dataset(tmp_path: Path):
    manifest_path = build_manifest(tmp_path, "maestro_sample")
    dataset = MaestroDataset(manifest_path, sample_rate=44100, segment_seconds=1.0)
    sample = dataset[0]
    assert sample["wave"].shape[1] == 44100
    assert isinstance(sample["midi"].notes[0].pitch, int)
    assert sample["style"]["text"] == "maestro_sample"
    assert sample["style"]["ref"] is None


def test_slakh_dataset(tmp_path: Path):
    manifest_path = build_manifest(tmp_path, "slakh_sample")
    dataset = SlakhDataset(manifest_path, sample_rate=44100, segment_seconds=1.0)
    sample = dataset[0]
    assert sample["wave"].shape[1] == 44100
    assert torch.is_tensor(sample["wave"])
