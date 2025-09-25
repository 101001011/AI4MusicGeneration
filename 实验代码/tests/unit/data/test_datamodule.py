from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pretty_midi
import torch

from hs_mad.data.datamodules import create_datamodule
from hs_mad.utils.io import write_json


def build_manifest(tmp_path: Path, name: str) -> Path:
    audio = tmp_path / f"{name}.wav"
    midi = tmp_path / f"{name}.mid"

    t = np.linspace(0, 1, 44100, endpoint=False)
    wave = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    import soundfile as sf

    sf.write(audio, wave, 44100)

    midi_obj = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(start=0.0, end=0.5, pitch=60, velocity=90))
    midi_obj.instruments.append(inst)
    midi_obj.write(str(midi))

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


def test_datamodule_collate(tmp_path):
    train_manifest = build_manifest(tmp_path, "train_sample")
    val_manifest = build_manifest(tmp_path, "val_sample")

    cfg = SimpleNamespace(
        sample_rate=44100,
        segment_seconds=1.0,
        latent_rate=137.8,
        collate={"batch_size": 2, "num_workers": 0, "pin_memory": False},
        train_datasets={"maestro": {"manifest": str(train_manifest)}},
        val_datasets={"maestro": {"manifest": str(val_manifest)}},
    )

    module = create_datamodule(cfg)
    module.setup("fit")
    loader = module.train_dataloader()
    batch = next(iter(loader))
    assert batch["wave"].shape[0] == 1
    assert batch["latent_frames"].dtype == torch.long
    assert batch["style"]["text"][0] == "train_sample"
