from __future__ import annotations

import torch

from hs_mad.modules.diffusion.scheduler import DiffusionScheduler, SchedulerConfig, get_timestep_embedding


def test_scheduler_add_noise():
    scheduler = DiffusionScheduler(SchedulerConfig(num_train_timesteps=10, schedule="linear"))
    x0 = torch.ones(2, 4, 8)
    noise = torch.zeros_like(x0)
    timesteps = torch.tensor([0, 5])
    noisy = scheduler.add_noise(x0, noise, timesteps)
    assert torch.allclose(noisy[0], x0[0], atol=1e-4)


def test_timestep_embedding_shape():
    t = torch.tensor([0, 1, 2])
    emb = get_timestep_embedding(t, 8)
    assert emb.shape == (3, 8)
