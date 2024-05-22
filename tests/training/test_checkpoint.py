"""Tests for atomic checkpoint saving, loading, RNG capture, and pruning."""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import pytest

from nocturne.training.checkpoint import (
    CheckpointManager,
    capture_rng_states,
    restore_rng_states,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)


def test_checkpoint_atomic_save_and_load(tmp_path):
    manager = CheckpointManager(tmp_path, keep_top_k=2)
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    path = manager.save(
        epoch=1,
        step=50,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        metrics={"val_loss": 1.5},
        is_best=True,
    )

    assert path.exists()
    assert (tmp_path / "latest.pt").exists()
    assert (tmp_path / "best.pt").exists()

    # Load back into fresh model
    model_loaded = DummyModel()
    optimizer_loaded = torch.optim.Adam(model_loaded.parameters(), lr=0.01)

    checkpoint = manager.load(path, model_loaded, optimizer_loaded)

    assert checkpoint["epoch"] == 1
    assert checkpoint["step"] == 50
    assert checkpoint["metrics"]["val_loss"] == 1.5

    # Check model weights match exactly
    for p1, p2 in zip(model.parameters(), model_loaded.parameters()):
        assert torch.equal(p1, p2)


def test_checkpoint_pruning(tmp_path):
    manager = CheckpointManager(tmp_path, keep_top_k=2)
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Save 4 checkpoints with decreasing validation losses
    p1 = manager.save(1, 10, model, optimizer, None, {"val_loss": 5.0})
    p2 = manager.save(2, 20, model, optimizer, None, {"val_loss": 3.0})
    p3 = manager.save(3, 30, model, optimizer, None, {"val_loss": 2.0})
    p4 = manager.save(4, 40, model, optimizer, None, {"val_loss": 1.0})

    # Only top 2 (p3 and p4) plus latest should remain
    assert p4.exists()
    assert p3.exists()
    assert not p1.exists()


def test_rng_capture_and_restoration():
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    states = capture_rng_states()

    # Advance generators
    val1_py = random.random()
    val1_np = np.random.rand()
    val1_torch = torch.rand(1).item()

    # Restore
    restore_rng_states(states)

    val2_py = random.random()
    val2_np = np.random.rand()
    val2_torch = torch.rand(1).item()

    assert val1_py == val2_py
    assert val1_np == val2_np
    assert val1_torch == val2_torch
