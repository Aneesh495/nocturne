"""Tests for checkpoint recovery, corruption detection, and deterministic resumption."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from nocturne.training.checkpoint import CheckpointManager
from nocturne.training.recovery import TrainingRecovery


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


def test_recovery_validates_clean_checkpoint(tmp_path):
    manager = CheckpointManager(tmp_path)
    model = SmallNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    path = manager.save(
        epoch=5,
        step=100,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        metrics={"val_loss": 0.42},
    )

    data = TrainingRecovery.verify_checkpoint_integrity(path)
    assert data["epoch"] == 5
    assert data["step"] == 100

    epoch, step, metrics = TrainingRecovery.resume_from_checkpoint(
        path, model, optimizer
    )
    assert epoch == 5
    assert step == 100
    assert metrics["val_loss"] == 0.42


def test_recovery_detects_corrupt_non_finite_checkpoint(tmp_path):
    manager = CheckpointManager(tmp_path)
    model = SmallNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Inject NaN into model weight before saving
    with torch.no_grad():
        model.fc.weight[0, 0] = float("nan")

    path = manager.save(
        epoch=1,
        step=10,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        metrics={"val_loss": 1.0},
    )

    with pytest.raises(ValueError, match="non-finite values detected"):
        TrainingRecovery.verify_checkpoint_integrity(path)
