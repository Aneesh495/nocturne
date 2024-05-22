"""Tests for optimizer construction, parameter grouping, scheduler, and clipping."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from nocturne.training.config import TrainingConfig
from nocturne.training.optimizer import create_optimizer, create_lr_scheduler, clip_gradients


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.norm = nn.LayerNorm(10)

    def forward(self, x):
        return self.norm(self.fc(x))


def test_optimizer_parameter_groups():
    model = SimpleModule()
    config = TrainingConfig(learning_rate=1e-3, weight_decay=1e-2)
    optimizer = create_optimizer(model, config)

    # 2 parameter groups: one with weight_decay, one without
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 1e-2
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

    # Linear weight should have weight decay
    decay_params = set(optimizer.param_groups[0]["params"])
    assert model.fc.weight in decay_params

    # Linear bias and norm parameters should have 0.0 weight decay
    no_decay_params = set(optimizer.param_groups[1]["params"])
    assert model.fc.bias in no_decay_params
    assert model.norm.weight in no_decay_params
    assert model.norm.bias in no_decay_params


def test_lr_scheduler_warmup_and_cosine():
    model = SimpleModule()
    config = TrainingConfig(learning_rate=1e-3, warmup_steps=10, min_lr_ratio=0.1)
    optimizer = create_optimizer(model, config)
    scheduler = create_lr_scheduler(optimizer, config, total_steps=100)

    # Step 0: warmup initial step
    lr_0 = scheduler.get_last_lr()[0]
    assert lr_0 == pytest.approx(1e-4)

    # Step 9: end of warmup
    for _ in range(9):
        optimizer.step()
        scheduler.step()
    lr_warmup_end = scheduler.get_last_lr()[0]
    assert lr_warmup_end == pytest.approx(1e-3)

    # Step 99: final step should reach approximately min_lr_ratio * 1e-3 = 1e-4
    for _ in range(90):
        optimizer.step()
        scheduler.step()
    lr_final = scheduler.get_last_lr()[0]
    assert lr_final == pytest.approx(1e-4, abs=5e-6)



def test_clip_gradients():
    model = SimpleModule()
    x = torch.randn(5, 10)
    out = model(x).sum()
    out.backward()

    # Artificially scale gradients
    for p in model.parameters():
        p.grad.data.mul_(100.0)

    grad_norm = clip_gradients(model, max_norm=1.0)
    assert grad_norm > 1.0

    # Recomputed norm after clipping should be <= 1.0
    post_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]),
        2,
    )
    assert post_norm.item() <= 1.0 + 1e-4
