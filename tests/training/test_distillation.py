"""Tests for knowledge distillation engine and generalized I-divergence loss."""

import pytest
import torch
import torch.nn as nn
from nocturne.training.distillation import (
    DistillationLoss,
    DistillationLossOutput,
    KnowledgeDistiller,
)


class DummyNeuralModel(nn.Module):
    def __init__(self, in_features: int = 10, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),
            nn.Softplus(),
        )
        self.latent_proj = nn.Linear(in_features, 8)

    def forward(self, x: torch.Tensor):
        rates = self.net(x)
        latents = self.latent_proj(x)
        return rates, latents


def test_distillation_loss_identical_rates_zero_divergence():
    loss_fn = DistillationLoss(alpha=0.5, beta=0.1)
    rates = torch.rand(4, 20, 10) + 0.1

    # Divergence of identical rates should be 0.0
    div = loss_fn.compute_rate_divergence(rates, rates)
    assert div.item() == pytest.approx(0.0, abs=1e-5)


def test_distillation_loss_forward_and_gradients():
    loss_fn = DistillationLoss(alpha=0.6, beta=0.2)
    s_rates = torch.rand(4, 15, 8, requires_grad=True)
    t_rates = torch.rand(4, 15, 8)
    targets = torch.poisson(s_rates.detach())
    s_lat = torch.randn(4, 15, 6, requires_grad=True)
    t_lat = torch.randn(4, 15, 6)

    out: DistillationLossOutput = loss_fn(
        student_rates=s_rates,
        teacher_rates=t_rates,
        targets=targets,
        student_latents=s_lat,
        teacher_latents=t_lat,
    )

    assert out.total_loss.ndim == 0
    assert out.total_loss.item() > 0.0
    assert out.task_loss.item() > 0.0
    assert out.rate_distill_loss.item() > 0.0
    assert out.latent_distill_loss.item() > 0.0

    out.total_loss.backward()
    assert s_rates.grad is not None
    assert s_lat.grad is not None


def test_distillation_loss_with_latent_adapter():
    loss_fn = DistillationLoss(alpha=0.5, beta=0.1)
    s_rates = torch.rand(2, 10, 5, requires_grad=True)
    t_rates = torch.rand(2, 10, 5)
    targets = torch.zeros(2, 10, 5)

    s_lat = torch.randn(2, 10, 4)
    t_lat = torch.randn(2, 10, 8)  # Larger teacher latent dimension
    adapter = nn.Linear(4, 8)

    out = loss_fn(
        student_rates=s_rates,
        teacher_rates=t_rates,
        targets=targets,
        student_latents=s_lat,
        teacher_latents=t_lat,
        latent_adapter=adapter,
    )
    assert out.total_loss.item() > 0.0


def test_knowledge_distiller_train_step():
    teacher = DummyNeuralModel(in_features=8, hidden_dim=32)
    student = DummyNeuralModel(in_features=8, hidden_dim=16)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)

    distiller = KnowledgeDistiller(
        teacher_model=teacher,
        student_model=student,
        optimizer=opt,
    )

    x = torch.rand(4, 10, 8)
    targets = torch.poisson(x)

    metrics = distiller.train_step(inputs=x, targets=targets)
    assert "total_loss" in metrics
    assert "task_loss" in metrics
    assert "rate_distill_loss" in metrics
    assert metrics["total_loss"] > 0.0
