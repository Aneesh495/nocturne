"""Unit tests for VLDM ELBO components, gradient propagation, and drive diagnostics."""

import math
import pytest
import torch

from nocturne.models.vldm.config import VLDMConfig
from nocturne.models.vldm.diagnostics import DynamicsAuditor
from nocturne.models.vldm.elbo import VLDMELBOCalculator
from nocturne.models.vldm.model import VariationalLatentDynamicsModel


def test_vldm_kl_z0_analytic():
    config = VLDMConfig(obs_dim=10, latent_dim=3)
    calculator = VLDMELBOCalculator(config)

    # When mu = 0 and log_var = 0 (std = 1), KL should be exactly 0
    mu_zero = torch.zeros(2, 3)
    log_var_zero = torch.zeros(2, 3)
    kl_zero = calculator.compute_kl_z0(mu_zero, log_var_zero)
    assert kl_zero.item() == pytest.approx(0.0, abs=1e-6)

    # When mu = 1 and log_var = 0: 0.5 * (1 + 1^2 - 1 - 0) * 3 = 0.5 * 3 = 1.5
    mu_one = torch.ones(2, 3)
    kl_one = calculator.compute_kl_z0(mu_one, log_var_zero)
    assert kl_one.item() == pytest.approx(1.5, rel=1e-5)


def test_vldm_gradient_flow():
    config = VLDMConfig(
        obs_dim=8,
        latent_dim=3,
        encoder_hidden_dim=16,
        decoder_hidden_dim=16,
        use_inferred_drive=True,
    )
    model = VariationalLatentDynamicsModel(config)

    counts = torch.randint(0, 3, (2, 10, 8)).float()
    rates_hz, latents, loss, metrics = model(counts)

    # Backpropagate loss directly
    loss.backward()

    # Check non-zero gradients in encoder, transition model, and decoder
    assert model.decoder.net[-1].weight.grad is not None
    assert model.decoder.net[-1].weight.grad.abs().sum().item() > 0.0
    assert model.transitions.net[0].weight.grad is not None
    assert model.transitions.net[0].weight.grad.abs().sum().item() > 0.0
    assert model.encoder.input_proj.weight.grad is not None
    assert model.encoder.input_proj.weight.grad.abs().sum().item() > 0.0


def test_dynamics_drive_audit():
    # Model with inferred drive disabled
    config_auto = VLDMConfig(obs_dim=6, latent_dim=2, use_inferred_drive=False)
    model_auto = VariationalLatentDynamicsModel(config_auto)
    counts = torch.randint(0, 3, (2, 10, 6)).float()

    audit_auto = DynamicsAuditor.audit_model(model_auto, counts)
    assert audit_auto.is_drive_dominated is False
    assert audit_auto.inferred_drive_norm == pytest.approx(0.0)
    assert audit_auto.autonomous_ratio == pytest.approx(1.0)
    assert "Strictly autonomous" in audit_auto.diagnostic_message
