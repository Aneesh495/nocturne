"""Unit tests for VariationalLatentDynamicsModel forward rollout and inference modes."""

import pytest
import torch

from nocturne.models.vldm.config import VLDMConfig
from nocturne.models.vldm.model import VariationalLatentDynamicsModel


def test_vldm_autonomous_forward():
    config = VLDMConfig(
        obs_dim=15,
        latent_dim=4,
        encoder_hidden_dim=32,
        decoder_hidden_dim=32,
        use_inferred_drive=False,
    )
    model = VariationalLatentDynamicsModel(config)

    B, T, N = 2, 20, 15
    counts = torch.randint(0, 4, (B, T, N)).float()
    obs_mask = torch.ones((B, T, N), dtype=torch.bool)
    tgt_mask = torch.ones((B, T, N), dtype=torch.bool)

    rates_hz, latents, loss, metrics = model(counts, observed_mask=obs_mask, target_mask=tgt_mask)

    assert rates_hz.shape == (B, T, N)
    assert latents.shape == (B, T, 4)
    assert torch.all(rates_hz >= 1e-4)
    assert loss.item() > 0.0
    assert "loss_recon" in metrics
    assert "loss_kl_z0" in metrics
    assert metrics["loss_drive_penalty"] == pytest.approx(0.0)


def test_vldm_driven_forward():
    config = VLDMConfig(
        obs_dim=12,
        latent_dim=4,
        encoder_hidden_dim=32,
        decoder_hidden_dim=32,
        use_inferred_drive=True,
        drive_dim=2,
    )
    model = VariationalLatentDynamicsModel(config)

    B, T, N = 2, 15, 12
    counts = torch.randint(0, 3, (B, T, N)).float()

    rates_hz, latents, loss, metrics = model(counts)

    assert rates_hz.shape == (B, T, N)
    assert latents.shape == (B, T, 4)
    assert loss.item() > 0.0
    assert metrics["loss_drive_penalty"] >= 0.0
