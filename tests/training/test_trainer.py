"""Tests for NeuralDynamicsTrainer fit execution and history logging."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)
from nocturne.training import (
    TrainingConfig,
    NeuralDynamicsTrainer,
)


def test_trainer_fit_attention_model(tmp_path):
    torch.manual_seed(42)
    B = 4
    T = 15
    N = 8

    # Generate small synthetic training and validation trials
    train_trials = [
        (torch.randint(0, 4, (T, N), dtype=torch.float32), None)
        for _ in range(8)
    ]
    val_trials = [
        (torch.randint(0, 4, (T, N), dtype=torch.float32), None)
        for _ in range(4)
    ]

    model_config = AttentionConfig(
        obs_dim=N,
        d_model=16,
        num_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=32,
    )
    model = SpatiotemporalAttentionModel(model_config, latent_dim=4)

    training_config = TrainingConfig(
        max_epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        warmup_steps=1,
        checkpoint_dir=str(tmp_path / "ckpts"),
        device="cpu",
    )

    trainer = NeuralDynamicsTrainer(
        model=model,
        config=training_config,
        train_data=train_trials,
        val_data=val_trials,
        bin_width_s=0.02,
    )

    history, best_ckpt = trainer.fit()

    assert len(history.epochs) == 2
    assert len(history.train_loss) == 2
    assert len(history.val_loss) == 2
    assert best_ckpt is not None
    assert best_ckpt.exists()
    assert (tmp_path / "ckpts" / "latest.pt").exists()
    assert (tmp_path / "ckpts" / "best.pt").exists()
