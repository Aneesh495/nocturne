"""Tests for batch collation, variable-length padding, and masking strategies."""

from __future__ import annotations

import torch
import pytest

from nocturne.training.collator import NeuralBatchCollator


def test_collator_variable_length_padding():
    collator = NeuralBatchCollator(mask_rate=0.0)
    # 3 trials with lengths 10, 15, 20
    trial1 = (torch.ones((10, 8)), torch.zeros((10, 2)))
    trial2 = (torch.ones((15, 8)) * 2, torch.zeros((15, 2)))
    trial3 = (torch.ones((20, 8)) * 3, torch.zeros((20, 2)))

    batch = collator([trial1, trial2, trial3])

    assert batch.counts.shape == (3, 20, 8)
    assert batch.lengths.tolist() == [10, 15, 20]
    # Padding mask should be False for valid steps, True for padded steps
    assert not batch.padding_mask[0, 9]
    assert batch.padding_mask[0, 10]
    assert batch.kinematics is not None
    assert batch.kinematics.shape == (3, 20, 2)


def test_collator_token_masking():
    collator = NeuralBatchCollator(mask_rate=0.25, mask_strategy="token")
    trial = (torch.ones((50, 10)), None)
    batch = collator([trial, trial])

    assert batch.counts.shape == (2, 50, 10)
    # Observed mask should have roughly 75% True and 25% False
    observed_ratio = batch.observed_mask.float().mean().item()
    assert 0.65 < observed_ratio < 0.85
    # Loss mask should evaluate masked positions
    assert (batch.loss_mask == (~batch.observed_mask)).all()


def test_collator_channel_masking():
    collator = NeuralBatchCollator(mask_rate=0.3, mask_strategy="channel")
    trial = (torch.ones((30, 10)), None)
    batch = collator([trial])

    # 3 out of 10 channels should be masked for trial 0
    # For any masked channel, it must be masked across all time steps
    masked_channels = (~batch.observed_mask[0, 0, :]).sum().item()
    assert masked_channels == 3
    # Check that mask is constant along time
    for t in range(30):
        assert (~batch.observed_mask[0, t, :]).sum().item() == 3


def test_collator_temporal_tail_masking():
    collator = NeuralBatchCollator(mask_rate=0.2, mask_strategy="temporal_tail")
    trial = (torch.ones((50, 10)), None)
    batch = collator([trial])

    # 20% of 50 is 10 bins: bins 40..49 should be masked
    assert batch.observed_mask[0, :40, :].all()
    assert not batch.observed_mask[0, 40:, :].any()
