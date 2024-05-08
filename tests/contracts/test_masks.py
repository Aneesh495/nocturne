"""Unit tests for observation, held-out, padding, and forecast mask contracts."""

import pytest
import torch

from nocturne.contracts.masks import ObservationMasks


def test_co_smoothing_masks():
    B, T, N = 2, 10, 5
    held_in = [0, 1, 2]
    held_out = [3, 4]
    masks = ObservationMasks.create_co_smoothing_masks(
        batch_size=B,
        num_time_bins=T,
        num_neurons=N,
        held_in_indices=held_in,
        held_out_indices=held_out,
    )

    assert masks.observed_mask.shape == (B, T, N)
    assert masks.target_mask.shape == (B, T, N)
    assert masks.padding_mask.shape == (B, T)
    assert masks.channel_mask.shape == (B, N)

    # Observed mask should be True only on held_in channels
    assert torch.all(masks.observed_mask[:, :, [0, 1, 2]])
    assert not torch.any(masks.observed_mask[:, :, [3, 4]])

    # Target mask should be True only on held_out channels
    assert torch.all(masks.target_mask[:, :, [3, 4]])
    assert not torch.any(masks.target_mask[:, :, [0, 1, 2]])

    # Total evaluated observations: 2 batches * 10 bins * 2 held_out channels = 40
    assert masks.num_evaluated_observations == 40


def test_co_smoothing_masks_with_padding():
    B, T, N = 2, 10, 4
    held_in = [0, 1]
    held_out = [2, 3]
    # batch 0 has 6 valid bins, batch 1 has 8 valid bins
    valid_lengths = [6, 8]
    masks = ObservationMasks.create_co_smoothing_masks(
        batch_size=B,
        num_time_bins=T,
        num_neurons=N,
        held_in_indices=held_in,
        held_out_indices=held_out,
        valid_lengths=valid_lengths,
    )

    # Evaluated observations: (6 * 2) + (8 * 2) = 12 + 16 = 28
    assert masks.num_evaluated_observations == 28
    assert masks.padding_mask[0, 5].item() is True
    assert masks.padding_mask[0, 6].item() is False


def test_forecast_masks():
    B, T, N = 1, 20, 4
    prefix_len = 12
    forecast_len = 8
    masks = ObservationMasks.create_forecast_masks(
        batch_size=B,
        num_time_bins=T,
        num_neurons=N,
        prefix_len=prefix_len,
        forecast_len=forecast_len,
        observed_channels=[0, 1, 2, 3],
        target_channels=[0, 1, 2, 3],
    )

    # Input observations only in [0, 12)
    assert torch.all(masks.observed_mask[:, :12, :])
    assert not torch.any(masks.observed_mask[:, 12:, :])

    # Target observations only in [12, 20)
    assert not torch.any(masks.target_mask[:, :12, :])
    assert torch.all(masks.target_mask[:, 12:20, :])

    # 1 batch * 8 forecast bins * 4 channels = 32
    assert masks.num_evaluated_observations == 32


def test_mask_validation_errors():
    B, T, N = 2, 5, 3
    obs = torch.zeros((B, T, N), dtype=torch.bool)
    tgt = torch.zeros((B, T, N), dtype=torch.bool)
    pad = torch.zeros((B, T), dtype=torch.bool)
    chan = torch.zeros((B, N), dtype=torch.bool)

    # Wrong shape
    with pytest.raises(ValueError, match="observed_mask must have shape"):
        ObservationMasks(
            batch_size=B,
            num_time_bins=T,
            num_neurons=N,
            observed_mask=torch.zeros((B, T + 1, N), dtype=torch.bool),
            target_mask=tgt,
            padding_mask=pad,
            channel_mask=chan,
        )

    # Wrong dtype
    with pytest.raises(ValueError, match="torch.bool dtype"):
        ObservationMasks(
            batch_size=B,
            num_time_bins=T,
            num_neurons=N,
            observed_mask=torch.zeros((B, T, N), dtype=torch.float32),
            target_mask=tgt,
            padding_mask=pad,
            channel_mask=chan,
        )
