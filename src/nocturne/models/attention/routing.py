"""Mask generation and routing utilities for attention-based neural modeling."""

from __future__ import annotations

import torch


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate boolean upper-triangular causal attention mask.

    Positions where mask is True cannot be attended to.
    For query i and key j: True if j > i (future cannot be seen).

    Args:
        seq_len: Temporal sequence length T.
        device: Target torch device.

    Returns:
        Boolean mask tensor of shape [T, T].
    """
    return torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device), diagonal=1)


def generate_cosmoothing_mask(
    batch_size: int,
    num_neurons: int,
    held_out_ratio: float = 0.2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate channel-level mask for co-smoothing evaluation.

    Args:
        batch_size: Number of trials B.
        num_neurons: Number of neurons N.
        held_out_ratio: Fraction of neurons to mask out for prediction.
        device: Target torch device.

    Returns:
        Boolean channel mask [B, N] where True indicates masked / held-out neuron.
    """
    mask = torch.zeros((batch_size, num_neurons), dtype=torch.bool, device=device)
    num_held_out = max(1, int(num_neurons * held_out_ratio))
    for b in range(batch_size):
        perm = torch.randperm(num_neurons, device=device)
        mask[b, perm[:num_held_out]] = True
    return mask


def generate_forecasting_mask(
    batch_size: int,
    seq_len: int,
    horizon_bins: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate temporal mask where future horizon bins are masked out.

    Args:
        batch_size: Number of trials B.
        seq_len: Total sequence length T.
        horizon_bins: Number of final bins to mask out.
        device: Target torch device.

    Returns:
        Boolean temporal mask [B, T] where True indicates masked future bin.
    """
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    cutoff = max(0, seq_len - horizon_bins)
    mask[:, cutoff:] = True
    return mask
