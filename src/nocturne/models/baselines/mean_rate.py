"""Baseline model B06: Independent training-only per-neuron empirical mean firing rate."""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

from nocturne.contracts.neurons import PopulationMetadata


class MeanRateBaseline(nn.Module):
    """Training-only per-neuron empirical mean rate prior.

    Computes the static mean firing rate for each neuron using only training-partition data:
        lambda_n = (sum_{t in Train} y_{t,n}) / (T_train * Delta_t)

    This model serves as the official reference baseline for held-out bits-per-spike scoring.
    """

    def __init__(self, num_neurons: int, default_rate_hz: float = 10.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.default_rate_hz = default_rate_hz
        self.register_buffer("mean_rates_hz", torch.full((num_neurons,), default_rate_hz, dtype=torch.float32))
        self.is_fitted = False

    def fit_from_counts(
        self,
        training_counts: np.ndarray | torch.Tensor,
        bin_width_s: float,
        mask: Optional[np.ndarray | torch.Tensor] = None,
    ) -> MeanRateBaseline:
        """Fit empirical mean firing rates from training counts.

        Args:
            training_counts: Array or tensor of shape [..., N] of integer counts.
            bin_width_s: Bin duration in seconds.
            mask: Optional boolean mask [..., N] where True indicates valid training observations.
        """
        if bin_width_s <= 0.0:
            raise ValueError(f"bin_width_s must be positive, got {bin_width_s}")

        if isinstance(training_counts, np.ndarray):
            counts_t = torch.from_numpy(training_counts).float()
        else:
            counts_t = training_counts.float()

        if counts_t.shape[-1] != self.num_neurons:
            raise ValueError(f"Counts channel dim {counts_t.shape[-1]} != num_neurons {self.num_neurons}")

        # Reshape to [-1, num_neurons]
        flat_counts = counts_t.reshape(-1, self.num_neurons)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_t = torch.from_numpy(mask).bool().reshape(-1, self.num_neurons)
            else:
                mask_t = mask.bool().reshape(-1, self.num_neurons)
            valid_counts = (flat_counts * mask_t.float()).sum(dim=0)
            valid_bins = mask_t.float().sum(dim=0)
            # Avoid division by zero
            denom = torch.clamp(valid_bins * bin_width_s, min=1e-6)
            rates = valid_counts / denom
            rates = torch.where(valid_bins > 0, rates, torch.full_like(rates, self.default_rate_hz))
        else:
            total_time_s = flat_counts.shape[0] * bin_width_s
            rates = flat_counts.sum(dim=0) / max(total_time_s, 1e-6)

        # Enforce positive rate floor
        rates = torch.clamp(rates, min=1e-4)
        self.mean_rates_hz.copy_(rates)
        self.is_fitted = True
        return self

    def forward(self, batch_size: int, num_time_bins: int) -> torch.Tensor:
        """Broadcast static fitted rates across requested batch and time dimensions [B, T, N]."""
        rates = self.mean_rates_hz[None, None, :]  # [1, 1, N]
        return rates.expand(batch_size, num_time_bins, self.num_neurons)

    def predict_numpy(self, num_time_bins: int) -> np.ndarray:
        """Return 2D NumPy array of static rates [T, N] in Hz."""
        rates = self.mean_rates_hz.cpu().numpy()
        return np.broadcast_to(rates[None, :], (num_time_bins, self.num_neurons)).copy()
