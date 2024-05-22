"""Autoregressive future horizon forecasting from conditioned history prefix."""

from __future__ import annotations

import time
from typing import Optional
import torch
import torch.nn as nn

from nocturne.inference.contracts import (
    InferenceMode,
    InferenceRequest,
    InferenceResult,
)


class AutoregressiveForecaster:
    """Rolls forward latent dynamics to forecast future population activity."""

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute forecasting rollout over request.forecast_horizon_bins.

        Args:
            request: InferenceRequest with positive forecast_horizon_bins.

        Returns:
            InferenceResult containing future forecasted firing rates and latent trajectory.
        """
        self.model.eval()
        t0 = time.perf_counter()

        counts = request.counts
        B, T_hist, N = counts.shape
        H = request.forecast_horizon_bins
        if H <= 0:
            raise ValueError(f"forecast_horizon_bins must be > 0, got {H}")

        device = counts.device
        dt = request.bin_width_s

        # Pad sequence with zeros for future horizon, marking future positions unobserved
        total_T = T_hist + H
        padded_counts = torch.zeros((B, total_T, N), dtype=torch.float32, device=device)
        padded_counts[:, :T_hist, :] = counts

        observed_mask = torch.zeros((B, total_T, N), dtype=torch.bool, device=device)
        observed_mask[:, :T_hist, :] = True

        # In causal mode, forward pass processes history and rolls through future tokens
        rates_hz, latents, _, _ = self.model(
            counts=padded_counts,
            observed_mask=observed_mask,
            causal=True,
        )

        # Slice future horizon predictions
        future_rates = rates_hz[:, T_hist:, :]
        future_latents = latents[:, T_hist:, :]

        held_out_future: Optional[torch.Tensor] = None
        if request.held_out_indices is not None and len(request.held_out_indices) > 0:
            held_out_future = future_rates[:, :, request.held_out_indices]

        latency = time.perf_counter() - t0

        return InferenceResult(
            mode=InferenceMode.FORECASTING,
            predicted_rates_hz=future_rates,
            latents=future_latents,
            held_out_rates_hz=held_out_future,
            latency_seconds=latency,
        )
