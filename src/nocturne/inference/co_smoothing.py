"""Offline co-smoothing inference across held-in and held-out neural populations."""

from __future__ import annotations

import time
from typing import List, Optional
import torch
import torch.nn as nn

from nocturne.inference.contracts import (
    InferenceMode,
    InferenceRequest,
    InferenceResult,
)


class CoSmoother:
    """Performs offline bidirectional co-smoothing using all observed time bins."""

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute co-smoothing inference on request.

        Args:
            request: InferenceRequest specifying counts, held-in, and held-out channels.

        Returns:
            InferenceResult with predicted rates and held-out slice.
        """
        self.model.eval()
        t0 = time.perf_counter()

        counts = request.counts
        B, T, N = counts.shape
        device = counts.device

        # Build observed mask
        observed_mask = torch.ones((B, T, N), dtype=torch.bool, device=device)
        if request.held_out_indices is not None and len(request.held_out_indices) > 0:
            for ch in request.held_out_indices:
                observed_mask[:, :, ch] = False

        # Run forward pass through model in bidirectional mode
        # Supports both STA and VLDM
        if hasattr(self.model, "causal") or "causal" in getattr(self.model, "config", {}).__dict__:
            rates_hz, latents, _, _ = self.model(
                counts=counts,
                observed_mask=observed_mask,
                causal=False,
            )
        else:
            rates_hz, latents, _, _ = self.model(
                counts=counts,
                observed_mask=observed_mask,
            )

        held_out_slice: Optional[torch.Tensor] = None
        if request.held_out_indices is not None and len(request.held_out_indices) > 0:
            held_out_slice = rates_hz[:, :, request.held_out_indices]

        latency = time.perf_counter() - t0

        return InferenceResult(
            mode=InferenceMode.OFFLINE_SMOOTHING,
            predicted_rates_hz=rates_hz,
            latents=latents,
            held_out_rates_hz=held_out_slice,
            latency_seconds=latency,
        )
