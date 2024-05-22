"""Online causal filtering where state estimates depend strictly on past observations."""

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


class CausalFilter:
    """Performs strictly causal state and rate filtering with zero future leakage."""

    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Execute causal filtering inference on request.

        Args:
            request: InferenceRequest specifying counts and channel configurations.

        Returns:
            InferenceResult with strictly causal rates and latents.
        """
        self.model.eval()
        t0 = time.perf_counter()

        counts = request.counts
        B, T, N = counts.shape
        device = counts.device

        observed_mask = torch.ones((B, T, N), dtype=torch.bool, device=device)
        if request.held_out_indices is not None and len(request.held_out_indices) > 0:
            for ch in request.held_out_indices:
                observed_mask[:, :, ch] = False

        # Strictly enforce causal temporal attention/transitions
        rates_hz, latents, _, _ = self.model(
            counts=counts,
            observed_mask=observed_mask,
            causal=True,
        )

        held_out_slice: Optional[torch.Tensor] = None
        if request.held_out_indices is not None and len(request.held_out_indices) > 0:
            held_out_slice = rates_hz[:, :, request.held_out_indices]

        latency = time.perf_counter() - t0

        return InferenceResult(
            mode=InferenceMode.ONLINE_FILTERING,
            predicted_rates_hz=rates_hz,
            latents=latents,
            held_out_rates_hz=held_out_slice,
            latency_seconds=latency,
        )
