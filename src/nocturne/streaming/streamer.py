"""Online causal streaming engine with bounded buffer and real-time execution."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

from nocturne.streaming.state import StreamingBufferState


@dataclass(frozen=True)
class StreamChunkResult:
    """Output metrics and predictions emitted for an incoming stream chunk."""

    predicted_rates_hz: torch.Tensor    # [1, delta_T, obs_dim]
    latents: torch.Tensor               # [1, delta_T, latent_dim]
    latency_ms: float                   # Wall-clock processing time in milliseconds
    total_bins_processed: int           # Cumulative bins processed in this streaming session


class OnlineCausalStreamer:
    """Real-time causal streaming coordinator operating over bounded history windows."""

    def __init__(
        self,
        model: nn.Module,
        obs_dim: int,
        max_window_bins: int = 100,
        device: torch.device | None = None,
    ):
        self.model = model
        self.obs_dim = obs_dim
        self.max_window_bins = max_window_bins
        self.device = device or next(model.parameters()).device

        self.model.eval()
        self.state = StreamingBufferState(
            obs_dim=obs_dim,
            max_window_bins=max_window_bins,
            device=self.device,
        )

    @torch.no_grad()
    def push_chunk(self, chunk: torch.Tensor) -> StreamChunkResult:
        """Process incoming neural count chunk causally.

        Args:
            chunk: Spike count tensor [1, delta_T, N] or [delta_T, N].

        Returns:
            StreamChunkResult with predictions for the new chunk slice and latency.
        """
        t0 = time.perf_counter()

        # Update sliding ring buffer
        window_counts, slice_start = self.state.push(chunk)
        delta_T = chunk.shape[1] if chunk.ndim == 3 else chunk.shape[0]

        # Execute causal forward pass on the active bounded window
        rates_hz, latents, _, _ = self.model(
            counts=window_counts,
            causal=True,
        )

        # Update cached latents in state
        self.state.window_latents = latents

        # Extract predictions for the newly received bins
        chunk_rates = rates_hz[:, slice_start : slice_start + delta_T, :]
        chunk_latents = latents[:, slice_start : slice_start + delta_T, :]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return StreamChunkResult(
            predicted_rates_hz=chunk_rates,
            latents=chunk_latents,
            latency_ms=latency_ms,
            total_bins_processed=self.state.total_received_bins,
        )

    def reset(self) -> None:
        """Reset streaming session."""
        self.state.reset()

    def checkpoint(self) -> Dict[str, Any]:
        """Capture atomic state checkpoint."""
        return self.state.capture_checkpoint()

    def restore(self, checkpoint_dict: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self.state.restore_checkpoint(checkpoint_dict)
