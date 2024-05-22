"""Bounded sliding ring buffer and streaming lifecycle state management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch


class StreamingBufferState:
    """Bounded sliding ring buffer maintaining recent neural history.

    Guarantees strictly bounded resident memory regardless of session duration:
    Memory footprint remains O(max_window_bins * N * d_model) rather than O(T_session).
    """

    def __init__(
        self,
        obs_dim: int,
        max_window_bins: int = 100,
        device: torch.device = torch.device("cpu"),
    ):
        if max_window_bins < 1:
            raise ValueError(f"max_window_bins must be >= 1, got {max_window_bins}")
        if obs_dim < 1:
            raise ValueError(f"obs_dim must be >= 1, got {obs_dim}")

        self.obs_dim = obs_dim
        self.max_window_bins = max_window_bins
        self.device = device

        self.total_received_bins: int = 0
        self.total_received_chunks: int = 0
        self.window_counts: torch.Tensor = torch.zeros(
            (1, 0, obs_dim), dtype=torch.float32, device=device
        )
        self.window_latents: Optional[torch.Tensor] = None

    def push(self, chunk: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Append incoming spike count chunk and slide window if exceeding max_window_bins.

        Args:
            chunk: Input tensor of shape [1, delta_T, obs_dim] or [delta_T, obs_dim].

        Returns:
            Tuple of:
                current_window: Active count buffer of shape [1, T_active, obs_dim].
                slice_start_idx: Start index in current_window corresponding to incoming chunk.
        """
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != self.obs_dim:
            raise ValueError(
                f"Expected chunk shape [1, delta_T, {self.obs_dim}], got {chunk.shape}"
            )

        chunk = chunk.to(self.device, dtype=torch.float32)
        delta_T = chunk.shape[1]

        # Concatenate along temporal axis
        combined = torch.cat([self.window_counts, chunk], dim=1)

        # Slide window if length exceeds max_window_bins
        if combined.shape[1] > self.max_window_bins:
            overflow = combined.shape[1] - self.max_window_bins
            self.window_counts = combined[:, overflow:, :].clone()
        else:
            self.window_counts = combined

        slice_start = max(0, self.window_counts.shape[1] - delta_T)


        self.total_received_bins += delta_T
        self.total_received_chunks += 1

        return self.window_counts, slice_start

    def reset(self) -> None:
        """Reset buffer state to pristine initial empty state."""
        self.total_received_bins = 0
        self.total_received_chunks = 0
        self.window_counts = torch.zeros(
            (1, 0, self.obs_dim), dtype=torch.float32, device=self.device
        )
        self.window_latents = None

    def capture_checkpoint(self) -> Dict[str, Any]:
        """Serialize state for atomic recovery."""
        return {
            "obs_dim": self.obs_dim,
            "max_window_bins": self.max_window_bins,
            "total_received_bins": self.total_received_bins,
            "total_received_chunks": self.total_received_chunks,
            "window_counts": self.window_counts.cpu().clone(),
            "window_latents": self.window_latents.cpu().clone() if self.window_latents is not None else None,
        }

    def restore_checkpoint(self, state: Dict[str, Any]) -> None:
        """Restore state from serialized checkpoint."""
        if state["obs_dim"] != self.obs_dim:
            raise ValueError(f"obs_dim mismatch: {state['obs_dim']} != {self.obs_dim}")
        self.max_window_bins = state["max_window_bins"]
        self.total_received_bins = state["total_received_bins"]
        self.total_received_chunks = state["total_received_chunks"]
        self.window_counts = state["window_counts"].to(self.device)
        self.window_latents = state["window_latents"].to(self.device) if state.get("window_latents") is not None else None
