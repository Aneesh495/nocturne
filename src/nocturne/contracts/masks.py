"""Explicit observation, held-out, causal, forecast, and padding mask contracts."""

from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field


class ObservationMasks(BaseModel):
    """Collection of boolean masks governing input observation and loss target evaluation.

    Shapes:
        observed_mask: [B, T, N] or [T, N] (True where spikes are visible as model inputs)
        target_mask:   [B, T, N] or [T, N] (True where model predictions are scored in loss/metrics)
        padding_mask:  [B, T] or [T]       (True where time bins are valid non-padding observations)
        channel_mask:  [B, N] or [N]       (True where neuron channel is valid/unbroken)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    batch_size: int = Field(..., ge=1, description="Batch size B")
    num_time_bins: int = Field(..., ge=1, description="Number of temporal bins T")
    num_neurons: int = Field(..., ge=1, description="Number of neural channels N")

    # Boolean PyTorch tensors
    observed_mask: torch.Tensor = Field(..., description="[B, T, N] True if channel/bin is observed as input")
    target_mask: torch.Tensor = Field(..., description="[B, T, N] True if channel/bin is an evaluated target")
    padding_mask: torch.Tensor = Field(..., description="[B, T] True if bin is valid (non-padding)")
    channel_mask: torch.Tensor = Field(..., description="[B, N] True if channel is valid/active")

    def model_post_init(self, __context: object) -> None:
        B, T, N = self.batch_size, self.num_time_bins, self.num_neurons

        if self.observed_mask.shape != (B, T, N) or self.observed_mask.dtype != torch.bool:
            raise ValueError(f"observed_mask must have shape ({B}, {T}, {N}) and torch.bool dtype, got {self.observed_mask.shape}, {self.observed_mask.dtype}")
        if self.target_mask.shape != (B, T, N) or self.target_mask.dtype != torch.bool:
            raise ValueError(f"target_mask must have shape ({B}, {T}, {N}) and torch.bool dtype, got {self.target_mask.shape}, {self.target_mask.dtype}")
        if self.padding_mask.shape != (B, T) or self.padding_mask.dtype != torch.bool:
            raise ValueError(f"padding_mask must have shape ({B}, {T}) and torch.bool dtype, got {self.padding_mask.shape}, {self.padding_mask.dtype}")
        if self.channel_mask.shape != (B, N) or self.channel_mask.dtype != torch.bool:
            raise ValueError(f"channel_mask must have shape ({B}, {N}) and torch.bool dtype, got {self.channel_mask.shape}, {self.channel_mask.dtype}")

    @property
    def valid_target_mask(self) -> torch.Tensor:
        """Combined boolean mask of valid targets: target_mask & padding_mask[:, :, None] & channel_mask[:, None, :]."""
        return self.target_mask & self.padding_mask.unsqueeze(-1) & self.channel_mask.unsqueeze(1)

    @property
    def num_evaluated_observations(self) -> int:
        """Total count of valid evaluated neuron-bin observations."""
        return int(self.valid_target_mask.sum().item())

    @classmethod
    def create_co_smoothing_masks(
        cls,
        batch_size: int,
        num_time_bins: int,
        num_neurons: int,
        held_in_indices: Sequence[int],
        held_out_indices: Sequence[int],
        valid_lengths: Sequence[int] | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> ObservationMasks:
        """Create standard offline co-smoothing masks.

        Held-in neurons are visible as inputs across all valid time bins.
        Held-out neurons are strictly unobserved (masked to 0 in input) and evaluated as targets.
        """
        observed = torch.zeros((batch_size, num_time_bins, num_neurons), dtype=torch.bool, device=device)
        target = torch.zeros((batch_size, num_time_bins, num_neurons), dtype=torch.bool, device=device)
        padding = torch.zeros((batch_size, num_time_bins), dtype=torch.bool, device=device)
        channels = torch.ones((batch_size, num_neurons), dtype=torch.bool, device=device)

        for b in range(batch_size):
            length = valid_lengths[b] if valid_lengths is not None else num_time_bins
            padding[b, :length] = True
            for ch in held_in_indices:
                observed[b, :length, ch] = True
            for ch in held_out_indices:
                target[b, :length, ch] = True

        return cls(
            batch_size=batch_size,
            num_time_bins=num_time_bins,
            num_neurons=num_neurons,
            observed_mask=observed,
            target_mask=target,
            padding_mask=padding,
            channel_mask=channels,
        )

    @classmethod
    def create_causal_filtering_masks(
        cls,
        batch_size: int,
        num_time_bins: int,
        num_neurons: int,
        observed_channels: Sequence[int],
        target_channels: Sequence[int],
        valid_lengths: Sequence[int] | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> ObservationMasks:
        """Create causal filtering masks where observed channels are visible as inputs up to current bin."""
        observed = torch.zeros((batch_size, num_time_bins, num_neurons), dtype=torch.bool, device=device)
        target = torch.zeros((batch_size, num_time_bins, num_neurons), dtype=torch.bool, device=device)
        padding = torch.zeros((batch_size, num_time_bins), dtype=torch.bool, device=device)
        channels = torch.ones((batch_size, num_neurons), dtype=torch.bool, device=device)

        for b in range(batch_size):
            length = valid_lengths[b] if valid_lengths is not None else num_time_bins
            padding[b, :length] = True
            for ch in observed_channels:
                observed[b, :length, ch] = True
            for ch in target_channels:
                target[b, :length, ch] = True

        return cls(
            batch_size=batch_size,
            num_time_bins=num_time_bins,
            num_neurons=num_neurons,
            observed_mask=observed,
            target_mask=target,
            padding_mask=padding,
            channel_mask=channels,
        )

    @classmethod
    def create_forecast_masks(
        cls,
        batch_size: int,
        num_time_bins: int,
        num_neurons: int,
        prefix_len: int,
        forecast_len: int,
        observed_channels: Sequence[int],
        target_channels: Sequence[int],
        device: torch.device = torch.device("cpu"),
    ) -> ObservationMasks:
        """Create forecasting masks where input is restricted to [0, prefix_len) and targets are in [prefix_len, prefix_len + forecast_len)."""
        if prefix_len + forecast_len > num_time_bins:
            raise ValueError(f"prefix_len ({prefix_len}) + forecast_len ({forecast_len}) > num_time_bins ({num_time_bins})")

        observed = torch.zeros((batch_size, num_time_bins, num_neurons), dtype=torch.bool, device=device)
        target = torch.zeros((batch_size, num_time_bins, num_neurons), dtype=torch.bool, device=device)
        padding = torch.zeros((batch_size, num_time_bins), dtype=torch.bool, device=device)
        channels = torch.ones((batch_size, num_neurons), dtype=torch.bool, device=device)

        total_valid = prefix_len + forecast_len
        padding[:, :total_valid] = True

        for ch in observed_channels:
            observed[:, :prefix_len, ch] = True

        for ch in target_channels:
            target[:, prefix_len:total_valid, ch] = True

        return cls(
            batch_size=batch_size,
            num_time_bins=num_time_bins,
            num_neurons=num_neurons,
            observed_mask=observed,
            target_mask=target,
            padding_mask=padding,
            channel_mask=channels,
        )
