"""Inference mode declarations, request structures, and result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union
import numpy as np
import torch


class InferenceMode(str, Enum):
    """Execution protocol for neural population dynamics estimation."""

    OFFLINE_SMOOTHING = "offline_smoothing"  # Bidirectional: observes all t in [1, T]
    ONLINE_FILTERING = "online_filtering"    # Strictly causal: state at t uses only observations <= t
    FORECASTING = "forecasting"              # Causal conditioning on prefix, roll forward into future


@dataclass(frozen=True)
class InferenceRequest:
    """Standardized input request for neural inference pipelines."""

    counts: torch.Tensor                     # [B, T, N] spike counts
    mode: InferenceMode
    held_in_indices: Optional[List[int]] = None    # Neural indices provided as inputs
    held_out_indices: Optional[List[int]] = None   # Neural indices to be evaluated/predicted
    forecast_horizon_bins: int = 0                 # Number of future bins to forecast
    bin_width_s: float = 0.02

    def __post_init__(self):
        if self.counts.ndim != 3:
            raise ValueError(f"counts must have 3 dimensions [B, T, N], got {self.counts.shape}")
        if self.counts.min() < 0:
            raise ValueError("counts cannot contain negative values")
        if self.forecast_horizon_bins < 0:
            raise ValueError("forecast_horizon_bins must be non-negative")


@dataclass(frozen=True)
class InferenceResult:
    """Standardized output container for inference results."""

    mode: InferenceMode
    predicted_rates_hz: torch.Tensor          # [B, T, N] or [B, T_future, N] rates in Hz
    latents: torch.Tensor                     # [B, T, D] inferred latent state trajectory
    held_out_rates_hz: Optional[torch.Tensor] = None  # [B, T, N_held_out] rates
    latency_seconds: float = 0.0              # Execution wall time
