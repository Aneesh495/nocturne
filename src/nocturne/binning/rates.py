"""Firing rate transformations, variance-stabilizing Anscombe transforms, and inverse softplus math."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field


class RateClippingStats(BaseModel):
    """Diagnostic tracking of rate clipping activations."""

    model_config = ConfigDict(frozen=True)

    total_values: int = Field(..., ge=0)
    clipped_low_count: int = Field(..., ge=0)
    clipped_high_count: int = Field(..., ge=0)
    min_observed_rate_hz: float = Field(...)
    max_observed_rate_hz: float = Field(...)

    @property
    def percent_clipped_low(self) -> float:
        return 0.0 if self.total_values == 0 else 100.0 * (self.clipped_low_count / self.total_values)

    @property
    def percent_clipped_high(self) -> float:
        return 0.0 if self.total_values == 0 else 100.0 * (self.clipped_high_count / self.total_values)


class RateTransformer:
    """Provides mathematically stable rate transforms, inverse link functions, and Anscombe variance stabilization."""

    def __init__(
        self,
        min_rate_hz: float = 1e-4,
        max_rate_hz: float = 2000.0,
        softplus_beta: float = 1.0,
    ):
        if min_rate_hz <= 0.0:
            raise ValueError(f"min_rate_hz must be strictly positive, got {min_rate_hz}")
        if max_rate_hz <= min_rate_hz:
            raise ValueError(f"max_rate_hz ({max_rate_hz}) must be greater than min_rate_hz ({min_rate_hz})")

        self.min_rate_hz = min_rate_hz
        self.max_rate_hz = max_rate_hz
        self.softplus_beta = softplus_beta

    def clip_rates_numpy(self, rates_hz: np.ndarray) -> Tuple[np.ndarray, RateClippingStats]:
        """Clip firing rates in Hz to [min_rate_hz, max_rate_hz] and record activation stats."""
        total = rates_hz.size
        min_obs = float(np.min(rates_hz)) if total > 0 else 0.0
        max_obs = float(np.max(rates_hz)) if total > 0 else 0.0

        low_mask = rates_hz < self.min_rate_hz
        high_mask = rates_hz > self.max_rate_hz

        clipped_low = int(np.sum(low_mask))
        clipped_high = int(np.sum(high_mask))

        clipped = np.clip(rates_hz, self.min_rate_hz, self.max_rate_hz)
        stats = RateClippingStats(
            total_values=total,
            clipped_low_count=clipped_low,
            clipped_high_count=clipped_high,
            min_observed_rate_hz=min_obs,
            max_observed_rate_hz=max_obs,
        )
        return clipped, stats

    def clip_rates_torch(self, rates_hz: torch.Tensor) -> Tuple[torch.Tensor, RateClippingStats]:
        """PyTorch tensor clipping with statistics extraction."""
        total = rates_hz.numel()
        min_obs = float(torch.min(rates_hz).item()) if total > 0 else 0.0
        max_obs = float(torch.max(rates_hz).item()) if total > 0 else 0.0

        low_count = int((rates_hz < self.min_rate_hz).sum().item())
        high_count = int((rates_hz > self.max_rate_hz).sum().item())

        clipped = torch.clamp(rates_hz, min=self.min_rate_hz, max=self.max_rate_hz)
        stats = RateClippingStats(
            total_values=total,
            clipped_low_count=low_count,
            clipped_high_count=high_count,
            min_observed_rate_hz=min_obs,
            max_observed_rate_hz=max_obs,
        )
        return clipped, stats

    def softplus(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Numerically stable softplus: (1/beta) * ln(1 + exp(beta * x))."""
        beta = self.softplus_beta
        if isinstance(x, torch.Tensor):
            return torch.nn.functional.softplus(x, beta=beta)
        # NumPy implementation: for beta*x > 20, softplus(x) approx x
        bx = beta * x
        large = bx > 20.0
        out = np.empty_like(bx, dtype=np.float64)
        out[large] = x[large]
        out[~large] = np.log1p(np.exp(bx[~large])) / beta
        return out

    def inverse_softplus(self, y: np.ndarray | torch.Tensor, eps: float = 1e-7) -> np.ndarray | torch.Tensor:
        """Exact inverse softplus: x = (1/beta) * ln(exp(beta * y) - 1).

        Used to initialize linear decoder biases given empirical mean firing rates.
        """
        beta = self.softplus_beta
        if isinstance(y, torch.Tensor):
            y_clamped = torch.clamp(y, min=eps)
            by = beta * y_clamped
            large = by > 20.0
            out = torch.empty_like(y_clamped)
            out[large] = y_clamped[large]
            out[~large] = torch.log(torch.expm1(by[~large])) / beta
            return out

        y_arr = np.maximum(np.asarray(y, dtype=np.float64), eps)
        by = beta * y_arr
        large = by > 20.0
        out = np.empty_like(y_arr)
        out[large] = y_arr[large]
        out[~large] = np.log(np.expm1(by[~large])) / beta
        return out

    @staticmethod
    def anscombe_transform(counts: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Anscombe variance-stabilizing transformation for Poisson counts: 2 * sqrt(y + 3/8).

        Transforms Poisson variable with mean mu and variance mu into approximately Gaussian
        variable with variance 1.0 (for mu >= 4).
        """
        if isinstance(counts, torch.Tensor):
            return 2.0 * torch.sqrt(torch.clamp(counts, min=0.0) + 0.375)
        counts_arr = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
        return 2.0 * np.sqrt(counts_arr + 0.375)

    @staticmethod
    def inverse_anscombe_transform(z: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Algebraic inverse of Anscombe transform: (z/2)^2 - 3/8."""
        if isinstance(z, torch.Tensor):
            return torch.clamp((0.25 * (z**2)) - 0.375, min=0.0)
        z_arr = np.asarray(z, dtype=np.float64)
        return np.maximum((0.25 * (z_arr**2)) - 0.375, 0.0)
