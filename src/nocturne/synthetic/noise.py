"""Controlled observation noise, neuron dropout, and model misspecification regimes."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class NoiseRegimeConfig(BaseModel):
    """Specification of corruption, dropout, and unobserved drive regimes."""

    model_config = ConfigDict(frozen=True)

    dropout_fraction: float = Field(default=0.0, ge=0.0, le=0.95, description="Fraction of neurons randomly dropped")
    hide_external_input: bool = Field(default=False, description="Whether task input drive u(t) is hidden from model")
    rate_jitter_sigma: float = Field(default=0.0, ge=0.0, description="Standard deviation of log-rate multiplicative noise")
    overdispersion_theta: Optional[float] = Field(default=None, gt=0.0, description="Negative binomial dispersion parameter")


class MisspecificationCorruptor:
    """Applies controlled corruptions and observation restrictions to synthetic data."""

    def __init__(self, config: NoiseRegimeConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

    def apply_neuron_dropout(self, num_neurons: int) -> Tuple[List[int], List[int]]:
        """Partition neurons into observed and dropped channels according to dropout_fraction."""
        frac = self.config.dropout_fraction
        if frac <= 0.0:
            return list(range(num_neurons)), []

        num_drop = int(round(num_neurons * frac))
        perm = self.rng.permutation(num_neurons)
        dropped_indices = sorted(list(perm[:num_drop]))
        observed_indices = sorted(list(perm[num_drop:]))
        return observed_indices, dropped_indices

    def corrupt_firing_rates(self, true_rates_hz: np.ndarray) -> np.ndarray:
        """Apply multiplicative log-normal rate jitter if configured."""
        if self.config.rate_jitter_sigma <= 0.0:
            return true_rates_hz

        sigma = self.config.rate_jitter_sigma
        jitter = np.exp(self.rng.normal(0.0, sigma, size=true_rates_hz.shape))
        return np.maximum(true_rates_hz * jitter, 1e-4)

    def filter_external_inputs(self, inputs: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Return None if external input drive is designated as hidden/unobserved."""
        if self.config.hide_external_input or inputs is None:
            return None
        return inputs.copy()
