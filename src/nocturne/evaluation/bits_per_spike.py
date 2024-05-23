"""Strict Bits-Per-Spike (BPS) evaluation and zero-spike denominator handling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional
import numpy as np
import torch


@dataclass(frozen=True)
class BPSResult:
    """Outcome of Bits-Per-Spike evaluation against a baseline null rate."""

    population_bps: Optional[float]           # Global micro-averaged bits per spike
    mean_neuron_bps: Optional[float]          # Unweighted macro-average across active neurons
    per_neuron_bps: List[Optional[float]]     # Individual neuron BPS (None if 0 spikes)
    total_spikes: int                         # Total evaluated spike count
    active_neurons: int                       # Count of neurons with >= 1 spike
    num_observations: int                     # Total evaluated bin-neuron observations


class BitsPerSpikeEvaluator:
    """Evaluates Bits-Per-Spike improvement of model rates over null baseline.

    Formula:
        BPS = [ ln L(model) - ln L(null) ] / [ N_spikes * ln(2) ]

    Strict Contract:
        If total evaluated spikes N_spikes == 0, returns None (never divides by zero).
    """

    def __init__(self, bin_width_s: float = 0.02, eps: float = 1e-9):
        if bin_width_s <= 0.0:
            raise ValueError(f"bin_width_s must be > 0, got {bin_width_s}")
        self.bin_width_s = bin_width_s
        self.eps = eps

    def evaluate(
        self,
        predicted_rates_hz: np.ndarray | torch.Tensor,
        target_counts: np.ndarray | torch.Tensor,
        null_rates_hz: np.ndarray | torch.Tensor,
        mask: Optional[np.ndarray | torch.Tensor] = None,
    ) -> BPSResult:
        """Calculate population and per-neuron Bits-Per-Spike.

        Args:
            predicted_rates_hz: Predicted rates [..., N] in Hz.
            target_counts: Ground truth counts [..., N].
            null_rates_hz: Baseline rates [..., N] (e.g. training marginal mean).
            mask: Optional boolean mask [..., N] indicating valid evaluation bins.

        Returns:
            BPSResult with macro/micro BPS and per-neuron array.
        """
        # Convert to numpy
        if isinstance(predicted_rates_hz, torch.Tensor):
            pred_r = predicted_rates_hz.detach().cpu().numpy().astype(np.float64)
        else:
            pred_r = np.asarray(predicted_rates_hz, dtype=np.float64)

        if isinstance(target_counts, torch.Tensor):
            targs = target_counts.detach().cpu().numpy().astype(np.float64)
        else:
            targs = np.asarray(target_counts, dtype=np.float64)

        if isinstance(null_rates_hz, torch.Tensor):
            null_r = null_rates_hz.detach().cpu().numpy().astype(np.float64)
        else:
            null_r = np.asarray(null_rates_hz, dtype=np.float64)

        # Broadcast null_rates if passed as 1D [N]
        if null_r.ndim < pred_r.ndim:
            null_r = np.broadcast_to(null_r, pred_r.shape)

        num_neurons = pred_r.shape[-1]
        per_neuron_bps: List[Optional[float]] = []

        total_log_diff = 0.0
        total_spikes = 0
        total_obs = 0

        for n in range(num_neurons):
            y_n = targs[..., n].ravel()
            mu_model_n = np.clip(pred_r[..., n].ravel() * self.bin_width_s, a_min=self.eps, a_max=None)
            mu_null_n = np.clip(null_r[..., n].ravel() * self.bin_width_s, a_min=self.eps, a_max=None)

            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    m_n = mask[..., n].detach().cpu().numpy().astype(bool).ravel()
                else:
                    m_n = np.asarray(mask[..., n], dtype=bool).ravel()
                y_n = y_n[m_n]
                mu_model_n = mu_model_n[m_n]
                mu_null_n = mu_null_n[m_n]

            n_spikes = int(np.sum(y_n))
            total_obs += len(y_n)

            if n_spikes == 0:
                per_neuron_bps.append(None)
                continue

            # Log-likelihood terms: y * ln(mu) - mu (factorials cancel out)
            ll_model = np.sum(y_n * np.log(mu_model_n) - mu_model_n)
            ll_null = np.sum(y_n * np.log(mu_null_n) - mu_null_n)
            log_diff = ll_model - ll_null

            bps_n = float(log_diff / (n_spikes * math.log(2.0)))
            per_neuron_bps.append(bps_n)

            total_log_diff += log_diff
            total_spikes += n_spikes

        # Population micro-average
        pop_bps: Optional[float] = None
        if total_spikes > 0:
            pop_bps = float(total_log_diff / (total_spikes * math.log(2.0)))

        # Population macro-average across active neurons
        active_vals = [b for b in per_neuron_bps if b is not None]
        mean_macro = float(np.mean(active_vals)) if len(active_vals) > 0 else None

        return BPSResult(
            population_bps=pop_bps,
            mean_neuron_bps=mean_macro,
            per_neuron_bps=per_neuron_bps,
            total_spikes=total_spikes,
            active_neurons=len(active_vals),
            num_observations=total_obs,
        )
