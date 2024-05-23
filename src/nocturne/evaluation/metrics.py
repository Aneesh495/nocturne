"""Independent point-process metric calculators, Poisson deviance, and pseudo-R2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch

from nocturne.binning.likelihood_math import get_log_factorial


@dataclass(frozen=True)
class LikelihoodSummary:
    """Summary container of point-process deviance and log-likelihood metrics."""

    total_spikes: int
    num_observations: int
    poisson_nll: float
    poisson_deviance: float
    deviance_pseudo_r2: Optional[float]
    mae_rates: float
    rmse_rates: float


class EvaluationMetricsCalculator:
    """Computes point-process evaluation metrics independently from training loss code."""

    def __init__(self, bin_width_s: float = 0.02, eps: float = 1e-9):
        if bin_width_s <= 0.0:
            raise ValueError(f"bin_width_s must be > 0, got {bin_width_s}")
        self.bin_width_s = bin_width_s
        self.eps = eps

    def compute_metrics(
        self,
        predicted_rates_hz: np.ndarray | torch.Tensor,
        target_counts: np.ndarray | torch.Tensor,
        null_rates_hz: Optional[np.ndarray | torch.Tensor] = None,
        mask: Optional[np.ndarray | torch.Tensor] = None,
    ) -> LikelihoodSummary:
        """Compute full suite of deviance, NLL, and pseudo-R2 metrics.

        Args:
            predicted_rates_hz: Array or Tensor [..., N] of predicted firing rates in Hz.
            target_counts: Array or Tensor [..., N] of integer spike counts.
            null_rates_hz: Optional baseline rate in Hz for computing pseudo-R2.
            mask: Optional boolean mask where True indicates valid evaluated entries.

        Returns:
            LikelihoodSummary with exact deviance and NLL values.
        """
        # Convert to numpy for precision and independent recomputation
        if isinstance(predicted_rates_hz, torch.Tensor):
            pred_rates = predicted_rates_hz.detach().cpu().numpy().astype(np.float64)
        else:
            pred_rates = np.asarray(predicted_rates_hz, dtype=np.float64)

        if isinstance(target_counts, torch.Tensor):
            targets = target_counts.detach().cpu().numpy().astype(np.float64)
        else:
            targets = np.asarray(target_counts, dtype=np.float64)

        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy().astype(bool)
            else:
                mask_np = np.asarray(mask, dtype=bool)
            pred_rates = pred_rates[mask_np]
            targets = targets[mask_np]
            if null_rates_hz is not None:
                if isinstance(null_rates_hz, torch.Tensor):
                    null_np = null_rates_hz.detach().cpu().numpy().astype(np.float64)
                else:
                    null_np = np.asarray(null_rates_hz, dtype=np.float64)
                null_rates = null_np[mask_np]
            else:
                null_rates = None
        else:
            pred_rates = pred_rates.ravel()
            targets = targets.ravel()
            if null_rates_hz is not None:
                if isinstance(null_rates_hz, torch.Tensor):
                    null_rates = null_rates_hz.detach().cpu().numpy().astype(np.float64).ravel()
                else:
                    null_rates = np.asarray(null_rates_hz, dtype=np.float64).ravel()
            else:
                null_rates = None

        N_obs = len(targets)
        total_spikes = int(np.sum(targets))

        # Expected counts mu = lambda * Delta_t
        mu = np.clip(pred_rates * self.bin_width_s, a_min=self.eps, a_max=None)

        # Poisson deviance: 2 * sum(y * ln(y / mu) - (y - mu))
        dev_terms = np.zeros_like(targets)
        nonzero = targets > 0
        dev_terms[nonzero] = targets[nonzero] * np.log(targets[nonzero] / mu[nonzero])
        deviance = 2.0 * np.sum(dev_terms - (targets - mu))

        # Poisson NLL per bin
        log_fact = np.array([get_log_factorial(int(k)) for k in targets])
        nll_total = np.sum(mu - targets * np.log(mu) + log_fact)
        mean_nll = float(nll_total / max(1, N_obs))

        # Deviance pseudo-R2
        pseudo_r2: Optional[float] = None
        if null_rates is not None:
            mu_null = np.clip(null_rates * self.bin_width_s, a_min=self.eps, a_max=None)
            dev_null_terms = np.zeros_like(targets)
            dev_null_terms[nonzero] = targets[nonzero] * np.log(targets[nonzero] / mu_null[nonzero])
            null_deviance = 2.0 * np.sum(dev_null_terms - (targets - mu_null))
            if null_deviance > 1e-12:
                pseudo_r2 = float(1.0 - (deviance / null_deviance))

        # Rate error metrics
        target_rates_hz = targets / self.bin_width_s
        mae = float(np.mean(np.abs(pred_rates - target_rates_hz)))
        rmse = float(np.sqrt(np.mean((pred_rates - target_rates_hz) ** 2)))

        return LikelihoodSummary(
            total_spikes=total_spikes,
            num_observations=N_obs,
            poisson_nll=mean_nll,
            poisson_deviance=float(deviance),
            deviance_pseudo_r2=pseudo_r2,
            mae_rates=mae,
            rmse_rates=rmse,
        )
