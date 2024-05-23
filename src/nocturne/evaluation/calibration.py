"""Dispersion index, Fano factor, and Poisson interval coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy import stats
import torch


@dataclass(frozen=True)
class CalibrationReport:
    """Summary of Poisson statistical calibration and dispersion diagnostics."""

    mean_fano_factor: float
    is_overdispersed: bool
    is_underdispersed: bool
    nominal_95_coverage: float       # Empirical fraction of counts inside 95% Poisson intervals
    mean_rate_bias: float            # E[pred_rate] - E[target_rate]
    per_neuron_fano: List[float]


class DispersionAndCoverageAuditor:
    """Evaluates whether point-process assumptions match empirical neural variance."""

    def __init__(self, bin_width_s: float = 0.02):
        self.bin_width_s = bin_width_s

    def audit_calibration(
        self,
        predicted_rates_hz: np.ndarray | torch.Tensor,
        target_counts: np.ndarray | torch.Tensor,
    ) -> CalibrationReport:
        """Measure dispersion and empirical coverage against Poisson model predictions.

        Args:
            predicted_rates_hz: Predicted rates [..., N] in Hz.
            target_counts: Observed counts [..., N].

        Returns:
            CalibrationReport with Fano factors and coverage proportions.
        """
        if isinstance(predicted_rates_hz, torch.Tensor):
            pred_r = predicted_rates_hz.detach().cpu().numpy().astype(np.float64)
        else:
            pred_r = np.asarray(predicted_rates_hz, dtype=np.float64)

        if isinstance(target_counts, torch.Tensor):
            targs = target_counts.detach().cpu().numpy().astype(np.float64)
        else:
            targs = np.asarray(target_counts, dtype=np.float64)

        num_neurons = targs.shape[-1]
        per_neuron_fano: List[float] = []

        for n in range(num_neurons):
            y_n = targs[..., n].ravel()
            mean_y = np.mean(y_n)
            var_y = np.var(y_n)
            if mean_y > 1e-6:
                per_neuron_fano.append(float(var_y / mean_y))
            else:
                per_neuron_fano.append(1.0)

        mean_fano = float(np.mean(per_neuron_fano))
        is_over = mean_fano > 1.25
        is_under = mean_fano < 0.75

        # 95% Nominal Poisson Interval Coverage
        mu = np.clip(pred_r * self.bin_width_s, a_min=1e-8, a_max=None)
        # Using Poisson quantile function (ppf) for lower (2.5%) and upper (97.5%) bounds
        lower_bounds = stats.poisson.ppf(0.025, mu)
        upper_bounds = stats.poisson.ppf(0.975, mu)

        inside_interval = (targs >= lower_bounds) & (targs <= upper_bounds)
        coverage_95 = float(np.mean(inside_interval))

        # Mean rate bias
        target_rates = targs / self.bin_width_s
        rate_bias = float(np.mean(pred_r - target_rates))

        return CalibrationReport(
            mean_fano_factor=mean_fano,
            is_overdispersed=is_over,
            is_underdispersed=is_under,
            nominal_95_coverage=coverage_95,
            mean_rate_bias=rate_bias,
            per_neuron_fano=per_neuron_fano,
        )
