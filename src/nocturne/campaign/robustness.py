"""Campaign runner for forecasting horizons and neuron dropout robustness.

Evaluates model degradation characteristics across autoregressive future rollout
horizons (H in {1, 2, 5, 10, 20} bins) and missing-neuron channel dropouts
(p in {0.0, 0.1, 0.25, 0.5, 0.75}) under rigorous point-process deviance contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from nocturne.evaluation.bits_per_spike import BitsPerSpikeEvaluator
from nocturne.evaluation.metrics import EvaluationMetricsCalculator
from nocturne.inference.contracts import InferenceMode, InferenceRequest
from nocturne.inference.forecasting import AutoregressiveForecaster


@dataclass(frozen=True)
class HorizonPoint:
    """Evaluation metrics for a specific forecasting rollout horizon."""

    horizon_bins: int
    horizon_ms: float
    poisson_nll: float
    poisson_deviance: float
    bits_per_spike: Optional[float]
    deviance_pseudo_r2: Optional[float]
    latency_ms: float


@dataclass(frozen=True)
class DropoutPoint:
    """Evaluation metrics under a specific neuron dropout rate."""

    dropout_rate: float
    dropped_channels_count: int
    poisson_nll: float
    poisson_deviance: float
    bits_per_spike: Optional[float]
    deviance_pseudo_r2: Optional[float]
    relative_loss_increase: float


@dataclass(frozen=True)
class RobustnessCampaignResult:
    """Complete multi-condition robustness report for a given model and dataset."""

    model_name: str
    dataset_name: str
    seed: int
    horizon_points: List[HorizonPoint]
    dropout_points: List[DropoutPoint]
    is_horizon_monotone: bool
    is_dropout_graceful: bool
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to nested dictionary."""
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "horizon_points": [asdict(p) for p in self.horizon_points],
            "dropout_points": [asdict(p) for p in self.dropout_points],
            "is_horizon_monotone": self.is_horizon_monotone,
            "is_dropout_graceful": self.is_dropout_graceful,
            "summary": self.summary,
        }


class RobustnessCampaignRunner:
    """Executes multi-horizon forecasting and missing-channel dropout campaigns.

    Supports evaluating autoregressive predictions over extended rollouts
    and graceful degradation under randomly masked channels, satisfying Gate BA08.
    """

    DEFAULT_HORIZONS: List[int] = [1, 2, 5, 10, 20]
    DEFAULT_DROPOUT_RATES: List[float] = [0.0, 0.1, 0.25, 0.5, 0.75]

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        dropout_rates: Optional[List[float]] = None,
        bin_width_s: float = 0.02,
    ):
        self.horizons = sorted(horizons or list(self.DEFAULT_HORIZONS))
        self.dropout_rates = sorted(dropout_rates or list(self.DEFAULT_DROPOUT_RATES))
        self.bin_width_s = bin_width_s
        self.metrics_calc = EvaluationMetricsCalculator(bin_width_s=bin_width_s)
        self.bps_eval = BitsPerSpikeEvaluator(bin_width_s=bin_width_s)

    def evaluate_horizon_sweep(
        self,
        model: nn.Module,
        counts: torch.Tensor,
        history_bins: Optional[int] = None,
        null_rates_hz: Optional[torch.Tensor] = None,
    ) -> List[HorizonPoint]:
        """Evaluate autoregressive forecasting degradation over increasing horizons.

        Args:
            model: PyTorch model supporting causal forecasting rollout.
            counts: Ground truth spike count tensor [B, T, N].
            history_bins: Number of prompt history bins (defaults to T - max(horizons)).
            null_rates_hz: Baseline firing rate tensor [B, T, N] or [1, 1, N] for BPS.

        Returns:
            List of HorizonPoint records sorted by horizon length.
        """
        B, T, N = counts.shape
        max_h = max(self.horizons)
        if history_bins is None:
            if T <= max_h:
                raise ValueError(
                    f"Total sequence length T={T} is too short for max horizon {max_h}."
                )
            history_bins = T - max_h

        if history_bins + max_h > T:
            raise ValueError(
                f"history_bins ({history_bins}) + max_horizon ({max_h}) exceeds T ({T})."
            )

        if null_rates_hz is None:
            # Baseline mean rate from history prefix
            hist_counts = counts[:, :history_bins, :]
            mean_rate = hist_counts.sum(dim=(0, 1), keepdim=True) / (
                B * history_bins * self.bin_width_s
            )
            mean_rate = torch.clamp(mean_rate, min=1e-4)
            null_rates = mean_rate.expand(B, max_h, N)
        else:
            null_rates = null_rates_hz

        forecaster = AutoregressiveForecaster(model=model)
        points: List[HorizonPoint] = []

        history_counts = counts[:, :history_bins, :]

        for H in self.horizons:
            req = InferenceRequest(
                mode=InferenceMode.FORECASTING,
                counts=history_counts,
                forecast_horizon_bins=H,
                bin_width_s=self.bin_width_s,
            )

            t0 = time.perf_counter()
            result = forecaster.infer(req)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            pred_rates = result.predicted_rates_hz  # [B, H, N]
            target_future = counts[:, history_bins : history_bins + H, :]  # [B, H, N]
            null_slice = null_rates[:, :H, :] if null_rates.shape[1] >= H else null_rates

            # Compute likelihood and deviance metrics
            summary = self.metrics_calc.compute_metrics(
                predicted_rates_hz=pred_rates,
                target_counts=target_future,
                null_rates_hz=null_slice,
            )

            # Compute Bits Per Spike
            bps_res = self.bps_eval.evaluate(
                predicted_rates_hz=pred_rates,
                target_counts=target_future,
                null_rates_hz=null_slice,
            )

            points.append(
                HorizonPoint(
                    horizon_bins=H,
                    horizon_ms=H * self.bin_width_s * 1000.0,
                    poisson_nll=summary.poisson_nll,
                    poisson_deviance=summary.poisson_deviance,
                    bits_per_spike=bps_res.population_bps,
                    deviance_pseudo_r2=summary.deviance_pseudo_r2,
                    latency_ms=elapsed_ms,
                )
            )

        return points

    def evaluate_dropout_sweep(
        self,
        model: nn.Module,
        counts: torch.Tensor,
        null_rates_hz: Optional[torch.Tensor] = None,
        num_repeats: int = 3,
        seed: int = 42,
    ) -> List[DropoutPoint]:
        """Evaluate performance degradation across missing neuron channel fractions.

        Args:
            model: PyTorch model supporting masked inputs.
            counts: Ground truth spike count tensor [B, T, N].
            null_rates_hz: Baseline firing rate tensor for pseudo-R2 / BPS.
            num_repeats: Number of randomized channel drop trials per fraction.
            seed: RNG seed for reproducible channel selection.

        Returns:
            List of DropoutPoint records sorted by dropout rate.
        """
        model.eval()
        B, T, N = counts.shape
        device = counts.device

        if null_rates_hz is None:
            mean_rate = counts.sum(dim=(0, 1), keepdim=True) / (B * T * self.bin_width_s)
            mean_rate = torch.clamp(mean_rate, min=1e-4)
            null_rates = mean_rate.expand(B, T, N)
        else:
            null_rates = null_rates_hz

        rng = np.random.RandomState(seed)
        points: List[DropoutPoint] = []
        baseline_nll: Optional[float] = None

        for rate in self.dropout_rates:
            k_dropped = int(round(N * rate))
            trial_nlls: List[float] = []
            trial_deviances: List[float] = []
            trial_bps: List[Optional[float]] = []
            trial_r2: List[Optional[float]] = []

            for _ in range(num_repeats):
                obs_mask = torch.ones((B, T, N), dtype=torch.bool, device=device)
                if k_dropped > 0:
                    for b in range(B):
                        perm = rng.permutation(N)
                        dropped_idx = perm[:k_dropped]
                        obs_mask[b, :, dropped_idx] = False

                with torch.no_grad():
                    rates_hz, _, _, _ = model(
                        counts=counts,
                        observed_mask=obs_mask,
                    )

                # Evaluate specifically on the held-out (dropped) channels if rate > 0
                eval_mask = (~obs_mask) if rate > 0.0 else obs_mask

                summary = self.metrics_calc.compute_metrics(
                    predicted_rates_hz=rates_hz,
                    target_counts=counts,
                    null_rates_hz=null_rates,
                    mask=eval_mask,
                )
                bps_res = self.bps_eval.evaluate(
                    predicted_rates_hz=rates_hz,
                    target_counts=counts,
                    null_rates_hz=null_rates,
                    mask=eval_mask,
                )

                trial_nlls.append(summary.poisson_nll)
                trial_deviances.append(summary.poisson_deviance)
                trial_bps.append(bps_res.population_bps)
                trial_r2.append(summary.deviance_pseudo_r2)

            mean_nll = float(np.mean(trial_nlls))
            mean_dev = float(np.mean(trial_deviances))
            valid_bps = [b for b in trial_bps if b is not None]
            avg_bps = float(np.mean(valid_bps)) if valid_bps else None
            valid_r2 = [r for r in trial_r2 if r is not None]
            avg_r2 = float(np.mean(valid_r2)) if valid_r2 else None

            if baseline_nll is None:
                baseline_nll = mean_nll
                rel_increase = 0.0
            else:
                rel_increase = (mean_nll - baseline_nll) / (abs(baseline_nll) + 1e-8)

            points.append(
                DropoutPoint(
                    dropout_rate=rate,
                    dropped_channels_count=k_dropped,
                    poisson_nll=mean_nll,
                    poisson_deviance=mean_dev,
                    bits_per_spike=avg_bps,
                    deviance_pseudo_r2=avg_r2,
                    relative_loss_increase=rel_increase,
                )
            )

        return points

    def run_campaign(
        self,
        model: nn.Module,
        model_name: str,
        counts: torch.Tensor,
        dataset_name: str = "synthetic",
        history_bins: Optional[int] = None,
        null_rates_hz: Optional[torch.Tensor] = None,
        num_repeats: int = 3,
        seed: int = 42,
    ) -> RobustnessCampaignResult:
        """Run complete robustness suite across horizons and neuron dropout rates.

        Args:
            model: PyTorch model.
            model_name: Name identifier for the model architecture.
            counts: Ground truth spike count tensor [B, T, N].
            dataset_name: Dataset identifier.
            history_bins: History prefix length for forecasting.
            null_rates_hz: Baseline firing rates for pseudo-R2 / BPS calculation.
            num_repeats: Channel dropout sampling trials.
            seed: RNG seed for reproducible dropout draws.

        Returns:
            RobustnessCampaignResult with full diagnostic verification.
        """
        horizon_points = self.evaluate_horizon_sweep(
            model=model,
            counts=counts,
            history_bins=history_bins,
            null_rates_hz=null_rates_hz,
        )

        dropout_points = self.evaluate_dropout_sweep(
            model=model,
            counts=counts,
            null_rates_hz=null_rates_hz,
            num_repeats=num_repeats,
            seed=seed,
        )

        # Check horizon monotonicity: Poisson NLL generally does not decrease dramatically
        # allow small numerical noise epsilon
        eps_tol = 0.05
        is_h_monotone = True
        for i in range(1, len(horizon_points)):
            if horizon_points[i].poisson_nll < horizon_points[i - 1].poisson_nll - eps_tol:
                is_h_monotone = False
                break

        # Check dropout gracefulness: loss at 50% dropout does not blow up by > 10x baseline
        baseline_pt = dropout_points[0]
        fifty_pct_pts = [p for p in dropout_points if p.dropout_rate == 0.5]
        is_graceful = True
        if fifty_pct_pts:
            pt_50 = fifty_pct_pts[0]
            if pt_50.relative_loss_increase > 10.0 or np.isnan(pt_50.poisson_nll):
                is_graceful = False

        summary = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "seed": seed,
            "min_horizon_nll": horizon_points[0].poisson_nll,
            "max_horizon_nll": horizon_points[-1].poisson_nll,
            "zero_dropout_nll": dropout_points[0].poisson_nll,
            "max_dropout_nll": dropout_points[-1].poisson_nll,
            "is_horizon_monotone": is_h_monotone,
            "is_dropout_graceful": is_graceful,
        }

        return RobustnessCampaignResult(
            model_name=model_name,
            dataset_name=dataset_name,
            seed=seed,
            horizon_points=horizon_points,
            dropout_points=dropout_points,
            is_horizon_monotone=is_h_monotone,
            is_dropout_graceful=is_graceful,
            summary=summary,
        )
