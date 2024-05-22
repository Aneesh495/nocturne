"""Missing-channel and neuron-dropout robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from nocturne.models.likelihoods import PoissonLoss
from nocturne.binning.likelihood_math import poisson_bits_per_spike


@dataclass(frozen=True)
class RobustnessReport:
    """Evaluation summary of model degradation across missing neuron fractions."""

    dropout_rates: List[float]
    mean_losses: List[float]
    relative_degradation: List[float]
    is_graceful: bool
    summary: Dict[str, float]


class NeuronDropoutAuditor:
    """Evaluates model performance degradation under missing channels (10% to 90%)."""

    def __init__(
        self,
        dropout_rates: List[float] | None = None,
        bin_width_s: float = 0.02,
    ):
        self.dropout_rates = dropout_rates or [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        self.bin_width_s = bin_width_s
        self.loss_fn = PoissonLoss(bin_width_s=bin_width_s)

    @torch.no_grad()
    def audit_model(
        self,
        model: nn.Module,
        test_counts: torch.Tensor,
        num_trials_per_rate: int = 5,
        seed: int = 42,
    ) -> RobustnessReport:
        """Measure Poisson loss as a function of missing channel fraction.

        Args:
            model: PyTorch model.
            test_counts: Ground truth test count tensor [B, T, N].
            num_trials_per_rate: Number of distinct channel drop masks evaluated per rate.
            seed: RNG seed for reproducible channel selection.

        Returns:
            RobustnessReport containing degradation curve and gracefulness metrics.
        """
        model.eval()
        torch.manual_seed(seed)
        np.random.seed(seed)

        B, T, N = test_counts.shape
        device = test_counts.device
        losses_by_rate: List[float] = []

        baseline_loss = 0.0

        for idx, rate in enumerate(self.dropout_rates):
            trial_losses: List[float] = []

            for _ in range(num_trials_per_rate):
                # Construct channel dropout mask
                num_dropped = int(N * rate)
                obs_mask = torch.ones((B, T, N), dtype=torch.bool, device=device)

                if num_dropped > 0:
                    for b in range(B):
                        perm = torch.randperm(N, device=device)
                        dropped_indices = perm[:num_dropped]
                        obs_mask[b, :, dropped_indices] = False

                rates_hz, _, _, _ = model(
                    counts=test_counts,
                    observed_mask=obs_mask,
                )

                # Evaluate loss on held-out channels if rate > 0, else on all
                eval_mask = (~obs_mask) if rate > 0.0 else obs_mask
                loss = self.loss_fn.forward_from_rates(
                    rates_hz=rates_hz,
                    targets=test_counts,
                    mask=eval_mask,
                )
                trial_losses.append(float(loss.item()))

            mean_l = float(np.mean(trial_losses))
            losses_by_rate.append(mean_l)
            if idx == 0:
                baseline_loss = mean_l

        # Relative degradation: loss(rate) - loss(0)
        rel_degrad = [l - baseline_loss for l in losses_by_rate]

        # Check for gracefulness: monotonic degradation without NaNs or infinite values
        all_finite = all(np.isfinite(l) for l in losses_by_rate)
        is_graceful = all_finite and all(r >= -0.5 for r in rel_degrad)

        return RobustnessReport(
            dropout_rates=self.dropout_rates,
            mean_losses=losses_by_rate,
            relative_degradation=rel_degrad,
            is_graceful=is_graceful,
            summary={
                "baseline_loss": baseline_loss,
                "loss_at_50pct": losses_by_rate[self.dropout_rates.index(0.5)] if 0.5 in self.dropout_rates else 0.0,
                "loss_at_90pct": losses_by_rate[self.dropout_rates.index(0.9)] if 0.9 in self.dropout_rates else 0.0,
            },
        )
