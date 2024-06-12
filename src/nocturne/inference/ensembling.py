"""Deep ensemble rate fusion and epistemic versus aleatoric uncertainty quantification.

Combines predictions across diverse models and random seeds, decomposing predictive
uncertainty into model disagreement (epistemic) and point-process shot noise (aleatoric).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class EnsemblePredictionResult:
    """Quantitative outcome of ensemble fusion with uncertainty decomposition."""

    mean_rates_hz: torch.Tensor          # [B, T, N]
    epistemic_variance: torch.Tensor     # [B, T, N] variance across model predictions
    aleatoric_variance: torch.Tensor     # [B, T, N] expected Poisson shot noise variance
    total_count_variance: torch.Tensor   # [B, T, N] total variance of predicted counts
    weights: List[float]


class DeepRateEnsemble:
    """Ensemble of point-process dynamics models with uncertainty quantification."""

    def __init__(self, bin_width_s: float = 0.02):
        if bin_width_s <= 0.0:
            raise ValueError(f"bin_width_s must be > 0, got {bin_width_s}")
        self.bin_width_s = bin_width_s
        self.models: List[nn.Module] = []
        self.weights: List[float] = []

    def add_model(self, model: nn.Module, weight: float = 1.0) -> None:
        """Add a trained model to the ensemble committee with non-negative weight.

        Args:
            model: PyTorch model emitting firing rates or forward tuple.
            weight: Non-negative scalar weight for Bayesian model averaging.
        """
        if weight < 0.0:
            raise ValueError(f"Model weight must be >= 0, got {weight}")
        self.models.append(model)
        self.weights.append(float(weight))

    def _get_normalized_weights(self) -> List[float]:
        total = sum(self.weights)
        if total <= 0.0:
            n = len(self.weights)
            return [1.0 / n] * n
        return [w / total for w in self.weights]

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        counts: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> EnsemblePredictionResult:
        """Execute ensemble forward passes and decompose predictive variance.

        Args:
            counts: Input spike count tensor [B, T, N].
            observed_mask: Optional boolean observation mask [B, T, N].

        Returns:
            EnsemblePredictionResult with epistemic, aleatoric, and total variance.
        """
        if not self.models:
            raise RuntimeError("Ensemble has no registered models.")

        device = counts.device
        norm_weights = self._get_normalized_weights()

        all_rates: List[torch.Tensor] = []

        for model in self.models:
            model.eval()
            # Standard model call: either model(counts, mask) or tuple output
            out = model(counts=counts, observed_mask=observed_mask)
            if isinstance(out, tuple):
                rates = out[0]
            else:
                rates = out
            all_rates.append(rates)

        # Stack predictions: [M, B, T, N]
        stacked_rates = torch.stack(all_rates, dim=0)
        M, B, T, N = stacked_rates.shape

        # Weighted ensemble mean rate: [B, T, N]
        w_tensor = torch.tensor(norm_weights, dtype=counts.dtype, device=device).view(M, 1, 1, 1)
        mean_rates = torch.sum(stacked_rates * w_tensor, dim=0)

        # Epistemic variance (model disagreement): sum w_m * (lambda_m - mean_lambda)^2
        diff = stacked_rates - mean_rates.unsqueeze(0)
        epistemic_var = torch.sum(w_tensor * (diff**2), dim=0)  # [B, T, N]

        # Aleatoric variance in rate units: Poisson shot noise rate variance is lambda / dt
        # For binned counts y ~ Poisson(lambda * dt), Var(y) = lambda * dt
        # In rate units (Hz^2): Var(lambda) = lambda / dt
        aleatoric_var = mean_rates / self.bin_width_s  # [B, T, N]

        # Total predicted count variance: Var(y) = Epistemic_count + Aleatoric_count
        # Epistemic_count = Var(lambda) * dt^2
        # Aleatoric_count = mean_lambda * dt
        total_count_var = (epistemic_var * (self.bin_width_s**2)) + (mean_rates * self.bin_width_s)

        return EnsemblePredictionResult(
            mean_rates_hz=mean_rates,
            epistemic_variance=epistemic_var,
            aleatoric_variance=aleatoric_var,
            total_count_variance=total_count_var,
            weights=norm_weights,
        )
