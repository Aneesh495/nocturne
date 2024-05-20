"""PyTorch point-process Poisson loss modules, mask weighting, and autograd gradients."""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from nocturne.binning.likelihood_math import get_log_factorial


class PoissonLoss(nn.Module):
    """Numerically stable Poisson negative log-likelihood loss for point-process spike counts.

    Supports both rate inputs (Hz) and unconstrained log-rate logits:
        mu = lambda * Delta_t = exp(log_rate) * Delta_t

    Loss formulation per valid observation:
        NLL(y, mu) = mu - y * ln(mu) + ln(y!)
        where y * ln(mu) evaluates to 0 when y == 0.
    """

    def __init__(
        self,
        bin_width_s: float,
        include_log_factorial: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        if bin_width_s <= 0.0:
            raise ValueError(f"bin_width_s must be strictly positive, got {bin_width_s}")
        self.bin_width_s = bin_width_s
        self.include_log_factorial = include_log_factorial
        self.eps = eps
        self.log_bin_width = math.log(bin_width_s)

    def forward_from_rates(
        self,
        rates_hz: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mean Poisson NLL given predicted firing rates in Hz.

        Args:
            rates_hz: Predicted rates in Hz (non-negative) [..., N].
            targets: Non-negative integer spike counts [..., N].
            mask: Optional boolean mask [..., N] where True indicates valid evaluated elements.
        """
        # mu = rates * Delta_t
        clamped_rates = torch.clamp(rates_hz, min=self.eps)
        mu = clamped_rates * self.bin_width_s

        # Pointwise Poisson NLL: mu - y * ln(mu)
        log_mu = torch.log(mu)
        elem_nll = mu - targets * log_mu

        if self.include_log_factorial:
            # lgamma(y + 1) = ln(y!)
            elem_nll = elem_nll + torch.lgamma(targets + 1.0)

        if mask is not None:
            if mask.shape != targets.shape:
                raise ValueError(f"mask shape {mask.shape} != targets shape {targets.shape}")
            num_valid = mask.sum().float()
            if num_valid.item() == 0:
                return torch.tensor(0.0, device=rates_hz.device, dtype=rates_hz.dtype)
            return (elem_nll * mask.float()).sum() / num_valid

        return elem_nll.mean()

    def forward_from_log_rates(
        self,
        log_rates: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute mean Poisson NLL given unconstrained log-rates eta = ln(lambda).

        Formula:
            ln(mu) = eta + ln(Delta_t)
            mu = exp(eta + ln(Delta_t))
            NLL = mu - y * ln(mu)
        """
        log_mu = log_rates + self.log_bin_width
        mu = torch.exp(torch.clamp(log_mu, max=15.0))  # Prevent exp overflow

        elem_nll = mu - targets * log_mu

        if self.include_log_factorial:
            elem_nll = elem_nll + torch.lgamma(targets + 1.0)

        if mask is not None:
            if mask.shape != targets.shape:
                raise ValueError(f"mask shape {mask.shape} != targets shape {targets.shape}")
            num_valid = mask.sum().float()
            if num_valid.item() == 0:
                return torch.tensor(0.0, device=log_rates.device, dtype=log_rates.dtype)
            return (elem_nll * mask.float()).sum() / num_valid

        return elem_nll.mean()
