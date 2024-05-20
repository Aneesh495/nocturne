"""Variational lower bound (ELBO) objective and loss components for VLDM."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from nocturne.models.likelihoods import PoissonLoss
from nocturne.models.vldm.config import VLDMConfig


class VLDMELBOCalculator(nn.Module):
    """Computes the Evidence Lower Bound (ELBO) and multi-component training loss."""

    def __init__(self, config: VLDMConfig):
        super().__init__()
        self.config = config
        self.poisson_loss = PoissonLoss(bin_width_s=config.bin_width_s, include_log_factorial=False)
        self.kl_weight = config.kl_weight

    def compute_kl_z0(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Analytic KL divergence KL( q(z_0 | y) || N(0, I) ).

        Formula: 0.5 * sum_d [ exp(log_var_d) + mu_d^2 - 1 - log_var_d ]
        """
        # [B, D] -> scalar mean over batch
        var = torch.exp(log_var)
        kl_element = 0.5 * (var + mu.pow(2) - 1.0 - log_var)
        return kl_element.sum(dim=-1).mean()

    def forward(
        self,
        predicted_rates_hz: torch.Tensor,
        targets: torch.Tensor,
        mu_0: torch.Tensor,
        log_var_0: torch.Tensor,
        drive_penalty: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute full VLDM loss: NLL_recon + beta * KL(z_0) + drive_penalty.

        Args:
            predicted_rates_hz: Decoded firing rates [B, T, N].
            targets: Target spike count observations [B, T, N].
            mu_0: Variational mean of initial state [B, D].
            log_var_0: Variational log-variance of initial state [B, D].
            drive_penalty: Scalar penalty from inferred drive network.
            target_mask: Optional boolean evaluation mask [B, T, N].

        Returns:
            Tuple of (total_loss_tensor, loss_breakdown_dict).
        """
        recon_loss = self.poisson_loss.forward_from_rates(
            rates_hz=predicted_rates_hz,
            targets=targets,
            mask=target_mask,
        )

        kl_z0 = self.compute_kl_z0(mu_0, log_var_0)

        total_loss = recon_loss + (self.kl_weight * kl_z0) + drive_penalty

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_recon": float(recon_loss.item()),
            "loss_kl_z0": float(kl_z0.item()),
            "loss_drive_penalty": float(drive_penalty.item()),
        }

        return total_loss, metrics
