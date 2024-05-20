"""Complete Variational Nonlinear Latent Dynamics Model (VLDM) architecture."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from nocturne.models.vldm.config import VLDMConfig
from nocturne.models.vldm.decoder import PositiveRateDecoder
from nocturne.models.vldm.drive import InferredDriveNetwork
from nocturne.models.vldm.elbo import VLDMELBOCalculator
from nocturne.models.vldm.encoder import ObservationEncoder
from nocturne.models.vldm.transitions import LatentTransitionModel


class VariationalLatentDynamicsModel(nn.Module):
    """Variational Nonlinear Latent Dynamics Model (VLDM).

    Consists of:
        1. Observation encoder q(z_0 | Y_{1:T})
        2. Latent transition model dz/dt = f_theta(z) + B * u
        3. Inferred external drive subnetwork u_t in R^M (or strictly autonomous switch)
        4. Positive-rate decoder lambda_{t,n} in R^+
        5. Variational ELBO objective with Poisson likelihood
    """

    def __init__(self, config: VLDMConfig):
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.transitions = LatentTransitionModel(config)
        self.drive_net = InferredDriveNetwork(config)
        self.decoder = PositiveRateDecoder(config)
        self.elbo = VLDMELBOCalculator(config)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from Gaussian posterior using reparameterization trick: z = mu + sigma * eps."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def rollout_latents(
        self,
        z_0: torch.Tensor,
        num_time_bins: int,
        inferred_drive: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Roll out latent states z_{1:T} through transition dynamics.

        Args:
            z_0: Initial latent states [B, D].
            num_time_bins: Trajectory horizon T.
            inferred_drive: Optional inferred drive [B, T, drive_dim].

        Returns:
            Latent trajectory tensor [B, T, D].
        """
        B, D = z_0.shape
        device = z_0.device

        drive_proj = self.drive_net.project_to_latent_velocity(inferred_drive)  # [B, T, D] or None

        latents = torch.zeros(B, num_time_bins, D, device=device)
        current_z = z_0
        latents[:, 0, :] = current_z

        for t in range(num_time_bins - 1):
            drive_t = drive_proj[:, t, :] if drive_proj is not None else None
            current_z = self.transitions.step(current_z, drive_t, sample_noise=self.training)
            latents[:, t + 1, :] = current_z

        return latents

    def forward(
        self,
        counts: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Complete forward pass: encode, rollout, decode, and compute ELBO loss.

        Args:
            counts: Observed spike counts [B, T, N].
            observed_mask: Mask of visible inputs [B, T, N].
            target_mask: Mask of evaluated targets [B, T, N].

        Returns:
            Tuple of (predicted_rates_hz [B, T, N], latents [B, T, D], loss_metrics_dict).
        """
        B, T, N = counts.shape

        # 1. Encode initial state posterior
        mu_0, log_var_0 = self.encoder(counts, mask=observed_mask)
        z_0 = self.reparameterize(mu_0, log_var_0)

        # 2. Infer external drive if enabled
        drive_u, drive_penalty = self.drive_net(counts, mask=observed_mask)

        # 3. Roll out latent dynamics
        latents = self.rollout_latents(z_0, num_time_bins=T, inferred_drive=drive_u)

        # 4. Decode into firing rates in Hz
        rates_hz = self.decoder(latents)

        # 5. Compute ELBO loss
        total_loss, metrics = self.elbo(
            predicted_rates_hz=rates_hz,
            targets=counts,
            mu_0=mu_0,
            log_var_0=log_var_0,
            drive_penalty=drive_penalty,
            target_mask=target_mask,
        )

        return rates_hz, latents, total_loss, metrics
