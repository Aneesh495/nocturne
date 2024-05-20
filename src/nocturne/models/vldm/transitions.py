"""Nonlinear latent transition dynamics model p(z_{t+1} | z_t, u_t)."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn

from nocturne.models.vldm.config import VLDMConfig


class LatentTransitionModel(nn.Module):
    """Parameterizes continuous-time residual dynamics: dz/dt = f_theta(z) + B * u.

    Discrete integration:
        z_{t+1} = z_t + Delta_t * (f_theta(z_t) + B * u_t) + w_t
        where w_t ~ N(0, sigma_w^2 I) is transition innovation noise.
    """

    def __init__(self, config: VLDMConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.bin_width_s = config.bin_width_s
        self.noise_std = config.transition_noise_std

        hidden_dim = config.decoder_hidden_dim

        # Lipschitz-bounded nonlinear neural vector field f_theta(z)
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.latent_dim),
        )

    def vector_field(
        self,
        z: torch.Tensor,
        drive_proj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate continuous velocity dz/dt = f_theta(z) + B * u.

        Args:
            z: Latent states [..., latent_dim].
            drive_proj: Optional projected drive B * u [..., latent_dim].

        Returns:
            Velocity tensor dz/dt [..., latent_dim].
        """
        vel = self.net(z)
        if drive_proj is not None:
            vel = vel + drive_proj
        return vel

    def step(
        self,
        z_t: torch.Tensor,
        drive_proj_t: Optional[torch.Tensor] = None,
        sample_noise: bool = True,
    ) -> torch.Tensor:
        """Advance state one discrete time step: z_{t+1} = z_t + dt * f(z_t) + w_t."""
        dt = self.bin_width_s
        vel = self.vector_field(z_t, drive_proj_t)
        z_next = z_t + dt * vel

        if sample_noise and self.training:
            noise = torch.randn_like(z_next) * self.noise_std
            z_next = z_next + noise

        return z_next
