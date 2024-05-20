"""Baseline model B08: Poisson Linear Dynamical System (PLDS) state-space definition."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoissonLinearDynamicalSystem(nn.Module):
    """Poisson Linear Dynamical System (PLDS).

    Latent transition:
        z_1 ~ N(mu_0, Q_0)
        z_{t+1} = A * z_t + w_t,  w_t ~ N(0, Q)

    Poisson emission:
        ln(lambda_{t,n}) = C_n * z_t + d_n
        y_{t,n} ~ Poisson(lambda_{t,n} * Delta_t)
    """

    def __init__(
        self,
        latent_dim: int,
        num_neurons: int,
        spectral_radius: float = 0.95,
        seed: int = 42,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_neurons = num_neurons

        rng = np.random.default_rng(seed)

        # Transition matrix A [D, D]
        A_raw = rng.normal(0.0, 1.0 / math.sqrt(latent_dim), size=(latent_dim, latent_dim))
        sr = np.max(np.abs(np.linalg.eigvals(A_raw)))
        A_init = (spectral_radius / (sr + 1e-8)) * A_raw
        self.A = nn.Parameter(torch.from_numpy(A_init).float())

        # Process noise log-variance (diagonal Q) [D]
        self.log_q_diag = nn.Parameter(torch.full((latent_dim,), -1.0))

        # Initial state prior mu_0 [D], log_q0 [D]
        self.mu_0 = nn.Parameter(torch.zeros(latent_dim))
        self.log_q0_diag = nn.Parameter(torch.zeros(latent_dim))

        # Emission matrix C [N, D] and bias d [N]
        C_init = rng.normal(0.0, 1.0 / math.sqrt(latent_dim), size=(num_neurons, latent_dim))
        self.C = nn.Parameter(torch.from_numpy(C_init).float())
        self.d = nn.Parameter(torch.full((num_neurons,), math.log(15.0)))

    @property
    def Q(self) -> torch.Tensor:
        """Process noise covariance matrix [D, D]."""
        return torch.diag(torch.exp(self.log_q_diag))

    @property
    def Q_0(self) -> torch.Tensor:
        """Initial state covariance matrix [D, D]."""
        return torch.diag(torch.exp(self.log_q0_diag))

    def decode_rates_hz(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent trajectory [..., D] to firing rates in Hz [..., N].

        Formula: lambda = exp(C * z + d)
        """
        logits = F.linear(z, self.C, self.d)
        clamped_logits = torch.clamp(logits, max=10.0, min=-10.0)
        return torch.exp(clamped_logits)

    def sample_trajectory(
        self,
        num_time_bins: int,
        bin_width_s: float,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample synthetic latents, rates, and Poisson spikes.

        Returns:
            Tuple of (latents [T, D], rates_hz [T, N], counts [T, N]).
        """
        if rng is None:
            rng = np.random.default_rng()

        D = self.latent_dim
        N = self.num_neurons

        A_np = self.A.detach().cpu().numpy()
        q_diag = np.exp(self.log_q_diag.detach().cpu().numpy())
        q0_diag = np.exp(self.log_q0_diag.detach().cpu().numpy())
        mu0 = self.mu_0.detach().cpu().numpy()
        C_np = self.C.detach().cpu().numpy()
        d_np = self.d.detach().cpu().numpy()

        latents = np.zeros((num_time_bins, D), dtype=np.float64)
        latents[0] = rng.normal(mu0, np.sqrt(q0_diag))

        for t in range(1, num_time_bins):
            w = rng.normal(0.0, np.sqrt(q_diag))
            latents[t] = A_np @ latents[t - 1] + w

        logits = latents @ C_np.T + d_np[None, :]
        rates_hz = np.exp(np.clip(logits, -10.0, 8.0))
        mu_counts = rates_hz * bin_width_s
        counts = rng.poisson(mu_counts).astype(np.int64)

        return latents, rates_hz, counts
