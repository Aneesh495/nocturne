"""Laplace approximation and MAP smoother for Poisson Linear Dynamical Systems."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from nocturne.models.baselines.plds import PoissonLinearDynamicalSystem


class PLDSLaplaceSmoother:
    """Laplace approximation for latent state inference in PLDS.

    Computes the maximum a posteriori (MAP) latent trajectory z_{1:T}
    by maximizing the joint log-posterior log p(z_{1:T}, y_{1:T})
    via second-order Newton-Raphson or L-BFGS optimization.
    """

    def __init__(self, plds_model: PoissonLinearDynamicalSystem):
        self.model = plds_model

    def infer_latents_map(
        self,
        counts: torch.Tensor,
        bin_width_s: float,
        num_iter: int = 50,
        lr: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Infer MAP latent trajectory z_hat and posterior precision/covariance.

        Args:
            counts: Observed spike counts [T, num_neurons].
            bin_width_s: Temporal bin width in seconds.
            num_iter: Optimization iterations.
            lr: Learning rate.

        Returns:
            Tuple of (map_latents [T, D], posterior_variance [T, D]).
        """
        T, N = counts.shape
        D = self.model.latent_dim
        device = counts.device

        A = self.model.A.detach().to(device)
        Q_diag = torch.exp(self.model.log_q_diag).detach().to(device)
        Q0_diag = torch.exp(self.model.log_q0_diag).detach().to(device)
        mu_0 = self.model.mu_0.detach().to(device)
        C = self.model.C.detach().to(device)
        d = self.model.d.detach().to(device)
        log_dt = math.log(bin_width_s)

        # Initialize z: [T, D]
        z = nn.Parameter(torch.zeros(T, D, device=device))
        optimizer = torch.optim.Adam([z], lr=lr)

        for _ in range(num_iter):
            optimizer.zero_grad()
            # 1. Prior negative log-density
            # t=0: (z_0 - mu_0)^2 / (2 * Q_0)
            diff_0 = z[0] - mu_0
            nll_prior = 0.5 * (diff_0.pow(2) / Q0_diag).sum()

            # t > 0: (z_t - A * z_{t-1})^2 / (2 * Q)
            if T > 1:
                pred_z = z[:-1] @ A.T  # [T-1, D]
                diff_trans = z[1:] - pred_z
                nll_prior = nll_prior + 0.5 * (diff_trans.pow(2) / Q_diag).sum()

            # 2. Observation negative log-likelihood
            # logits: C * z + d
            logits = z @ C.T + d[None, :]  # [T, N]
            log_mu = logits + log_dt
            mu = torch.exp(torch.clamp(log_mu, max=10.0))
            # NLL = mu - y * log_mu
            nll_obs = (mu - counts * log_mu).sum()

            total_loss = nll_prior + nll_obs
            total_loss.backward()
            optimizer.step()

        map_z = z.detach()

        # Compute diagonal Laplace posterior variance: inverse of Hessian diagonal
        # Observation Hessian w.r.t z_t: C^T diag(mu_t) C
        logits = map_z @ C.T + d[None, :]
        mu = torch.exp(torch.clamp(logits + log_dt, max=10.0))
        # Diagonal approximation: Var(z_{t,d}) approx 1 / (H_prior + sum_n C_{n,d}^2 * mu_{t,n})
        H_prior = 1.0 / Q_diag[None, :]  # [1, D]
        H_obs = (C.pow(2).T[None, :, :] * mu[:, None, :]).sum(dim=-1)  # [T, D]
        post_var = 1.0 / (H_prior + H_obs + 1e-6)

        return map_z, post_var
