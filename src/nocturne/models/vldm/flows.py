"""Invertible normalizing flow priors for expressive multi-modal latent representations.

Provides planar and radial normalizing flows to transform standard Gaussian priors
into rich multi-modal distributions for variational latent state spaces.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarFlow(nn.Module):
    """Planar invertible transformation with tractable Jacobian determinant.

    Mapping:
        f(z) = z + u * tanh(w^T z + b)
    Jacobian Determinant:
        |det df/dz| = |1 + u^T psi(z)|
        where psi(z) = (1 - tanh^2(w^T z + b)) * w
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn(1, dim) * 0.1)
        self.u = nn.Parameter(torch.randn(1, dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def _get_invertible_u(self) -> torch.Tensor:
        """Enforce invertibility condition w^T u >= -1 via softplus reparameterization."""
        w_norm_sq = torch.sum(self.w**2, dim=-1, keepdim=True) + 1e-8
        w_dot_u = torch.sum(self.w * self.u, dim=-1, keepdim=True)
        m_wu = -1.0 + F.softplus(w_dot_u)
        return self.u + (m_wu - w_dot_u) * (self.w / w_norm_sq)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform latent sample and accumulate log Jacobian determinant.

        Args:
            z: Input latent tensor [..., dim].

        Returns:
            Tuple of (transformed_z, log_det_jacobian).
        """
        u_hat = self._get_invertible_u()
        lin = torch.sum(self.w * z, dim=-1, keepdim=True) + self.b  # [..., 1]
        tanh_lin = torch.tanh(lin)

        # f(z) = z + u_hat * tanh(w^T z + b)
        z_next = z + u_hat * tanh_lin

        # psi(z) = (1 - tanh^2(lin)) * w
        psi = (1.0 - tanh_lin**2) * self.w  # [..., dim]
        psi_dot_u = torch.sum(psi * u_hat, dim=-1, keepdim=True)  # [..., 1]
        det = torch.abs(1.0 + psi_dot_u) + 1e-8
        log_det = torch.log(det).squeeze(-1)  # [...]

        return z_next, log_det


class RadialFlow(nn.Module):
    """Radial invertible contraction/expansion flow around a reference center.

    Mapping:
        f(z) = z + (beta / (alpha + r)) * (z - z0)
        where r = ||z - z0||
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.z0 = nn.Parameter(torch.randn(1, dim) * 0.1)
        self.alpha_raw = nn.Parameter(torch.zeros(1))
        self.beta_raw = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply radial transformation and compute exact log determinant.

        Args:
            z: Latent tensor [..., dim].

        Returns:
            Tuple of (transformed_z, log_det_jacobian).
        """
        alpha = F.softplus(self.alpha_raw) + 1e-4
        # Invertibility: beta >= -alpha
        beta = -alpha + F.softplus(self.beta_raw)

        diff = z - self.z0
        r = torch.norm(diff, p=2, dim=-1, keepdim=True) + 1e-8  # [..., 1]
        h = 1.0 / (alpha + r)

        # f(z) = z + beta * h * (z - z0)
        z_next = z + beta * h * diff

        # Jacobian log determinant in d dimensions
        term1 = (self.dim - 1) * torch.log(torch.abs(1.0 + beta * h) + 1e-8)
        term2 = torch.log(torch.abs(1.0 + beta * h - (beta * r) / ((alpha + r)**2)) + 1e-8)
        log_det = (term1 + term2).squeeze(-1)

        return z_next, log_det


class NormalizingFlowPrior(nn.Module):
    """Multi-layer composable normalizing flow prior for latent state spaces."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        flow_type: str = "planar",
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            if flow_type == "planar":
                self.layers.append(PlanarFlow(dim))
            elif flow_type == "radial":
                self.layers.append(RadialFlow(dim))
            else:
                raise ValueError(f"Unknown flow_type {flow_type}, expected 'planar' or 'radial'")

    def sample_with_log_prob(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from base Gaussian N(0, I) and transform through flow layers.

        Args:
            batch_size: Number of latent vectors to sample.
            device: Target torch device.

        Returns:
            Tuple of (sampled_latents [B, dim], log_prob [B]).
        """
        # Base distribution: z_0 ~ N(0, I)
        z = torch.randn(batch_size, self.dim, device=device)
        # Base log prob: -0.5 * (d * ln(2*pi) + ||z_0||^2)
        log_prob = -0.5 * (
            self.dim * math.log(2.0 * math.pi) + torch.sum(z**2, dim=-1)
        )

        # Forward flow composition
        for layer in self.layers:
            z, log_det = layer(z)
            log_prob = log_prob - log_det

        return z, log_prob

    def forward_transform(self, z0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform given latent sample through the flow and return total log_det.

        Args:
            z0: Initial latent tensor [..., dim].

        Returns:
            Tuple of (final_z [..., dim], total_log_det [...]).
        """
        z = z0
        total_log_det = torch.zeros(z0.shape[:-1], dtype=z0.dtype, device=z0.device)

        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det

        return z, total_log_det
