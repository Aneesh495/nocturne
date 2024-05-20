"""Baseline model B07: Low-rank point-process Poisson factor model."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nocturne.models.likelihoods import PoissonLoss


class LowRankPoissonFactorModel(nn.Module):
    """Low-rank Poisson matrix factorization baseline.

    Parameterizes neural firing rates through K latent temporal factors:
        lambda_{t,n} = softplus( sum_{k=1}^K u_{t,k} * v_{n,k} + b_n )

    Trained by minimizing Poisson negative log-likelihood on training partitions.
    """

    def __init__(
        self,
        num_neurons: int,
        rank: int = 5,
        softplus_beta: float = 1.0,
    ):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        self.num_neurons = num_neurons
        self.rank = rank
        self.softplus_beta = softplus_beta

        # Loading vectors V: [num_neurons, rank]
        self.neuron_loadings = nn.Parameter(
            torch.randn(num_neurons, rank) / math.sqrt(rank)
        )
        # Baseline bias b: [num_neurons]
        self.bias = nn.Parameter(torch.zeros(num_neurons))

    def decode_rates(self, temporal_factors: torch.Tensor) -> torch.Tensor:
        """Decode temporal factors [..., rank] into firing rates [..., num_neurons]."""
        # logits = factors @ V^T + b
        logits = F.linear(temporal_factors, self.neuron_loadings, self.bias)
        return F.softplus(logits, beta=self.softplus_beta)

    def fit_factors_and_loadings(
        self,
        counts: torch.Tensor,
        bin_width_s: float,
        num_steps: int = 150,
        lr: float = 0.05,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """Fit temporal factors U and neuron loadings V on a batch of count matrices.

        Args:
            counts: Observed counts [B, T, N].
            bin_width_s: Bin width in seconds.
            num_steps: Optimization gradient iterations.
            lr: Learning rate.
            mask: Optional observation mask [B, T, N].

        Returns:
            Tuple of (fitted_factors [B, T, rank], final_loss).
        """
        B, T, N = counts.shape
        device = counts.device

        # Initialize temporal factors U: [B, T, rank]
        factors = nn.Parameter(torch.randn(B, T, self.rank, device=device) * 0.1)
        optimizer = torch.optim.Adam([self.neuron_loadings, self.bias, factors], lr=lr)
        loss_fn = PoissonLoss(bin_width_s=bin_width_s)

        final_loss = 0.0
        for _ in range(num_steps):
            optimizer.zero_grad()
            rates = self.decode_rates(factors)
            loss = loss_fn.forward_from_rates(rates, counts, mask=mask)
            # L2 regularization on loadings and factors
            reg = 1e-4 * (factors.pow(2).mean() + self.neuron_loadings.pow(2).mean())
            total_loss = loss + reg
            total_loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        return factors.detach(), final_loss
