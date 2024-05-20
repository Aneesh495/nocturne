"""Positive-rate neural decoder mapping latent states to firing rates lambda in R^N."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nocturne.models.vldm.config import VLDMConfig


class PositiveRateDecoder(nn.Module):
    """Feedforward readout mapping latent trajectory z_{1:T} to firing rates lambda_{1:T} in Hz.

    Formula:
        lambda_{t,n} = softplus( W_2 * relu(W_1 * z_t + b_1) + b_2 )
    """

    def __init__(self, config: VLDMConfig):
        super().__init__()
        self.config = config
        latent_dim = config.latent_dim
        hidden_dim = config.decoder_hidden_dim
        obs_dim = config.obs_dim
        self.softplus_beta = config.softplus_beta

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Baseline bias initialization centered around ~15 Hz
        nn.init.constant_(self.net[-1].bias, math.log(15.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent trajectory [B, T, D] into firing rates [B, T, N] in Hz."""
        logits = self.net(z)
        # Bounded softplus: max(1e-4, softplus(logits))
        rates = F.softplus(logits, beta=self.softplus_beta)
        return torch.clamp(rates, min=1e-4, max=2000.0)
