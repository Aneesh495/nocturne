"""Observation encoder parameterizing the variational posterior over initial latent state q(z_0 | y)."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn

from nocturne.models.vldm.config import VLDMConfig


class ObservationEncoder(nn.Module):
    """Bidirectional or causal recurrent observation encoder.

    Processes masked spike count observations Y_{1:T} and parameterizes:
        q(z_0 | Y_{1:T}) = N(mu_{z_0}, diag(sigma_{z_0}^2))
    """

    def __init__(self, config: VLDMConfig):
        super().__init__()
        self.config = config
        obs_dim = config.obs_dim
        hidden_dim = config.encoder_hidden_dim
        latent_dim = config.latent_dim

        # Input linear projection with Anscombe-like stabilization
        self.input_proj = nn.Linear(obs_dim, hidden_dim)

        # Bidirectional GRU for sequence integration
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Projections to Gaussian parameters (mean and log-variance)
        combined_dim = hidden_dim * 2
        self.fc_mu = nn.Linear(combined_dim, latent_dim)
        self.fc_log_var = nn.Linear(combined_dim, latent_dim)

    def forward(
        self,
        counts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode counts sequence into variational posterior parameters.

        Args:
            counts: Observed spike counts [B, T, N].
            mask: Optional observation mask [B, T, N] where False indicates masked/unobserved channels.

        Returns:
            Tuple of (mu_0 [B, D], log_var_0 [B, D]).
        """
        # Apply mask: zero out unobserved spike counts
        if mask is not None:
            counts_in = counts * mask.float()
        else:
            counts_in = counts

        # Feature projection
        h = torch.relu(self.input_proj(counts_in))  # [B, T, hidden]
        out, _ = self.gru(h)  # [B, T, 2 * hidden]

        # Use initial bidirectional summary (first time bin captures forward context and backward summary)
        summary = out[:, 0, :]  # [B, 2 * hidden]

        mu = self.fc_mu(summary)  # [B, D]
        log_var = torch.clamp(self.fc_log_var(summary), min=-8.0, max=4.0)  # [B, D]

        return mu, log_var
