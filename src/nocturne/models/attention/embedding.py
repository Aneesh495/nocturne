"""Spatiotemporal token embedding and learned neuron identity representations."""

from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn

from nocturne.models.attention.config import AttentionConfig


class SpatiotemporalEmbedding(nn.Module):
    """Embeds scalar spike counts, learned neuron identities, and temporal positions.

    Token embedding formulation:
        h_{b,t,n} = W_in * y_{b,t,n} + E_{neuron}[n] + E_{time}[t]
    Unobserved / held-out tokens are substituted with a learned [MASK] token.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.obs_dim = config.obs_dim

        # Scalar count projection
        self.count_proj = nn.Linear(1, self.d_model)

        # Learned [MASK] token for unobserved channels
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, self.d_model) * 0.02)

        # Learned neuron identity embeddings [N, d_model]
        self.neuron_embeddings = nn.Parameter(
            torch.randn(self.obs_dim, self.d_model) / math.sqrt(self.d_model)
        )

        # Learned temporal position embeddings [max_seq_len, d_model]
        self.temporal_embeddings = nn.Parameter(
            torch.randn(config.max_seq_len, self.d_model) / math.sqrt(self.d_model)
        )

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        counts: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Construct token representation tensor of shape [B, T, N, d_model].

        Args:
            counts: Observed spike counts [B, T, N].
            observed_mask: Optional boolean mask [B, T, N] where True indicates visible channel/bin.

        Returns:
            Embedded representation tensor [B, T, N, d_model].
        """
        B, T, N = counts.shape
        device = counts.device

        # Project counts [B, T, N, 1] -> [B, T, N, d_model]
        counts_4d = counts.unsqueeze(-1)
        count_embed = self.count_proj(counts_4d)

        # Substitute unobserved tokens with learned mask token
        if observed_mask is not None:
            mask_4d = observed_mask.unsqueeze(-1)  # [B, T, N, 1]
            token_embed = torch.where(mask_4d, count_embed, self.mask_token.expand(B, T, N, self.d_model))
        else:
            token_embed = count_embed

        # Add neuron identity embeddings [1, 1, N, d_model]
        neuron_embed = self.neuron_embeddings[:N].unsqueeze(0).unsqueeze(0)
        # Add temporal position embeddings [1, T, 1, d_model]
        time_embed = self.temporal_embeddings[:T].unsqueeze(0).unsqueeze(2)

        h = token_embed + neuron_embed + time_embed
        return self.dropout(self.layer_norm(h))
