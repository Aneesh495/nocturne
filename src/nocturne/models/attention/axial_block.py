"""Factorized axial transformer block interleaving temporal and spatial attention."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

from nocturne.models.attention.config import AttentionConfig
from nocturne.models.attention.temporal import TemporalAttention
from nocturne.models.attention.spatial import SpatialAttention


class AxialAttentionBlock(nn.Module):
    """Axial transformer block factoring spatiotemporal attention into rank-1 subspaces.

    Sequentially applies:
    1. Temporal self-attention across sequence duration T: O(B * N * T^2)
    2. Spatial population self-attention across channels N: O(B * T * N^2)
    3. Position-wise point feedforward network: O(B * T * N * d_ff)

    This factorized axial design avoids full 2D quadratic attention O((T * N)^2),
    enabling scaling to long recording windows and large neural populations.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config

        self.temporal_attn = TemporalAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            causal=config.causal,
        )

        self.spatial_attn = SpatialAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal: Optional[bool] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through axial attention block.

        Args:
            x: Input tensor [B, T, N, d_model].
            causal: Override causal masking flag.
            key_padding_mask: Temporal padding mask [B, T].
            channel_mask: Spatial channel mask [B, N].

        Returns:
            Transformed spatiotemporal tensor [B, T, N, d_model].
        """
        # Step 1: Temporal attention across time
        x = self.temporal_attn(
            x,
            causal=causal,
            key_padding_mask=key_padding_mask,
        )

        # Step 2: Spatial attention across population
        x = self.spatial_attn(
            x,
            channel_mask=channel_mask,
        )

        # Step 3: Feedforward MLP with residual and norm
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        return x
