"""Spatial attention module operating across the neural population dimension."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """Multi-head attention applied along the neural channel dimension for each time step.

    Operates on spatiotemporal tensors of shape [B, T, N, d_model].
    Reshapes to [(B * T), N, d_model] for efficient batched self-attention across neurons.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply spatial population attention with residual connection and layer normalization.

        Args:
            x: Input tensor of shape [B, T, N, d_model].
            channel_mask: Optional boolean mask [B, N] where True indicates masked (unobserved) channels.

        Returns:
            Output tensor of shape [B, T, N, d_model].
        """
        B, T, N, D = x.shape

        # Reshape [B, T, N, D] -> [(B * T), N, D]
        flat_x = x.view(B * T, N, D)

        flat_key_padding_mask: Optional[torch.Tensor] = None
        if channel_mask is not None:
            # channel_mask is [B, N], expand to [B, T, N] -> [(B * T), N]
            flat_key_padding_mask = (
                channel_mask.unsqueeze(1)
                .expand(B, T, N)
                .contiguous()
                .view(B * T, N)
            )

        attn_out, _ = self.self_attn(
            query=flat_x,
            key=flat_x,
            value=flat_x,
            key_padding_mask=flat_key_padding_mask,
            need_weights=False,
        )

        flat_x = flat_x + self.dropout(attn_out)
        flat_x = self.norm(flat_x)

        return flat_x.view(B, T, N, D)
