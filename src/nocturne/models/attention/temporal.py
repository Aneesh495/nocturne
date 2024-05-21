"""Temporal attention module operating across the sequence duration dimension."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Multi-head attention applied along the temporal dimension for each neuron.

    Operates on spatiotemporal tensors of shape [B, T, N, d_model].
    Reshapes to [(B * N), T, d_model] for efficient batched self-attention across time.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.causal = causal

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
        causal: Optional[bool] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply temporal self-attention with residual connection and layer normalization.

        Args:
            x: Input tensor of shape [B, T, N, d_model].
            causal: If True, enforce causal attention mask. Defaults to self.causal.
            key_padding_mask: Optional boolean mask [B, T] where True indicates padded bins.

        Returns:
            Output tensor of shape [B, T, N, d_model].
        """
        B, T, N, D = x.shape
        use_causal = self.causal if causal is None else causal

        # Reshape [B, T, N, D] -> [B, N, T, D] -> [(B * N), T, D]
        x_perm = x.permute(0, 2, 1, 3).contiguous()
        flat_x = x_perm.view(B * N, T, D)

        # Build causal attention mask if requested: [T, T]
        attn_mask: Optional[torch.Tensor] = None
        if use_causal:
            # MultiheadAttention requires float or bool mask where True indicates masked position
            attn_mask = torch.triu(
                torch.ones((T, T), dtype=torch.bool, device=x.device),
                diagonal=1,
            )

        # Handle key padding mask: [B, T] -> repeat for N neurons -> [(B * N), T]
        flat_key_padding_mask: Optional[torch.Tensor] = None
        if key_padding_mask is not None:
            flat_key_padding_mask = (
                key_padding_mask.unsqueeze(1)
                .expand(B, N, T)
                .contiguous()
                .view(B * N, T)
            )

        attn_out, _ = self.self_attn(
            query=flat_x,
            key=flat_x,
            value=flat_x,
            attn_mask=attn_mask,
            key_padding_mask=flat_key_padding_mask,
            need_weights=False,
        )

        flat_x = flat_x + self.dropout(attn_out)
        flat_x = self.norm(flat_x)

        # Reshape back [(B * N), T, D] -> [B, N, T, D] -> [B, T, N, D]
        out = flat_x.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return out
