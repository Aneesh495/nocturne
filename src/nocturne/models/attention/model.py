"""Spatiotemporal Attention Model (STA) for neural population dynamics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nocturne.models.attention.config import AttentionConfig
from nocturne.models.attention.embedding import SpatiotemporalEmbedding
from nocturne.models.attention.axial_block import AxialAttentionBlock
from nocturne.models.likelihoods import PoissonLoss


class SpatiotemporalAttentionModel(nn.Module):
    """Axial Spatiotemporal Masked Attention Model for neural population recordings.

    Combines tokenized count embeddings, learned neuron identity and time embeddings,
    stacked factorized temporal and spatial self-attention blocks, and a positive
    Poisson rate readout head.
    """

    def __init__(self, config: AttentionConfig, latent_dim: int = 16):
        super().__init__()
        self.config = config
        self.bin_width_s = config.bin_width_s
        self.softplus_beta = config.softplus_beta
        self.latent_dim = latent_dim

        # Input token embedding
        self.embedding = SpatiotemporalEmbedding(config)

        # Stack of axial transformer blocks
        self.blocks = nn.ModuleList([
            AxialAttentionBlock(config) for _ in range(config.num_layers)
        ])

        # Final normalization
        self.final_norm = nn.LayerNorm(config.d_model)

        # Non-negative firing rate readout head
        self.rate_head = nn.Linear(config.d_model, 1)

        # Latent trajectory bottleneck projection [B, T, latent_dim]
        self.latent_proj = nn.Linear(config.d_model, latent_dim)

        # Likelihood loss module
        self.loss_fn = PoissonLoss(bin_width_s=config.bin_width_s)

    def forward(
        self,
        counts: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        target_counts: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
        """Forward pass through spatiotemporal attention model.

        Args:
            counts: Observed spike counts [B, T, N].
            observed_mask: Boolean mask [B, T, N] where True indicates observed channel.
            target_counts: Optional ground truth target counts [B, T, N] for loss calculation.
            loss_mask: Optional boolean mask [B, T, N] indicating which bins/channels contribute to loss.
            causal: Override config.causal setting.
            key_padding_mask: Optional temporal padding mask [B, T].
            channel_mask: Optional spatial channel mask [B, N].

        Returns:
            Tuple of:
                rates_hz: Predicted non-negative firing rates [B, T, N] in Hz.
                latents: Latent population state trajectory [B, T, latent_dim].
                total_loss: Scalar loss tensor (or None if target_counts is None).
                metrics: Dictionary of diagnostic evaluation metrics.
        """
        # Step 1: Spatiotemporal token embedding [B, T, N, d_model]
        h = self.embedding(counts, observed_mask=observed_mask)

        # Step 2: Pass through stacked axial attention blocks
        for block in self.blocks:
            h = block(
                h,
                causal=causal,
                key_padding_mask=key_padding_mask,
                channel_mask=channel_mask,
            )

        h = self.final_norm(h)

        # Step 3: Rate readout via softplus [B, T, N]
        rate_logits = self.rate_head(h).squeeze(-1)
        expected_counts = F.softplus(rate_logits, beta=self.softplus_beta)
        rates_hz = expected_counts / self.bin_width_s

        # Step 4: Latent state projection [B, T, latent_dim]
        # Pool across neural channels to extract population-level trajectory
        h_pooled = h.mean(dim=2)  # [B, T, d_model]
        latents = self.latent_proj(h_pooled)

        # Step 5: Compute loss if target counts are provided
        total_loss: Optional[torch.Tensor] = None
        metrics: Dict[str, float] = {}

        if target_counts is not None:
            total_loss = self.loss_fn.forward_from_rates(
                rates_hz=rates_hz,
                targets=target_counts,
                mask=loss_mask,
            )
            metrics["poisson_loss"] = float(total_loss.item())
            metrics["mean_pred_rate"] = float(rates_hz.mean().item())

        return rates_hz, latents, total_loss, metrics
