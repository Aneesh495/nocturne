"""Axial Spatiotemporal Masked Attention Architecture (STA)."""

from __future__ import annotations

from nocturne.models.attention.config import AttentionConfig
from nocturne.models.attention.embedding import SpatiotemporalEmbedding
from nocturne.models.attention.temporal import TemporalAttention
from nocturne.models.attention.spatial import SpatialAttention
from nocturne.models.attention.axial_block import AxialAttentionBlock
from nocturne.models.attention.routing import (
    generate_causal_mask,
    generate_cosmoothing_mask,
    generate_forecasting_mask,
)
from nocturne.models.attention.model import SpatiotemporalAttentionModel
from nocturne.models.attention.rotary import (
    ContinuousRotaryEmbedding,
    rotate_half,
)

__all__ = [
    "AttentionConfig",
    "SpatiotemporalEmbedding",
    "TemporalAttention",
    "SpatialAttention",
    "AxialAttentionBlock",
    "generate_causal_mask",
    "generate_cosmoothing_mask",
    "generate_forecasting_mask",
    "SpatiotemporalAttentionModel",
    "ContinuousRotaryEmbedding",
    "rotate_half",
]
