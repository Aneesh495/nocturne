"""Sparse neighborhood electrode graph spatial attention for high-density arrays.

Restricts spatial attention routing to anatomically or physically proximate electrode
channels according to physical probe Euclidean geometry and k-nearest neighbor graphs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ElectrodeAdjacencyGraph:
    """Constructs spatial adjacency masks based on physical recording electrode coordinates."""

    def __init__(
        self,
        coordinates: np.ndarray | torch.Tensor,
        max_radius_um: Optional[float] = None,
        k_neighbors: Optional[int] = None,
    ):
        """Initialize electrode graph with physical 2D/3D coordinates.

        Args:
            coordinates: Electrode positions [N, D] in micrometers (D in {2, 3}).
            max_radius_um: Metric radius threshold in micrometers.
            k_neighbors: Optional k-nearest neighbors constraint per channel.
        """
        if isinstance(coordinates, torch.Tensor):
            coords = coordinates.detach().cpu().numpy().astype(np.float64)
        else:
            coords = np.asarray(coordinates, dtype=np.float64)

        if coords.ndim != 2:
            raise ValueError(f"Coordinates must have shape [N, D], got {coords.shape}")

        N, D = coords.shape
        self.num_channels = N
        self.dim = D
        self.coordinates = coords

        # Compute pairwise Euclidean distance matrix [N, N]
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        self.dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        # Build boolean adjacency mask: self-loops are always connected
        adj = np.eye(N, dtype=bool)

        if max_radius_um is not None:
            adj = adj | (self.dist_matrix <= max_radius_um)

        if k_neighbors is not None:
            k = min(k_neighbors, N)
            for i in range(N):
                nearest_idx = np.argsort(self.dist_matrix[i])[:k]
                adj[i, nearest_idx] = True

        self.adjacency_mask = adj

    def to_torch_mask(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return boolean attention mask tensor [N, N] where True indicates allowable attention."""
        return torch.tensor(self.adjacency_mask, dtype=torch.bool, device=device)

    def get_sparsity_ratio(self) -> float:
        """Fraction of pruned edges in the spatial attention graph."""
        total_pairs = self.num_channels * self.num_channels
        active_pairs = int(np.sum(self.adjacency_mask))
        return 1.0 - (active_pairs / float(total_pairs))


class SparseSpatialAttention(nn.Module):
    """Multi-head spatial attention restricted to physical electrode neighborhoods."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply sparse spatial multi-head attention across channels.

        Args:
            x: Input token tensor of shape [B * T, N, d_model].
            spatial_mask: Boolean tensor of shape [N, N] or [B * T, N, N] where
                          True indicates allowable connection and False is masked.

        Returns:
            Output tensor of shape [B * T, N, d_model].
        """
        BT, N, d = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x).view(BT, N, H, D).transpose(1, 2)  # [BT, H, N, D]
        k = self.k_proj(x).view(BT, N, H, D).transpose(1, 2)  # [BT, H, N, D]
        v = self.v_proj(x).view(BT, N, H, D).transpose(1, 2)  # [BT, H, N, D]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [BT, H, N, N]

        if spatial_mask is not None:
            # Mask should be boolean: False positions get -1e9
            if spatial_mask.dtype == torch.bool:
                if spatial_mask.ndim == 2:  # [N, N]
                    mask_expanded = spatial_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, N, N]
                else:  # [BT, N, N]
                    mask_expanded = spatial_mask.unsqueeze(1)  # [BT, 1, N, N]
                scores = scores.masked_fill(~mask_expanded, -1e9)
            else:
                scores = scores + spatial_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)  # [BT, H, N, D]
        context = context.transpose(1, 2).contiguous().view(BT, N, d)  # [BT, N, d_model]

        return self.out_proj(context)
