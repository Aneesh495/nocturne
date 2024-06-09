"""Unit tests for sparse neighborhood electrode graph spatial attention."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nocturne.models.attention.sparse_spatial import (
    ElectrodeAdjacencyGraph,
    SparseSpatialAttention,
)


def test_electrode_adjacency_graph_linear_probe() -> None:
    # 10 electrodes placed linearly along z-axis with 50 um pitch: (0, 0, 50*i)
    coords = np.zeros((10, 3))
    coords[:, 2] = np.arange(10) * 50.0

    # Max radius 75 um connects immediate neighbors (distance 50 um <= 75 um)
    # but not 2nd neighbors (distance 100 um > 75 um)
    graph = ElectrodeAdjacencyGraph(coords, max_radius_um=75.0)

    assert graph.num_channels == 10
    mask = graph.adjacency_mask

    # Interior channel (e.g. index 3) should connect to 2, 3, 4 (3 channels)
    assert np.sum(mask[3]) == 3
    assert mask[3, 2]
    assert mask[3, 3]
    assert mask[3, 4]
    assert not mask[3, 5]

    # Border channel (index 0) connects to 0 and 1 (2 channels)
    assert np.sum(mask[0]) == 2

    # Graph should have positive sparsity ratio
    assert graph.get_sparsity_ratio() > 0.60


def test_sparse_spatial_attention_forward_and_gradients() -> None:
    BT, N, d_model = 4, 8, 16
    coords = np.random.randn(N, 2) * 100.0
    graph = ElectrodeAdjacencyGraph(coords, k_neighbors=3)
    spatial_mask = graph.to_torch_mask()

    attn = SparseSpatialAttention(d_model=d_model, num_heads=2)
    x = torch.randn(BT, N, d_model, requires_grad=True)

    out = attn(x, spatial_mask=spatial_mask)

    assert out.shape == (BT, N, d_model)
    assert torch.all(torch.isfinite(out))

    loss = torch.sum(out**2)
    loss.backward()

    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))


def test_d_model_divisibility_error() -> None:
    with pytest.raises(ValueError, match="divisible by num_heads"):
        SparseSpatialAttention(d_model=17, num_heads=4)
