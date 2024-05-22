"""Tests for inference contracts, request validation, and result containers."""

from __future__ import annotations

import torch
import pytest

from nocturne.inference.contracts import (
    InferenceMode,
    InferenceRequest,
    InferenceResult,
)


def test_inference_request_valid():
    counts = torch.randint(0, 5, (2, 20, 10), dtype=torch.float32)
    req = InferenceRequest(
        counts=counts,
        mode=InferenceMode.OFFLINE_SMOOTHING,
        held_in_indices=[0, 1, 2, 3],
        held_out_indices=[4, 5],
        forecast_horizon_bins=0,
    )
    assert req.mode == InferenceMode.OFFLINE_SMOOTHING
    assert req.counts.shape == (2, 20, 10)
    assert req.held_in_indices == [0, 1, 2, 3]


def test_inference_request_invalid_shapes():
    with pytest.raises(ValueError, match="counts must have 3 dimensions"):
        InferenceRequest(
            counts=torch.zeros((10, 5)),
            mode=InferenceMode.ONLINE_FILTERING,
        )

    with pytest.raises(ValueError, match="counts cannot contain negative values"):
        InferenceRequest(
            counts=torch.tensor([[[-1.0]]]),
            mode=InferenceMode.ONLINE_FILTERING,
        )
