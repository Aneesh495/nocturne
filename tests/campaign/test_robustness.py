"""Unit tests for forecasting horizon and neuron dropout robustness campaigns."""

from __future__ import annotations

import pytest
import torch

from nocturne.campaign.robustness import (
    DropoutPoint,
    HorizonPoint,
    RobustnessCampaignResult,
    RobustnessCampaignRunner,
)
from nocturne.models.attention.config import AttentionConfig
from nocturne.models.attention.model import SpatiotemporalAttentionModel


@pytest.fixture
def mock_attention_model() -> SpatiotemporalAttentionModel:
    cfg = AttentionConfig(
        obs_dim=8,
        d_model=16,
        num_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_seq_len=64,
    )
    return SpatiotemporalAttentionModel(cfg)


@pytest.fixture
def test_spike_data() -> torch.Tensor:
    torch.manual_seed(42)
    # 4 trials, 40 bins, 8 channels of Poisson-like spike counts
    return torch.poisson(torch.full((4, 40, 8), 1.2))


def test_horizon_sweep_5_points(
    mock_attention_model: SpatiotemporalAttentionModel,
    test_spike_data: torch.Tensor,
) -> None:
    runner = RobustnessCampaignRunner(
        horizons=[1, 2, 5, 10, 20],
        bin_width_s=0.02,
    )
    points = runner.evaluate_horizon_sweep(
        model=mock_attention_model,
        counts=test_spike_data,
        history_bins=20,
    )

    assert len(points) == 5
    expected_horizons = [1, 2, 5, 10, 20]
    for pt, expected_h in zip(points, expected_horizons):
        assert isinstance(pt, HorizonPoint)
        assert pt.horizon_bins == expected_h
        assert pt.horizon_ms == expected_h * 20.0
        assert pt.poisson_nll > 0.0
        assert pt.poisson_deviance >= 0.0
        assert pt.latency_ms > 0.0


def test_dropout_sweep_5_rates(
    mock_attention_model: SpatiotemporalAttentionModel,
    test_spike_data: torch.Tensor,
) -> None:
    runner = RobustnessCampaignRunner(
        dropout_rates=[0.0, 0.1, 0.25, 0.5, 0.75],
        bin_width_s=0.02,
    )
    points = runner.evaluate_dropout_sweep(
        model=mock_attention_model,
        counts=test_spike_data,
        num_repeats=2,
        seed=123,
    )

    assert len(points) == 5
    expected_rates = [0.0, 0.1, 0.25, 0.5, 0.75]
    for pt, expected_r in zip(points, expected_rates):
        assert isinstance(pt, DropoutPoint)
        assert pt.dropout_rate == expected_r
        assert pt.poisson_nll > 0.0
        assert pt.poisson_deviance >= 0.0
        assert isinstance(pt.relative_loss_increase, float)

    # 0.0 dropout should have 0 relative loss increase
    assert points[0].relative_loss_increase == 0.0


def test_run_campaign_end_to_end(
    mock_attention_model: SpatiotemporalAttentionModel,
    test_spike_data: torch.Tensor,
) -> None:
    runner = RobustnessCampaignRunner(
        horizons=[1, 2, 5],
        dropout_rates=[0.0, 0.25, 0.5],
        bin_width_s=0.02,
    )
    res = runner.run_campaign(
        model=mock_attention_model,
        model_name="SpatiotemporalAttention",
        counts=test_spike_data,
        dataset_name="synthetic_oscillator",
        history_bins=25,
        num_repeats=2,
        seed=999,
    )

    assert isinstance(res, RobustnessCampaignResult)
    assert res.model_name == "SpatiotemporalAttention"
    assert res.dataset_name == "synthetic_oscillator"
    assert len(res.horizon_points) == 3
    assert len(res.dropout_points) == 3
    assert res.is_dropout_graceful is True

    as_dict = res.to_dict()
    assert as_dict["model_name"] == "SpatiotemporalAttention"
    assert "summary" in as_dict
    assert "horizon_points" in as_dict
    assert len(as_dict["horizon_points"]) == 3


def test_horizon_sweep_length_validation(
    mock_attention_model: SpatiotemporalAttentionModel,
) -> None:
    short_data = torch.ones((2, 10, 8))
    runner = RobustnessCampaignRunner(horizons=[5, 10, 20])
    with pytest.raises(ValueError, match="too short"):
        runner.evaluate_horizon_sweep(
            model=mock_attention_model,
            counts=short_data,
        )


def test_dropout_repeat_averaging_deterministic(
    mock_attention_model: SpatiotemporalAttentionModel,
    test_spike_data: torch.Tensor,
) -> None:
    runner = RobustnessCampaignRunner(dropout_rates=[0.25, 0.5])
    pts_1 = runner.evaluate_dropout_sweep(
        model=mock_attention_model,
        counts=test_spike_data,
        num_repeats=3,
        seed=42,
    )
    pts_2 = runner.evaluate_dropout_sweep(
        model=mock_attention_model,
        counts=test_spike_data,
        num_repeats=3,
        seed=42,
    )

    for p1, p2 in zip(pts_1, pts_2):
        assert pytest.approx(p1.poisson_nll) == p2.poisson_nll
        assert pytest.approx(p1.poisson_deviance) == p2.poisson_deviance
