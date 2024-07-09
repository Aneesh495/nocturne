"""Tests for end-to-end synthetic circuit pipeline."""

from nocturne.workflows.synthetic_pipeline import (
    SyntheticPipelineResult,
    run_synthetic_pipeline,
)


def test_run_synthetic_pipeline():
    res: SyntheticPipelineResult = run_synthetic_pipeline(
        regime_name="nonlinear_oscillator",
        num_trials=3,
        num_neurons=8,
        duration_s=0.5,
        bin_width_ms=20,
        seed=42,
    )

    assert res.regime_name == "nonlinear_oscillator"
    assert res.num_trials == 3
    assert res.num_neurons == 8
    assert res.total_spikes > 0
    assert res.mean_trajectory_curvature > 0.0
    assert "<svg" in res.svg_phase_portrait
    assert "</svg>" in res.svg_phase_portrait
