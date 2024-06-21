"""Tests for hyperparameter sensitivity sweeps and dimensionality saturation elbow analysis."""

import numpy as np
import pytest
from nocturne.campaign.sensitivity import (
    HyperparameterSensitivitySweep,
    SensitivityPoint,
    SensitivityCurve,
)


def test_sensitivity_elbow_detection():
    # Construct curve with clear saturation elbow at x = 4
    points = [
        SensitivityPoint("d", 2.0, 0.10, 0.20, 1.5, 0.1),
        SensitivityPoint("d", 4.0, 0.35, 0.65, 1.0, 0.2),  # Elbow
        SensitivityPoint("d", 8.0, 0.38, 0.68, 0.9, 0.4),
        SensitivityPoint("d", 16.0, 0.39, 0.69, 0.85, 0.8),
    ]
    curve = SensitivityCurve("d", points)
    elbow = curve.find_elbow_point()

    assert elbow.parameter_value == 4.0
    assert curve.values == [2.0, 4.0, 8.0, 16.0]
    assert len(curve.bps_values) == 4
    assert len(curve.r2_values) == 4


def test_sweep_latent_dimension():
    sweep = HyperparameterSensitivitySweep(random_seed=42)
    curve = sweep.sweep_latent_dimension(dimensions=[2, 4, 8], T=80, N=12)

    assert curve.parameter_name == "latent_dimension"
    assert len(curve.points) == 3
    for p in curve.points:
        assert isinstance(p.co_smoothing_bps, float)
        assert isinstance(p.decoding_r2, float)
        assert p.runtime_seconds > 0.0


def test_sweep_bin_width():
    sweep = HyperparameterSensitivitySweep(random_seed=42)
    curve = sweep.sweep_bin_width(bin_widths_ms=[10, 20, 50], total_time_s=1.0, N=12)

    assert curve.parameter_name == "bin_width_ms"
    assert len(curve.points) == 3
    for p in curve.points:
        assert isinstance(p.co_smoothing_bps, float)
        assert isinstance(p.decoding_r2, float)


def test_format_markdown_table():
    sweep = HyperparameterSensitivitySweep(random_seed=42)
    points = [
        SensitivityPoint("dim", 4.0, 0.25, 0.70, 1.2, 0.05),
        SensitivityPoint("dim", 8.0, 0.28, 0.72, 1.1, 0.10),
    ]
    curve = SensitivityCurve("dim", points)
    md = sweep.format_markdown_table(curve)

    assert "| `dim` | Co-smoothing BPS | Decoding $R^2$ |" in md
    assert "| 4 | 0.2500 | 0.7000 | 0.050 |" in md
