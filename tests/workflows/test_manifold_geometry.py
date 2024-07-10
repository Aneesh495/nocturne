"""Unit tests for the differential manifold geometry workflow."""

from __future__ import annotations

import pytest
from nocturne.workflows.manifold_geometry import (
    ManifoldGeometryWorkflowResult,
    run_manifold_geometry_workflow,
)


def test_run_manifold_geometry_workflow_oscillator() -> None:
    res = run_manifold_geometry_workflow(
        regime_name="nonlinear_oscillator",
        num_conditions=3,
        num_neurons=8,
        trial_duration_s=0.5,
        bin_width_ms=20,
        elevation_deg=25.0,
        azimuth_deg=45.0,
        seed=42,
    )

    assert isinstance(res, ManifoldGeometryWorkflowResult)
    assert res.regime_name == "nonlinear_oscillator"
    assert res.num_conditions == 3
    assert res.latent_dim == 2
    assert res.mean_path_length > 0.0
    assert res.mean_speed > 0.0
    assert res.mean_curvature >= 0.0
    assert res.max_curvature >= res.mean_curvature
    assert res.mean_intrinsic_dim > 0.0
    assert res.mean_grassmann_drift >= 0.0
    assert res.mean_procrustes_disparity >= 0.0
    assert len(res.condition_curvatures) == 3
    assert len(res.condition_intrinsic_dims) == 3
    assert "<svg" in res.svg_manifold_mesh
    assert "</svg>" in res.svg_manifold_mesh


def test_run_manifold_geometry_workflow_driven_recurrent() -> None:
    res = run_manifold_geometry_workflow(
        regime_name="driven_recurrent",
        num_conditions=2,
        num_neurons=8,
        trial_duration_s=0.5,
        bin_width_ms=20,
        seed=101,
    )

    assert isinstance(res, ManifoldGeometryWorkflowResult)
    assert res.latent_dim == 4
    assert res.num_conditions == 2
    assert res.mean_path_length > 0.0
