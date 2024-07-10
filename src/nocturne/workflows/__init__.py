"""Automated scientific and benchmarking workflows."""

from __future__ import annotations

from nocturne.workflows.manifold_geometry import (
    ManifoldGeometryWorkflowResult,
    run_manifold_geometry_workflow,
)
from nocturne.workflows.streaming_simulation import (
    StreamingSimulationResult,
    run_streaming_simulation,
)
from nocturne.workflows.synthetic_pipeline import (
    SyntheticPipelineResult,
    run_synthetic_pipeline,
)

__all__ = [
    "ManifoldGeometryWorkflowResult",
    "StreamingSimulationResult",
    "SyntheticPipelineResult",
    "run_manifold_geometry_workflow",
    "run_streaming_simulation",
    "run_synthetic_pipeline",
]
