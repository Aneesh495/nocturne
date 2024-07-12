"""Automated scientific and benchmarking workflows."""

from __future__ import annotations

from nocturne.workflows.distillation_workflow import (
    DistillationWorkflowResult,
    run_distillation_workflow,
)
from nocturne.workflows.generalization_campaign import (
    GeneralizationCampaignResult,
    run_generalization_campaign,
)
from nocturne.workflows.manifold_geometry import (
    ManifoldGeometryWorkflowResult,
    run_manifold_geometry_workflow,
)
from nocturne.workflows.reproducibility_audit import (
    ReproducibilityAuditResult,
    run_reproducibility_audit,
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
    "DistillationWorkflowResult",
    "GeneralizationCampaignResult",
    "ManifoldGeometryWorkflowResult",
    "ReproducibilityAuditResult",
    "StreamingSimulationResult",
    "SyntheticPipelineResult",
    "run_distillation_workflow",
    "run_generalization_campaign",
    "run_manifold_geometry_workflow",
    "run_reproducibility_audit",
    "run_streaming_simulation",
    "run_synthetic_pipeline",
]
