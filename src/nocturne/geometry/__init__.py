"""Latent geometry, gauge alignment, fixed points, and dynamics audits."""

from __future__ import annotations

from nocturne.geometry.alignment import (
    ProcrustesAlignmentResult,
    OrthogonalProcrustesAligner,
)
from nocturne.geometry.cca import (
    CCAResult,
    CanonicalCorrelationAnalysis,
)
from nocturne.geometry.fixed_points import (
    FixedPoint,
    FixedPointFinder,
)
from nocturne.geometry.jacobians import (
    JacobianDiagnostics,
    JacobianAnalyzer,
)
from nocturne.geometry.audit import (
    GeometricDivergenceReport,
    PredictionStructureDivergenceAudit,
)

__all__ = [
    "ProcrustesAlignmentResult",
    "OrthogonalProcrustesAligner",
    "CCAResult",
    "CanonicalCorrelationAnalysis",
    "FixedPoint",
    "FixedPointFinder",
    "JacobianDiagnostics",
    "JacobianAnalyzer",
    "GeometricDivergenceReport",
    "PredictionStructureDivergenceAudit",
]
