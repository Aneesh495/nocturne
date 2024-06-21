"""Campaign and empirical benchmarking infrastructure.

Orchestrates multi-seed model evaluation, 5-family ablation matrices,
forecasting and missing-channel robustness sweeps, and scientific artifact aggregation.
"""

from __future__ import annotations

from nocturne.campaign.ablations import (
    AblationConfig,
    AblationFamily,
    AblationResult,
    AblationRunner,
)
from nocturne.campaign.benchmark import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    BenchmarkSuiteRunner,
)
from nocturne.campaign.matrix import (
    AblationGroupSummary,
    AggregatedMetric,
    BenchmarkGroupSummary,
    MatrixAggregator,
    SignificanceResult,
)
from nocturne.campaign.robustness import (
    DropoutPoint,
    HorizonPoint,
    RobustnessCampaignResult,
    RobustnessCampaignRunner,
)
from nocturne.campaign.sensitivity import (
    HyperparameterSensitivitySweep,
    SensitivityPoint,
    SensitivityCurve,
)

__all__ = [
    "AblationConfig",
    "AblationFamily",
    "AblationResult",
    "AblationRunner",
    "BenchmarkRunConfig",
    "BenchmarkRunResult",
    "BenchmarkSuiteRunner",
    "AblationGroupSummary",
    "AggregatedMetric",
    "BenchmarkGroupSummary",
    "MatrixAggregator",
    "SignificanceResult",
    "DropoutPoint",
    "HorizonPoint",
    "RobustnessCampaignResult",
    "RobustnessCampaignRunner",
    "HyperparameterSensitivitySweep",
    "SensitivityPoint",
    "SensitivityCurve",
]

