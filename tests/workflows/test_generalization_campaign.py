"""Unit tests for the Leave-One-Condition-Out (LOCO) generalization campaign workflow."""

from __future__ import annotations

import pytest
from nocturne.workflows.generalization_campaign import (
    GeneralizationCampaignResult,
    run_generalization_campaign,
)


def test_run_generalization_campaign_basic() -> None:
    res = run_generalization_campaign(
        regime_name="driven_recurrent",
        num_conditions=3,
        trials_per_condition=4,
        num_neurons=8,
        trial_duration_s=0.5,
        bin_width_ms=20,
        gap_threshold=0.50,
        seed=42,
    )

    assert isinstance(res, GeneralizationCampaignResult)
    assert res.regime_name == "driven_recurrent"
    assert res.num_conditions == 3
    assert res.num_trials_per_condition == 4
    assert len(res.condition_scores) == 3
    assert res.generalization_gap >= -1.0
    assert "Cross-Condition Generalization Campaign" in res.summary_markdown
    assert "| condition_1 |" in res.summary_markdown
