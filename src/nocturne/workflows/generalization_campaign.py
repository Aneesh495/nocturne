"""End-to-end scientific workflow: Leave-One-Condition-Out (LOCO) generalization campaign."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from nocturne.evaluation.cross_condition import (
    ConditionMetricRecord,
    CrossConditionEvaluator,
    CrossConditionResult,
)
from nocturne.synthetic.generator import SyntheticDatasetGenerator


@dataclass(frozen=True)
class GeneralizationCampaignResult:
    """Quantitative outcome of cross-condition generalization campaign."""

    regime_name: str
    num_conditions: int
    num_trials_per_condition: int
    in_distribution_mean_r2: float
    out_of_distribution_mean_r2: float
    generalization_gap: float
    is_robust_generalizer: bool
    condition_scores: Dict[str, float]
    summary_markdown: str


def run_generalization_campaign(
    regime_name: str = "driven_recurrent",
    num_conditions: int = 4,
    trials_per_condition: int = 6,
    num_neurons: int = 16,
    trial_duration_s: float = 1.0,
    bin_width_ms: int = 20,
    ridge_alpha: float = 1.0,
    gap_threshold: float = 0.35,
    seed: int = 42,
) -> GeneralizationCampaignResult:
    """Execute Leave-One-Condition-Out (LOCO) generalization audit across experimental conditions.

    Args:
        regime_name: Synthetic circuit regime identifier.
        num_conditions: Number of behavioral conditions or reaching targets.
        trials_per_condition: Number of independent trials generated per condition.
        num_neurons: Monitored neural population channel count.
        trial_duration_s: Trial duration in seconds.
        bin_width_ms: Sampling bin width in milliseconds.
        ridge_alpha: L2 regularizer coefficient for the linear readout probe.
        gap_threshold: Maximum allowable in-distribution vs out-of-distribution drop.
        seed: Random seed for repeatability.

    Returns:
        GeneralizationCampaignResult containing evaluation metrics and Markdown report.
    """
    bin_width_s = bin_width_ms / 1000.0
    generator = SyntheticDatasetGenerator(seed=seed)

    latents_by_condition: Dict[str, np.ndarray] = {}
    behavior_by_condition: Dict[str, np.ndarray] = {}

    # 1. Generate multi-trial neural and behavioral data per condition
    for c_idx in range(num_conditions):
        cond_label = f"condition_{c_idx + 1}"
        c_seed = seed + c_idx * 100

        # Instantiate generator for condition
        c_gen = SyntheticDatasetGenerator(seed=c_seed)
        sims, _ = c_gen.generate_regime_corpus(
            regime_name=regime_name,
            num_trials=trials_per_condition,
            num_neurons=num_neurons,
            bin_width_s=bin_width_s,
            trial_duration_s=trial_duration_s,
        )

        # Extract latents [B_c, T, D]
        lat_trials = np.stack([s.latent.states for s in sims], axis=0)
        latents_by_condition[cond_label] = lat_trials

        # Synthesize behavioral kinematics (e.g. 2D velocity / position) modulated by condition
        angle = (2.0 * np.pi * c_idx) / num_conditions
        B_c, T, D = lat_trials.shape
        # Kinematics correlate with latent projections plus directional offset
        t_vec = np.linspace(0, trial_duration_s, T)
        kin_base_x = np.cos(angle) * np.sin(2.0 * np.pi * t_vec)
        kin_base_y = np.sin(angle) * np.sin(2.0 * np.pi * t_vec)
        kin_base = np.stack([kin_base_x, kin_base_y], axis=-1)  # [T, 2]

        # Add small trial-to-trial jitter
        kin_trials = np.zeros((B_c, T, 2), dtype=np.float64)
        for b in range(B_c):
            noise = c_gen.rng.normal(0, 0.05, size=(T, 2))
            kin_trials[b] = kin_base + noise

        behavior_by_condition[cond_label] = kin_trials

    # 2. Run LOCO cross-validation evaluation
    evaluator = CrossConditionEvaluator(
        ridge_alpha=ridge_alpha,
        gap_threshold=gap_threshold,
    )
    loco_res: CrossConditionResult = evaluator.evaluate_loco(
        latents_by_condition=latents_by_condition,
        behavior_by_condition=behavior_by_condition,
    )

    # 3. Compile condition scores dictionary
    cond_scores: Dict[str, float] = {
        rec.condition_label: float(rec.probe_r2)
        for rec in loco_res.condition_records
    }

    # 4. Generate structured Markdown summary
    summary_lines = [
        f"# Cross-Condition Generalization Campaign: {regime_name}",
        "",
        f"- Number of Conditions: {num_conditions}",
        f"- Trials per Condition: {trials_per_condition}",
        f"- In-Distribution Mean R2: {loco_res.in_distribution_mean_r2:.4f}",
        f"- Out-of-Distribution Mean R2: {loco_res.out_of_distribution_mean_r2:.4f}",
        f"- Generalization Gap: {loco_res.generalization_gap:.4f}",
        f"- Robust Generalizer (Threshold {gap_threshold:.2f}): {loco_res.is_robust_generalizer}",
        "",
        "## Per-Condition Out-of-Distribution Decoding R2",
        "",
        "| Condition | Trials | Held-Out R2 | Poisson Deviance |",
        "|:---|:---:|:---:|:---:|",
    ]

    for rec in loco_res.condition_records:
        summary_lines.append(
            f"| {rec.condition_label} | {rec.num_trials} | {rec.probe_r2:.4f} | {rec.poisson_deviance:.4f} |"
        )

    summary_md = "\n".join(summary_lines)

    return GeneralizationCampaignResult(
        regime_name=regime_name,
        num_conditions=num_conditions,
        num_trials_per_condition=trials_per_condition,
        in_distribution_mean_r2=loco_res.in_distribution_mean_r2,
        out_of_distribution_mean_r2=loco_res.out_of_distribution_mean_r2,
        generalization_gap=loco_res.generalization_gap,
        is_robust_generalizer=loco_res.is_robust_generalizer,
        condition_scores=cond_scores,
        summary_markdown=summary_md,
    )
