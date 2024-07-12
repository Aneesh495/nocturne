"""Unit tests for the teacher-student knowledge distillation workflow."""

from __future__ import annotations

import pytest
from nocturne.workflows.distillation_workflow import (
    DistillationWorkflowResult,
    run_distillation_workflow,
)


def test_run_distillation_workflow_basic() -> None:
    res = run_distillation_workflow(
        num_neurons=8,
        teacher_d_model=16,
        student_d_model=8,
        teacher_layers=1,
        student_layers=1,
        latent_dim=2,
        num_steps=5,
        batch_size=2,
        seq_len=16,
        seed=42,
    )

    assert isinstance(res, DistillationWorkflowResult)
    assert res.teacher_params > res.student_params
    assert res.parameter_compression_ratio > 1.0
    assert len(res.loss_history) == 5
    assert res.final_total_loss > 0.0
    assert res.student_latency_ms > 0.0
    assert res.teacher_latency_ms > 0.0
    assert -1.0 <= res.teacher_student_rate_correlation <= 1.0
