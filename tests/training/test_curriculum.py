"""Tests for curriculum learning schedules and state serialization."""

import pytest
from nocturne.training.curriculum import CurriculumScheduler, CurriculumState


def test_curriculum_scheduler_linear_and_cosine():
    sched = CurriculumScheduler(
        total_epochs=100,
        warmup_epochs=10,
        initial_mask_rate=0.05,
        target_mask_rate=0.35,
        schedule_type="linear",
    )

    # During warmup
    assert sched.get_mask_rate(0) == 0.0
    assert sched.get_mask_rate(5) == pytest.approx(0.025)
    # At end of warmup
    assert sched.get_mask_rate(10) == pytest.approx(0.05)
    # Midpoint
    assert sched.get_mask_rate(55) == pytest.approx(0.20)
    # Final epoch
    assert sched.get_mask_rate(100) == pytest.approx(0.35)


def test_curriculum_strategy_transitions():
    custom_schedule = {
        0: "token",
        20: "channel",
        60: "temporal_tail",
    }
    sched = CurriculumScheduler(
        total_epochs=100,
        strategy_schedule=custom_schedule,
    )

    assert sched.get_mask_strategy(0) == "token"
    assert sched.get_mask_strategy(19) == "token"
    assert sched.get_mask_strategy(20) == "channel"
    assert sched.get_mask_strategy(59) == "channel"
    assert sched.get_mask_strategy(60) == "temporal_tail"
    assert sched.get_mask_strategy(100) == "temporal_tail"


def test_curriculum_sequence_length_progression():
    sched = CurriculumScheduler(
        total_epochs=50,
        min_seq_len=20,
        max_seq_len=100,
    )

    assert sched.get_sequence_length(0) == 20
    assert sched.get_sequence_length(25) == 60
    assert sched.get_sequence_length(50) == 100
    assert sched.get_sequence_length(60) == 100


def test_curriculum_step_and_serialization():
    sched = CurriculumScheduler(
        total_epochs=40,
        warmup_epochs=4,
        initial_mask_rate=0.10,
        target_mask_rate=0.40,
        min_seq_len=10,
        max_seq_len=50,
    )

    state: CurriculumState = sched.step(epoch=20)
    assert state.epoch == 20
    assert 0.10 <= state.mask_rate <= 0.40
    assert state.teacher_forcing_ratio == pytest.approx(0.50)
    assert state.max_sequence_length == 30

    sd = sched.state_dict()
    restored = CurriculumScheduler.from_state_dict(sd)
    assert restored.total_epochs == sched.total_epochs
    assert restored.initial_mask_rate == sched.initial_mask_rate
    assert restored.target_mask_rate == sched.target_mask_rate
    assert restored.min_seq_len == sched.min_seq_len


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="total_epochs must be > 0"):
        CurriculumScheduler(total_epochs=0)

    with pytest.raises(ValueError, match="Invalid warmup_epochs"):
        CurriculumScheduler(total_epochs=50, warmup_epochs=60)

    with pytest.raises(ValueError, match="Mask rates must be within"):
        CurriculumScheduler(initial_mask_rate=1.5)
