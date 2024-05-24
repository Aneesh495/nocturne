"""Tests verifying that all 12 executable negative controls are caught."""

from __future__ import annotations

import pytest
from nocturne.verification.controls import NegativeControlSuite


def test_all_negative_controls_caught():
    results = NegativeControlSuite.run_all()
    assert len(results) == 12

    for r in results:
        assert r.caught is True, f"Negative control {r.control_id} ({r.name}) was not caught: {r.error_message}"
        assert r.exception_type != "None"


def test_individual_negative_controls():
    # Test specific key controls individually
    nc01 = NegativeControlSuite.nc01_temporal_interval_inversion()
    assert nc01.caught is True

    nc02 = NegativeControlSuite.nc02_dataset_split_train_test_overlap()
    assert nc02.caught is True

    nc04 = NegativeControlSuite.nc04_bps_zero_spike_denominator()
    assert nc04.caught is True

    nc05 = NegativeControlSuite.nc05_causal_leakage_future_perturbation()
    assert nc05.caught is True

    nc11 = NegativeControlSuite.nc11_recovery_corrupt_nan_checkpoint()
    assert nc11.caught is True
