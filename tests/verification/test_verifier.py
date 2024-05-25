"""Tests for system verifier and regression audit reporting."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from nocturne.verification.verifier import SystemVerifier, SystemVerificationReport


def test_system_verifier_checks_individually():
    math_chk = SystemVerifier.verify_math_exactness()
    assert math_chk.passed is True
    assert math_chk.check_id == "CHK_MATH_01"

    data_chk = SystemVerifier.verify_dataset_fixtures()
    assert data_chk.passed is True
    assert data_chk.check_id == "CHK_DATA_01"

    circ_chk = SystemVerifier.verify_synthetic_circuits()
    assert circ_chk.passed is True
    assert circ_chk.check_id == "CHK_CIRC_01"

    stream_chk = SystemVerifier.verify_causal_streaming()
    assert stream_chk.passed is True
    assert stream_chk.check_id == "CHK_STREAM_01"


def test_system_verifier_run_all():
    report = SystemVerifier.run_all()
    assert isinstance(report, SystemVerificationReport)
    assert report.all_passed is True
    assert report.failed_checks == 0
    assert report.total_checks == 16
    assert len(report.checks) == 4
    assert len(report.negative_controls) == 12
    assert "torch_version" in report.platform_info


def test_system_verifier_save_report(tmp_path: Path):
    out_file = tmp_path / "verification_report.json"
    saved_path = SystemVerifier.save_report(out_file)
    assert saved_path.exists()

    with open(saved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["all_passed"] is True
    assert data["total_checks"] == 16
    assert len(data["negative_controls"]) == 12
