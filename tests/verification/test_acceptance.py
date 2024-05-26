"""Tests for acceptance bundle generator and cryptographic manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from nocturne.verification.acceptance import AcceptanceBundleGenerator, AcceptanceBundleReport


def test_acceptance_specs_generation():
    gen = AcceptanceBundleGenerator()

    env_spec = gen.generate_environment_spec()
    assert "python_version" in env_spec
    assert "torch_version" in env_spec
    assert "git_commit" in env_spec

    src_spec = gen.generate_source_inputs_spec()
    assert len(src_spec["datasets"]) == 3
    assert src_spec["datasets"][0]["dandi_dandiset_id"] == "000128"

    data_spec = gen.generate_datasets_spec()
    assert "mc_maze" in data_spec
    assert "mc_rtt" in data_spec
    assert "area2_bump" in data_spec
    assert data_spec["mc_maze"]["split_disjoint"] is True

    fp_spec = gen.generate_evaluation_fingerprints_spec()
    assert "mc_maze" in fp_spec["dataset_fingerprints"]
    assert len(fp_spec["dataset_fingerprints"]["mc_maze"]["sha256"]) == 64


def test_acceptance_bundle_build(tmp_path: Path):
    bundle_dir = tmp_path / "test_bundle"
    gen = AcceptanceBundleGenerator(output_dir=bundle_dir)
    report = gen.build_bundle()

    assert isinstance(report, AcceptanceBundleReport)
    assert report.all_checks_passed is True
    assert len(report.files_emitted) == 6

    # Verify all files exist
    expected_files = [
        "ENVIRONMENT.json",
        "SOURCE_INPUTS.json",
        "DATASETS.json",
        "EVALUATION_FINGERPRINTS.json",
        "SYSTEM_VERIFICATION.json",
        "ACCEPTANCE_MANIFEST.json",
    ]
    for ef in expected_files:
        assert (bundle_dir / ef).exists()

    # Verify cryptographic integrity of manifest
    manifest_file = bundle_dir / "ACCEPTANCE_MANIFEST.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    for ef in expected_files[:-1]:  # Manifest doesn't hash itself
        assert ef in manifest_data
        recorded_hash = manifest_data[ef]["sha256"]

        # Recalculate hash from file on disk
        with open(bundle_dir / ef, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert recorded_hash == actual_hash, f"Hash mismatch for {ef}"
