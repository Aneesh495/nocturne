"""Tests for acceptance bundle CLI subcommand."""

import argparse
from pathlib import Path
from nocturne.cli_subcommands.cmd_acceptance import (
    run_acceptance_command,
    add_acceptance_arguments,
    verify_bundle_integrity,
)


def test_add_acceptance_arguments():
    parser = argparse.ArgumentParser()
    add_acceptance_arguments(parser)
    args = parser.parse_args(["-o", "my_bundle", "--verify-only"])
    assert args.output_dir == "my_bundle"
    assert args.verify_only is True


def test_run_acceptance_command_build_and_verify(tmp_path, capsys):
    bundle_dir = tmp_path / "acceptance_test"
    # 1. Build bundle
    exit_code_build = run_acceptance_command(output_dir=str(bundle_dir), verify_only=False, verbose=True)
    assert exit_code_build == 0
    assert (bundle_dir / "ACCEPTANCE_MANIFEST.json").exists()

    captured = capsys.readouterr()
    assert "Building Acceptance Bundle" in captured.out
    assert "Manifest SHA-256:" in captured.out

    # 2. Verify integrity of generated bundle
    exit_code_verify = run_acceptance_command(output_dir=str(bundle_dir), verify_only=True, verbose=True)
    assert exit_code_verify == 0
    captured_verify = capsys.readouterr()
    assert "Integrity Status: VALID" in captured_verify.out


def test_verify_bundle_integrity_missing_manifest(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    valid = verify_bundle_integrity(empty_dir, verbose=False)
    assert valid is False
