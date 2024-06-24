"""Tests for system verify CLI subcommand."""

import argparse
from pathlib import Path
from nocturne.cli_subcommands.cmd_verify import (
    run_verify_command,
    add_verify_arguments,
)


def test_add_verify_arguments():
    parser = argparse.ArgumentParser()
    add_verify_arguments(parser)
    args = parser.parse_args(["--output", "report.json", "--controls-only"])
    assert args.output == "report.json"
    assert args.controls_only is True


def test_run_verify_command(tmp_path, capsys):
    out_file = tmp_path / "verify_report.json"
    exit_code = run_verify_command(output_path=str(out_file), verbose=True)

    assert exit_code == 0
    assert out_file.exists()
    assert out_file.stat().st_size > 100

    captured = capsys.readouterr()
    assert "Nocturne Verification & Negative Controls Audit" in captured.out
    assert "NC01" in captured.out
    assert "ALL CHECKS PASSED" in captured.out
