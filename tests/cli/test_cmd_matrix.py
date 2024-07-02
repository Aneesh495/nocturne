"""Tests for benchmark matrix aggregation CLI subcommand."""

import argparse
import json
from pathlib import Path
from nocturne.cli_subcommands.cmd_matrix import (
    run_matrix_command,
    add_matrix_arguments,
)


def test_add_matrix_arguments():
    parser = argparse.ArgumentParser()
    add_matrix_arguments(parser)
    args = parser.parse_args(["-b", "results.json", "-o", "matrix_dir", "--format", "latex"])
    assert args.benchmark_file == "results.json"
    assert args.output_dir == "matrix_dir"
    assert args.format == "latex"


def test_run_matrix_command_default_and_export(tmp_path, capsys):
    out_dir = tmp_path / "matrix_artifacts"
    exit_code = run_matrix_command(
        output_dir=str(out_dir),
        format_type="all",
        verbose=True,
    )

    assert exit_code == 0
    assert (out_dir / "benchmark_matrix.md").exists()
    assert (out_dir / "benchmark_matrix.tex").exists()
    assert (out_dir / "benchmark_matrix.csv").exists()

    captured = capsys.readouterr()
    assert "Nocturne Scientific Benchmark Matrix" in captured.out
    assert r"\begin{table}" in captured.out
