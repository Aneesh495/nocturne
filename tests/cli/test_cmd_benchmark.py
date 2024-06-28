"""Tests for benchmark campaign CLI subcommand."""

import argparse
from pathlib import Path
from nocturne.cli_subcommands.cmd_benchmark import (
    run_benchmark_command,
    add_benchmark_arguments,
)


def test_add_benchmark_arguments():
    parser = argparse.ArgumentParser()
    add_benchmark_arguments(parser)
    args = parser.parse_args(["--dataset", "mc_maze", "--model", "mean_rate", "--steps", "10"])
    assert args.dataset == "mc_maze"
    assert args.model == "mean_rate"
    assert args.steps == 10


def test_run_benchmark_command_execution(tmp_path, capsys):
    out_file = tmp_path / "bench_out.json"
    exit_code = run_benchmark_command(
        dataset="mc_maze",
        model="mean_rate",
        seeds=(42,),
        num_steps=5,
        output_path=str(out_file),
        verbose=True,
    )

    assert exit_code == 0
    assert out_file.exists()

    captured = capsys.readouterr()
    assert "Nocturne Benchmark Campaign" in captured.out
    assert "Held-in BPS" in captured.out
