"""Tests for ablation matrix CLI subcommand."""

import argparse
from nocturne.cli_subcommands.cmd_ablate import (
    run_ablation_command,
    add_ablation_arguments,
)


def test_add_ablation_arguments():
    parser = argparse.ArgumentParser()
    add_ablation_arguments(parser)
    args = parser.parse_args(["--family", "depth", "--dataset", "mc_maze", "--steps", "8"])
    assert args.family == "depth"
    assert args.dataset == "mc_maze"
    assert args.steps == 8


def test_run_ablation_command_single_family(tmp_path, capsys):
    out_file = tmp_path / "ablation_out.json"
    exit_code = run_ablation_command(
        family="causality",
        dataset="mc_maze",
        seeds=(42,),
        num_steps=5,
        output_path=str(out_file),
        verbose=True,
    )

    assert exit_code == 0
    assert out_file.exists()

    captured = capsys.readouterr()
    assert "Nocturne Five-Family Ablation Engine" in captured.out
    assert "causal_temporal" in captured.out or "bidirectional_full" in captured.out


def test_run_ablation_command_invalid_family():
    exit_code = run_ablation_command(
        family="non_existent_family",
        verbose=False,
    )
    assert exit_code == 1
