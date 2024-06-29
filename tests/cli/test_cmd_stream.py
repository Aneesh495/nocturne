"""Tests for streaming simulation CLI subcommand."""

import argparse
from pathlib import Path
from nocturne.cli_subcommands.cmd_stream import (
    run_stream_command,
    add_stream_arguments,
)


def test_add_stream_arguments():
    parser = argparse.ArgumentParser()
    add_stream_arguments(parser)
    args = parser.parse_args(["--neurons", "16", "--chunk-size", "2", "--chunks", "30"])
    assert args.neurons == 16
    assert args.chunk_size == 2
    assert args.chunks == 30


def test_run_stream_command_execution(tmp_path, capsys):
    out_file = tmp_path / "stream_metrics.json"
    exit_code = run_stream_command(
        num_neurons=12,
        chunk_size_bins=1,
        num_chunks=25,
        window_bins=20,
        bin_width_ms=50,
        output_path=str(out_file),
        verbose=True,
    )

    assert exit_code == 0
    assert out_file.exists()

    captured = capsys.readouterr()
    assert "Nocturne Online Causal Streaming Simulation" in captured.out
    assert "REALTIME CAPABLE" in captured.out
    assert "VERIFIED (Sliding Buffer Stable)" in captured.out
