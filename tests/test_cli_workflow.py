"""Unit tests for the workflow CLI subcommand dispatcher."""

from __future__ import annotations

import pytest
from nocturne.cli import main


def test_cli_workflow_synthetic(capsys: pytest.CaptureFixture) -> None:
    code = main(["workflow", "--name", "synthetic", "--trials", "2", "--neurons", "8", "--duration", "0.5"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Nocturne Automated Workflow: synthetic" in captured.out
    assert "Total Spikes Emitted" in captured.out


def test_cli_workflow_streaming(capsys: pytest.CaptureFixture) -> None:
    code = main(["workflow", "--name", "streaming", "--neurons", "8", "--duration", "0.4", "--bin-width-ms", "20"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Nocturne Automated Workflow: streaming" in captured.out
    assert "Mean Processing Latency" in captured.out


def test_cli_workflow_manifold(capsys: pytest.CaptureFixture) -> None:
    code = main(["workflow", "--name", "manifold", "--trials", "2", "--neurons", "8", "--duration", "0.5"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Nocturne Automated Workflow: manifold" in captured.out
    assert "Mean Extrinsic Curvature" in captured.out


def test_cli_workflow_reproducibility(capsys: pytest.CaptureFixture) -> None:
    code = main(["workflow", "--name", "reproducibility", "--neurons", "8", "--seed", "42"])
    assert code == 0
    captured = capsys.readouterr()
    assert "Nocturne Automated Workflow: reproducibility" in captured.out
    assert "PASSED" in captured.out
