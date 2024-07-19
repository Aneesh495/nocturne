"""Automated verification suite executing all standalone examples in examples/."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"

EXAMPLE_SCRIPTS = [
    "quickstart_synthetic.py",
    "train_vldm_continuous.py",
    "axial_attention_co_smoothing.py",
    "realtime_streaming_demo.py",
    "fixed_point_topology.py",
    "full_ablation_sweep.py",
    "adversarial_falsification.py",
    "hyperparameter_knee_sweep.py",
]


@pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
def test_example_script_executes_successfully(script_name: str) -> None:
    script_path = EXAMPLES_DIR / script_name
    assert script_path.exists(), f"Example script {script_path} does not exist"

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, (
        f"Example {script_name} failed with exit code {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    assert len(proc.stdout) > 0, f"Example {script_name} produced no stdout output"
