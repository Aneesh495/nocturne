"""Tests for doctor diagnostic subcommand."""

from nocturne.cli_subcommands.cmd_doctor import probe_environment, run_doctor_command


def test_probe_environment():
    report = probe_environment()
    assert report.python_version.startswith("3.")
    assert report.healthy is True
    assert report.architecture != ""
    assert report.device_available in ("cpu", "mps", "cuda") or "mps" in report.device_available or "cuda" in report.device_available


def test_run_doctor_command_stdout(capsys):
    exit_code = run_doctor_command(verbose=True)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "=== Nocturne System Diagnostics ===" in captured.out
    assert "Python:" in captured.out
    assert "HEALTHY" in captured.out
