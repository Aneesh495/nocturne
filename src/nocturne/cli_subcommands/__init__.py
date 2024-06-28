"""Modular CLI subcommand handlers."""

from __future__ import annotations

from nocturne.cli_subcommands.cmd_doctor import probe_environment, run_doctor_command
from nocturne.cli_subcommands.cmd_synthetic import (
    run_synthetic_command,
    add_synthetic_arguments,
)
from nocturne.cli_subcommands.cmd_verify import (
    run_verify_command,
    add_verify_arguments,
)
from nocturne.cli_subcommands.cmd_acceptance import (
    run_acceptance_command,
    add_acceptance_arguments,
    verify_bundle_integrity,
)
from nocturne.cli_subcommands.cmd_benchmark import (
    run_benchmark_command,
    add_benchmark_arguments,
)

__all__ = [
    "probe_environment",
    "run_doctor_command",
    "run_synthetic_command",
    "add_synthetic_arguments",
    "run_verify_command",
    "add_verify_arguments",
    "run_acceptance_command",
    "add_acceptance_arguments",
    "verify_bundle_integrity",
    "run_benchmark_command",
    "add_benchmark_arguments",
]




