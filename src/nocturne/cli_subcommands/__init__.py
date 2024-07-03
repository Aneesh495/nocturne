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
from nocturne.cli_subcommands.cmd_ablate import (
    run_ablation_command,
    add_ablation_arguments,
)
from nocturne.cli_subcommands.cmd_stream import (
    run_stream_command,
    add_stream_arguments,
)
from nocturne.cli_subcommands.cmd_raster import (
    run_raster_command,
    add_raster_arguments,
)
from nocturne.cli_subcommands.cmd_phase_portrait import (
    run_phase_portrait_command,
    add_phase_portrait_arguments,
)
from nocturne.cli_subcommands.cmd_matrix import (
    run_matrix_command,
    add_matrix_arguments,
)
from nocturne.cli_subcommands.cmd_falsify import (
    run_falsify_command,
    add_falsify_arguments,
)
from nocturne.cli_subcommands.cmd_sensitivity import (
    run_sensitivity_command,
    add_sensitivity_arguments,
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
    "run_ablation_command",
    "add_ablation_arguments",
    "run_stream_command",
    "add_stream_arguments",
    "run_raster_command",
    "add_raster_arguments",
    "run_phase_portrait_command",
    "add_phase_portrait_arguments",
    "run_matrix_command",
    "add_matrix_arguments",
    "run_falsify_command",
    "add_falsify_arguments",
    "run_sensitivity_command",
    "add_sensitivity_arguments",
]











