"""Unified command line interface for Nocturne neural dynamics laboratory."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from nocturne.cli_subcommands.cmd_doctor import run_doctor_command
from nocturne.cli_subcommands.cmd_synthetic import (
    add_synthetic_arguments,
    run_synthetic_command,
)
from nocturne.cli_subcommands.cmd_verify import (
    add_verify_arguments,
    run_verify_command,
)
from nocturne.cli_subcommands.cmd_acceptance import (
    add_acceptance_arguments,
    run_acceptance_command,
)
from nocturne.cli_subcommands.cmd_benchmark import (
    add_benchmark_arguments,
    run_benchmark_command,
)
from nocturne.cli_subcommands.cmd_ablate import (
    add_ablation_arguments,
    run_ablation_command,
)
from nocturne.cli_subcommands.cmd_stream import (
    add_stream_arguments,
    run_stream_command,
)
from nocturne.cli_subcommands.cmd_raster import (
    add_raster_arguments,
    run_raster_command,
)
from nocturne.cli_subcommands.cmd_phase_portrait import (
    add_phase_portrait_arguments,
    run_phase_portrait_command,
)
from nocturne.cli_subcommands.cmd_matrix import (
    add_matrix_arguments,
    run_matrix_command,
)
from nocturne.cli_subcommands.cmd_falsify import (
    add_falsify_arguments,
    run_falsify_command,
)
from nocturne.cli_subcommands.cmd_sensitivity import (
    add_sensitivity_arguments,
    run_sensitivity_command,
)
from nocturne.cli_subcommands.cmd_spectrogram import (
    add_spectrogram_arguments,
    run_spectrogram_command,
)
from nocturne.cli_subcommands.cmd_manifold import (
    add_manifold_arguments,
    run_manifold_command,
)


def build_parser() -> argparse.ArgumentParser:
    """Build unified argument parser with all registered subcommand suites."""
    parser = argparse.ArgumentParser(
        prog="nocturne",
        description="Nocturne: Probabilistic learning and empirical verification of neural population dynamics",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0 (neural population dynamics laboratory)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available laboratory subcommands")

    # doctor
    subparsers.add_parser("doctor", help="Probe hardware acceleration, PyTorch MPS/CUDA, and runtime environment")

    # synthetic
    p_synth = subparsers.add_parser("synthetic", help="Simulate and generate controlled synthetic dynamical circuits")
    add_synthetic_arguments(p_synth)

    # verify
    p_ver = subparsers.add_parser("verify", help="Execute complete system verification and negative controls (NC01-NC12)")
    add_verify_arguments(p_ver)

    # acceptance
    p_acc = subparsers.add_parser("acceptance", help="Build or verify cryptographic acceptance bundle and manifest")
    add_acceptance_arguments(p_acc)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Execute multi-seed model evaluation campaign across datasets")
    add_benchmark_arguments(p_bench)

    # ablate
    p_abl = subparsers.add_parser("ablate", help="Run five-family ablation experiments across model architectures")
    add_ablation_arguments(p_abl)

    # stream
    p_str = subparsers.add_parser("stream", help="Simulate online causal streaming and benchmark realtime latency bounds")
    add_stream_arguments(p_str)

    # raster
    p_rast = subparsers.add_parser("raster", help="Render neural spike rasters into terminal ASCII or standalone SVG")
    add_raster_arguments(p_rast)

    # phase_portrait
    p_phase = subparsers.add_parser("phase_portrait", help="Render continuous latent state trajectories and vector field phase planes")
    add_phase_portrait_arguments(p_phase)

    # matrix
    p_mat = subparsers.add_parser("matrix", help="Aggregate empirical benchmark runs into publication LaTeX and Markdown tables")
    add_matrix_arguments(p_mat)

    # falsify
    p_fals = subparsers.add_parser("falsify", help="Execute adversarial attractor falsification suite with Fourier surrogates")
    add_falsify_arguments(p_fals)

    # sensitivity
    p_sens = subparsers.add_parser("sensitivity", help="Compute hyperparameter sensitivity curves and dimensionality knee analysis")
    add_sensitivity_arguments(p_sens)

    # spectrogram
    p_spec = subparsers.add_parser("spectrogram", help="Compute time-frequency power spectrogram and render standalone vector SVG")
    add_spectrogram_arguments(p_spec)

    # manifold
    p_man = subparsers.add_parser("manifold", help="Render 3D isometric latent manifold orbit and surface mesh vector SVG")
    add_manifold_arguments(p_man)

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Entry point for the Nocturne CLI."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 0

    cmd = parsed.command

    if cmd == "doctor":
        return run_doctor_command()
    elif cmd == "synthetic":
        return run_synthetic_command(
            regime=parsed.regime,
            num_trials=parsed.trials,
            num_neurons=parsed.neurons,
            bin_width_ms=parsed.bin_width_ms,
            duration_s=parsed.duration,
            seed=parsed.seed,
        )
    elif cmd == "verify":
        return run_verify_command(
            output_path=parsed.output,
            controls_only=parsed.controls_only,
        )
    elif cmd == "acceptance":
        return run_acceptance_command(
            output_dir=parsed.output_dir,
            verify_only=parsed.verify_only,
        )
    elif cmd == "benchmark":
        seeds_tuple = tuple(int(s.strip()) for s in parsed.seeds.split(","))
        return run_benchmark_command(
            dataset=parsed.dataset,
            model=parsed.model,
            seeds=seeds_tuple,
            num_steps=parsed.steps,
            output_path=parsed.output,
        )
    elif cmd == "ablate":
        seeds_tuple = tuple(int(s.strip()) for s in parsed.seeds.split(","))
        return run_ablation_command(
            family=parsed.family,
            dataset=parsed.dataset,
            seeds=seeds_tuple,
            num_steps=parsed.steps,
            output_path=parsed.output,
        )
    elif cmd == "stream":
        return run_stream_command(
            num_neurons=parsed.neurons,
            chunk_size_bins=parsed.chunk_size,
            num_chunks=parsed.chunks,
            window_bins=parsed.window_bins,
            bin_width_ms=parsed.bin_width_ms,
            output_path=parsed.output,
        )
    elif cmd == "raster":
        return run_raster_command(
            dataset=parsed.dataset,
            format_type=parsed.format,
            max_bins=parsed.max_bins,
            max_neurons=parsed.max_neurons,
            output_path=parsed.output,
        )
    elif cmd == "phase_portrait":
        return run_phase_portrait_command(
            regime=parsed.regime,
            format_type=parsed.format,
            num_trials=parsed.num_trials,
            duration_s=parsed.duration,
            output_path=parsed.output,
        )
    elif cmd == "matrix":
        return run_matrix_command(
            benchmark_file=parsed.benchmark_file,
            output_dir=parsed.output_dir,
            format_type=parsed.format,
        )
    elif cmd == "falsify":
        return run_falsify_command(
            num_seeds=parsed.seeds,
            surrogate_shuffles=parsed.shuffles,
            output_path=parsed.output,
        )
    elif cmd == "sensitivity":
        return run_sensitivity_command(
            parameter=parsed.parameter,
            output_path=parsed.output,
        )
    elif cmd == "spectrogram":
        return run_spectrogram_command(
            dataset=parsed.dataset,
            max_bins=parsed.max_bins,
            num_freq_bins=parsed.num_freq_bins,
            output_path=parsed.output,
        )
    elif cmd == "manifold":
        return run_manifold_command(
            regime=parsed.regime,
            elevation_deg=parsed.elevation,
            azimuth_deg=parsed.azimuth,
            duration_s=parsed.duration,
            output_path=parsed.output,
        )
    else:
        print(f"Unknown subcommand: {cmd}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
