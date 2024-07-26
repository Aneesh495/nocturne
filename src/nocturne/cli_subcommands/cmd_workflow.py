"""Scientific laboratory workflows CLI subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from nocturne.workflows import (
    run_distillation_workflow,
    run_generalization_campaign,
    run_manifold_geometry_workflow,
    run_reproducibility_audit,
    run_streaming_simulation,
    run_synthetic_pipeline,
)


def run_workflow_command(
    name: str = "synthetic",
    num_trials: int = 4,
    num_neurons: int = 16,
    duration_s: float = 1.0,
    bin_width_ms: int = 20,
    seed: int = 42,
    output_svg: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Execute automated scientific workflow from the command line.

    Args:
        name: Name of laboratory workflow to execute.
        num_trials: Trial or condition count.
        num_neurons: Population neuron count.
        duration_s: Duration in seconds.
        bin_width_ms: Sampling bin width in milliseconds.
        seed: Random seed.
        output_svg: Optional path to save generated SVG graphics.
        verbose: Verbose terminal output.

    Returns:
        Process exit code (0 for success).
    """
    if verbose:
        print(f"=== Nocturne Automated Workflow: {name} (Seed={seed}) ===")

    if name == "synthetic":
        res_syn = run_synthetic_pipeline(
            regime_name="nonlinear_oscillator",
            num_trials=num_trials,
            num_neurons=num_neurons,
            duration_s=duration_s,
            bin_width_ms=bin_width_ms,
            seed=seed,
        )
        if verbose:
            print(f"  Regime                   : {res_syn.regime_name}")
            print(f"  Total Spikes Emitted     : {res_syn.total_spikes}")
            print(f"  Identified Fixed Points  : {res_syn.identified_fixed_points}")
            print(f"  Mean Trajectory Curvature: {res_syn.mean_trajectory_curvature:.4f}")
            print(f"  Reconstruction Pseudo-R2 : {res_syn.reconstruction_r2:.4f}")
        if output_svg and res_syn.svg_phase_portrait:
            Path(output_svg).write_text(res_syn.svg_phase_portrait, encoding="utf-8")
            if verbose:
                print(f"  Phase portrait SVG saved to: {output_svg}")

    elif name == "streaming":
        total_bins = int(duration_s * 1000.0 / bin_width_ms)
        res_stream = run_streaming_simulation(
            num_neurons=num_neurons,
            latent_dim=4,
            total_bins=total_bins,
            chunk_size_bins=2,
            bin_width_ms=bin_width_ms,
            rate_threshold_hz=35.0,
            circuit_name="nonlinear_oscillator",
            seed=seed,
        )
        if verbose:
            print(f"  Total Chunks Processed   : {res_stream.total_chunks}")
            print(f"  Mean Processing Latency  : {res_stream.mean_latency_ms:.2f} ms")
            print(f"  p95 Processing Latency   : {res_stream.p95_latency_ms:.2f} ms")
            print(f"  Max Processing Latency   : {res_stream.max_latency_ms:.2f} ms")
            print(f"  Deadline Violations      : {res_stream.deadline_violations}")
            print(f"  Stimulation Pulses       : {res_stream.num_stim_events}")
            print(f"  Artifact Blanked Bins    : {res_stream.num_blanked_bins}")
        if output_svg and res_stream.svg_monitor:
            Path(output_svg).write_text(res_stream.svg_monitor, encoding="utf-8")
            if verbose:
                print(f"  Streaming monitor SVG saved to: {output_svg}")

    elif name == "manifold":
        res_man = run_manifold_geometry_workflow(
            regime_name="nonlinear_oscillator",
            num_conditions=num_trials,
            num_neurons=num_neurons,
            trial_duration_s=duration_s,
            bin_width_ms=bin_width_ms,
            seed=seed,
        )
        if verbose:
            print(f"  Evaluated Conditions     : {res_man.num_conditions}")
            print(f"  Mean Path Length         : {res_man.mean_path_length:.4f}")
            print(f"  Mean Latent Speed        : {res_man.mean_speed:.4f}")
            print(f"  Mean Extrinsic Curvature : {res_man.mean_curvature:.4f}")
            print(f"  Mean Intrinsic Dimension : {res_man.mean_intrinsic_dim:.4f}")
            print(f"  Grassmannian Drift       : {res_man.mean_grassmann_drift:.4f}")
            print(f"  Procrustes Disparity     : {res_man.mean_procrustes_disparity:.4f}")
        if output_svg and res_man.svg_manifold_mesh:
            Path(output_svg).write_text(res_man.svg_manifold_mesh, encoding="utf-8")
            if verbose:
                print(f"  Manifold mesh SVG saved to: {output_svg}")

    elif name == "generalization":
        res_gen = run_generalization_campaign(
            regime_name="driven_recurrent",
            num_conditions=num_trials,
            trials_per_condition=4,
            num_neurons=num_neurons,
            trial_duration_s=duration_s,
            bin_width_ms=bin_width_ms,
            seed=seed,
        )
        if verbose:
            print(f"  Evaluated Conditions     : {res_gen.num_conditions}")
            print(f"  In-Distribution Mean R2  : {res_gen.in_distribution_mean_r2:.4f}")
            print(f"  Out-of-Distribution R2   : {res_gen.out_of_distribution_mean_r2:.4f}")
            print(f"  Generalization Gap       : {res_gen.generalization_gap:.4f}")
            print(f"  Robust Generalizer       : {res_gen.is_robust_generalizer}")

    elif name == "distillation":
        res_dist = run_distillation_workflow(
            num_neurons=num_neurons,
            num_steps=10,
            seed=seed,
        )
        if verbose:
            print(f"  Teacher Parameters       : {res_dist.teacher_params}")
            print(f"  Student Parameters       : {res_dist.student_params}")
            print(f"  Compression Ratio        : {res_dist.parameter_compression_ratio:.2f}x")
            print(f"  Rate Field Correlation   : {res_dist.teacher_student_rate_correlation:.4f}")
            print(f"  Inference Speedup Factor : {res_dist.latency_speedup_factor:.2f}x")

    elif name == "reproducibility":
        res_rep = run_reproducibility_audit(
            seed=seed,
            num_neurons=num_neurons,
            num_steps=5,
        )
        if verbose:
            print(f"  Bitwise Identical Runs   : {res_rep.is_bitwise_identical}")
            print(f"  Max Weight Discrepancy   : {res_rep.max_weight_discrepancy:.2e}")
            print(f"  SHA-256 State Checksum   : {res_rep.sha256_run1}")
            print(f"  Reproducibility Verdict  : {'PASSED' if res_rep.audit_passed else 'FAILED'}")
        if not res_rep.audit_passed:
            return 1

    else:
        if verbose:
            print(f"Error: Unknown workflow '{name}'. Supported: synthetic, streaming, manifold, generalization, distillation, reproducibility")
        return 1

    if verbose:
        print("\nWorkflow execution successfully completed.")
    return 0


def add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    """Register command line arguments for the workflow subcommand."""
    parser.add_argument(
        "--name",
        type=str,
        default="synthetic",
        choices=["synthetic", "streaming", "manifold", "generalization", "distillation", "reproducibility"],
        help="Target automated scientific workflow to execute",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=4,
        help="Number of trials or experimental conditions",
    )
    parser.add_argument(
        "--neurons",
        type=int,
        default=16,
        help="Neural population channel count",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Trajectory duration in seconds",
    )
    parser.add_argument(
        "--bin-width-ms",
        type=int,
        default=20,
        help="Sampling bin width in milliseconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable execution",
    )
    parser.add_argument(
        "--output-svg",
        type=str,
        default=None,
        help="Optional file path to export vector SVG artifacts",
    )
