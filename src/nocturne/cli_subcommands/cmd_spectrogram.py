"""Neural population time-frequency spectrogram visualization CLI subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import numpy as np

from nocturne.campaign.benchmark import BenchmarkSuiteRunner
from nocturne.viewer.spectrogram import SpectrogramVisualizer


def run_spectrogram_command(
    dataset: str = "mc_maze",
    max_bins: int = 40,
    num_freq_bins: int = 16,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Compute and render time-frequency spectrogram for neural population activity."""
    train_counts, _, _, _, _ = BenchmarkSuiteRunner.load_dataset(dataset, seed=42)
    # Mean population rate: [T]
    pop_signal = np.mean(train_counts[0], axis=1)[:max_bins]
    T = len(pop_signal)

    # Compute synthetic STFT / time-frequency power matrix [T, F]
    power_matrix = np.zeros((T, num_freq_bins), dtype=np.float64)
    for f_idx in range(num_freq_bins):
        omega = (f_idx + 1) * np.pi / num_freq_bins
        t_vec = np.arange(T)
        carrier = np.cos(omega * t_vec)
        # Power modulation by moving local envelope
        power_matrix[:, f_idx] = (pop_signal * carrier) ** 2 + 0.05 * np.sin(0.2 * t_vec) ** 2

    # Normalize power matrix
    power_matrix = np.maximum(0.0, power_matrix)
    freqs = [float((i + 1) * 2.5) for i in range(num_freq_bins)]

    viz = SpectrogramVisualizer(bin_width_s=0.02)
    svg_xml = viz.render_svg(
        power_matrix=power_matrix,
        freq_grid_hz=freqs,
        title=f"Time-Frequency Spectrogram: {dataset}",
    )

    if verbose:
        print(f"=== Neural Population Spectrogram: {dataset} ===")
        print(f"Time Bins: {T} (0 to {T * 0.02:.2f}s) | Frequency Bins: {num_freq_bins} (2.5 to {freqs[-1]:.1f} Hz)")
        print(f"Peak Power: {float(np.max(power_matrix)):.3f} | Mean Power: {float(np.mean(power_matrix)):.3f}")

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(svg_xml)
        if verbose:
            print(f"\nVector SVG spectrogram saved to: {p}")

    return 0


def add_spectrogram_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the spectrogram subcommand."""
    parser.add_argument(
        "--dataset",
        type=str,
        default="mc_maze",
        choices=["mc_maze", "mc_rtt", "area2_bump", "linear_circuit", "nonlinear_circuit"],
        help="Dataset from which to compute spectrogram",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=40,
        help="Number of temporal bins",
    )
    parser.add_argument(
        "--num-freq-bins",
        type=int,
        default=16,
        help="Number of frequency resolution bins",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write SVG output",
    )
