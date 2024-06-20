"""Adversarial attractor falsification and null-hypothesis surrogate testing.

Falsifies claims of non-trivial limit cycles and multistable attractors against:
1. Ornstein-Uhlenbeck stochastic drift processes (pure continuous diffusion).
2. Overdamped linear spiral sinks (linear dynamics with single trivial origin sink).
3. Phase-randomized Fourier surrogate trajectories (narrowband noise masquerading as limit cycles).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import torch

from nocturne.geometry.fixed_points import FixedPointFinder, FixedPoint
from nocturne.geometry.jacobians import JacobianAnalyzer


@dataclass(frozen=True)
class FalsificationResult:
    """Outcome of adversarial attractor falsification test."""

    regime_name: str
    passed: bool
    spurious_attractor_detected: bool
    detected_fixed_points_count: int
    surrogate_p_value: float
    details: Dict[str, float]


@dataclass(frozen=True)
class FalsificationReport:
    """Consolidated audit across all adversarial null regimes."""

    all_passed: bool
    results: List[FalsificationResult]


class AdversarialAttractorFalsifier:
    """Validates that geometric dynamics analyzers reject spurious attractors on null data."""

    def __init__(
        self,
        dt: float = 0.01,
        num_seeds: int = 25,
        surrogate_shuffles: int = 50,
        random_seed: int = 42,
    ):
        self.dt = dt
        self.num_seeds = num_seeds
        self.surrogate_shuffles = surrogate_shuffles
        self.rng = np.random.RandomState(random_seed)
        self.fp_finder = FixedPointFinder(speed_tol=1e-3, slow_tol=5e-2, max_iters=100)

    def generate_ornstein_uhlenbeck(
        self,
        T: int = 500,
        D: int = 4,
        theta: float = 0.5,
        sigma: float = 0.2,
    ) -> np.ndarray:
        """Simulate mean-reverting Ornstein-Uhlenbeck stochastic drift."""
        traj = np.zeros((T, D), dtype=np.float64)
        for t in range(1, T):
            drift = -theta * traj[t - 1] * self.dt
            diffusion = sigma * np.sqrt(self.dt) * self.rng.randn(D)
            traj[t] = traj[t - 1] + drift + diffusion
        return traj

    def generate_linear_spiral_sink(
        self,
        T: int = 500,
        decay_rate: float = 1.0,
        omega: float = 2.0,
    ) -> np.ndarray:
        """Simulate damped 2D linear spiral decaying strictly into the origin."""
        t = np.linspace(0, T * self.dt, T)
        r = np.exp(-decay_rate * t)
        x = r * np.cos(omega * t)
        y = r * np.sin(omega * t)
        return np.stack([x, y], axis=1)

    def generate_fourier_surrogates(self, trajectory: np.ndarray) -> np.ndarray:
        """Generate phase-randomized Fourier surrogate preserving power spectrum but destroying non-linear coupling."""
        T, D = trajectory.shape
        fft_coeffs = np.fft.rfft(trajectory, axis=0)
        random_phases = self.rng.uniform(0, 2 * np.pi, size=fft_coeffs.shape)
        # Preserve zero-frequency (DC) phase and Nyquist phase for real signal consistency
        random_phases[0] = 0.0
        if T % 2 == 0:
            random_phases[-1] = 0.0
        surrogate_fft = np.abs(fft_coeffs) * np.exp(1j * random_phases)
        surrogate = np.fft.irfft(surrogate_fft, n=T, axis=0)
        return surrogate


    def test_linear_sink_falsification(self) -> FalsificationResult:
        """Verify that a linear sink yields at most 1 trivial fixed point at origin and no limit cycles."""
        A = torch.tensor([[-0.8, -1.5], [1.5, -0.8]], dtype=torch.float32)

        def linear_field(z: torch.Tensor) -> torch.Tensor:
            return torch.matmul(z, A.T)

        seeds = torch.randn(self.num_seeds, 2, dtype=torch.float32)
        fps = self.fp_finder.find_fixed_points(linear_field, seeds)

        # In a linear sink, all candidate seeds should converge to the origin (||z|| < 0.1)
        non_trivial_fps = [fp for fp in fps if np.linalg.norm(fp.location) > 0.15]
        spurious_detected = len(non_trivial_fps) > 0

        passed = not spurious_detected and len(fps) <= 1
        return FalsificationResult(
            regime_name="linear_spiral_sink",
            passed=passed,
            spurious_attractor_detected=spurious_detected,
            detected_fixed_points_count=len(fps),
            surrogate_p_value=1.0,
            details={
                "total_fixed_points": float(len(fps)),
                "non_trivial_count": float(len(non_trivial_fps)),
            },
        )

    def test_stochastic_diffusion_surrogate(self) -> FalsificationResult:
        """Verify phase-randomized surrogate test rejects limit cycle hypothesis on random walk."""
        ou_traj = self.generate_ornstein_uhlenbeck(T=400, D=2)

        # Metric: variance of trajectory radius from centroid (for a true circular limit cycle, this variance is near 0)
        def cycle_circularity(x: np.ndarray) -> float:
            r = np.linalg.norm(x - np.mean(x, axis=0), axis=1)
            return float(np.std(r) / (np.mean(r) + 1e-6))

        base_circularity = cycle_circularity(ou_traj)

        surrogate_circularities = []
        for _ in range(self.surrogate_shuffles):
            surr = self.generate_fourier_surrogates(ou_traj)
            surrogate_circularities.append(cycle_circularity(surr))

        # Empirical p-value that base data is more circular than surrogates
        p_val = float(np.mean([s <= base_circularity for s in surrogate_circularities]))

        # For pure noise, base trajectory is not significantly more circular than linear surrogates (p_val >= 0.05)
        passed = p_val >= 0.05
        return FalsificationResult(
            regime_name="ornstein_uhlenbeck_diffusion",
            passed=passed,
            spurious_attractor_detected=not passed,
            detected_fixed_points_count=0,
            surrogate_p_value=p_val,
            details={
                "base_circularity": base_circularity,
                "surrogate_mean_circularity": float(np.mean(surrogate_circularities)),
            },
        )

    def run_falsification_suite(self) -> FalsificationReport:
        """Run all adversarial null controls."""
        r_linear = self.test_linear_sink_falsification()
        r_diffusion = self.test_stochastic_diffusion_surrogate()

        all_passed = r_linear.passed and r_diffusion.passed
        return FalsificationReport(
            all_passed=all_passed,
            results=[r_linear, r_diffusion],
        )
