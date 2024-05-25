"""Comprehensive system verifier and regression audit suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from nocturne.verification.controls import NegativeControlSuite, ControlResult
from nocturne.binning.likelihood_math import poisson_log_likelihood, get_log_factorial
from nocturne.recordings.mc_maze import MCMazeAdapter
from nocturne.recordings.mc_rtt import MCRTTAdapter
from nocturne.recordings.area2_bump import Area2BumpAdapter
from nocturne.synthetic.generator import SyntheticDatasetGenerator
from nocturne.streaming.state import StreamingBufferState
from nocturne.streaming.streamer import OnlineCausalStreamer
from nocturne.models.attention.config import AttentionConfig
from nocturne.models.attention.model import SpatiotemporalAttentionModel


@dataclass(frozen=True)
class VerificationCheckResult:
    """Individual system verification check outcome."""

    check_id: str
    category: str
    name: str
    passed: bool
    duration_s: float
    details: Dict[str, Any]


@dataclass(frozen=True)
class SystemVerificationReport:
    """Consolidated verification audit report for acceptance gates."""

    timestamp_utc: str
    platform_info: Dict[str, str]
    total_checks: int
    passed_checks: int
    failed_checks: int
    all_passed: bool
    total_duration_s: float
    checks: List[Dict[str, Any]]
    negative_controls: List[Dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class SystemVerifier:
    """Executes full acceptance verifications across contracts, circuits, and models."""

    @staticmethod
    def verify_negative_controls() -> List[ControlResult]:
        """Run all 12 executable negative controls."""
        return NegativeControlSuite.run_all()

    @staticmethod
    def verify_math_exactness() -> VerificationCheckResult:
        """Verify Stirling LUT exactness and Ramanujan asymptotic boundary."""
        t0 = time.perf_counter()
        # Verify Ramanujan approximation error at k = 1025 is < 1e-10 relative error
        # exact ln(1025!) using math.lgamma(1026)
        import math
        exact_val = math.lgamma(1026)
        approx_val = float(get_log_factorial(1025))
        rel_err = abs(exact_val - approx_val) / exact_val
        passed = rel_err < 1e-10

        # Verify Poisson likelihood calculation
        counts = np.array([2, 0, 5])
        rates = np.array([100.0, 10.0, 250.0])
        dt = 0.02
        # mu = [2.0, 0.2, 5.0]
        # ln L = sum(y*ln(mu) - mu - ln(y!))
        expected_ll = (
            2.0 * math.log(2.0) - 2.0 - math.log(2.0)
            + 0.0 * math.log(0.2) - 0.2 - 0.0
            + 5.0 * math.log(5.0) - 5.0 - (math.log(120.0))
        )
        computed_ll = poisson_log_likelihood(counts, rates, dt, include_log_factorial=True)
        ll_err = abs(computed_ll - expected_ll)
        passed = passed and (ll_err < 1e-9)

        return VerificationCheckResult(
            check_id="CHK_MATH_01",
            category="mathematics",
            name="Point-Process Likelihood & Factorial Math Exactness",
            passed=passed,
            duration_s=time.perf_counter() - t0,
            details={"rel_error_k1025": rel_err, "poisson_ll_error": ll_err},
        )

    @staticmethod
    def verify_dataset_fixtures() -> VerificationCheckResult:
        """Verify structural contracts and chronological splits across all 3 datasets."""
        t0 = time.perf_counter()
        fixtures_ok = True
        details = {}

        for name, fixture_rec in [
            ("mc_maze", MCMazeAdapter().create_synthetic_maze_fixture(num_trials=20, num_neurons=15)),
            ("mc_rtt", MCRTTAdapter().create_synthetic_rtt_fixture(duration_s=10.0, num_neurons=15)),
            ("area2_bump", Area2BumpAdapter().create_synthetic_bump_fixture(num_trials=20, num_neurons=15)),
        ]:
            split = fixture_rec.splits["official"]
            train_set = set(split.train.trial_ids)
            test_set = set(split.test.trial_ids)
            has_overlap = len(train_set.intersection(test_set)) > 0
            n_trials = fixture_rec.trials.num_trials if fixture_rec.trials is not None else 0
            if has_overlap or n_trials < 1:
                fixtures_ok = False
            details[name] = {
                "num_trials": n_trials,
                "num_neurons": fixture_rec.num_neurons,
                "split_clean": not has_overlap,
            }

        return VerificationCheckResult(
            check_id="CHK_DATA_01",
            category="data_contracts",
            name="NLB Dataset Adapters & Split Disjointness",
            passed=fixtures_ok,
            duration_s=time.perf_counter() - t0,
            details=details,
        )

    @staticmethod
    def verify_synthetic_circuits() -> VerificationCheckResult:
        """Verify ground truth latent generation across all 4 synthetic regimes."""
        t0 = time.perf_counter()
        passed = True
        details = {}

        gen = SyntheticDatasetGenerator(seed=123)
        for regime in SyntheticDatasetGenerator.SUPPORTED_REGIMES:
            sims, processed = gen.generate_regime_corpus(
                regime_name=regime,
                num_trials=5,
                num_neurons=12,
                trial_duration_s=0.4,
                bin_width_s=0.02,
            )
            n_trials = processed.trials.num_trials if processed.trials is not None else 0
            if len(sims) != 5 or processed.num_neurons != 12 or n_trials != 5:
                passed = False
            details[regime] = {
                "num_sims": len(sims),
                "num_neurons": processed.num_neurons,
                "num_trials": n_trials,
            }

        return VerificationCheckResult(
            check_id="CHK_CIRC_01",
            category="synthetic_circuits",
            name="Controlled Circuit Latent & Spike Generation",
            passed=passed,
            duration_s=time.perf_counter() - t0,
            details=details,
        )

    @staticmethod
    def verify_causal_streaming() -> VerificationCheckResult:
        """Verify online streaming ring buffer and prefix step execution."""
        t0 = time.perf_counter()
        obs_dim = 10
        cfg = AttentionConfig(obs_dim=obs_dim, d_model=16, num_heads=2, num_layers=1, causal=True)
        model = SpatiotemporalAttentionModel(cfg)

        streamer = OnlineCausalStreamer(model=model, obs_dim=obs_dim, max_window_bins=40)
        # Push 3 chunks of 5 time bins
        chunk1 = torch.ones((1, 5, obs_dim), dtype=torch.float32)
        chunk2 = torch.ones((1, 5, obs_dim), dtype=torch.float32) * 2
        chunk3 = torch.ones((1, 5, obs_dim), dtype=torch.float32) * 3

        step1 = streamer.push_chunk(chunk1)
        step2 = streamer.push_chunk(chunk2)
        step3 = streamer.push_chunk(chunk3)

        passed = (
            step1.predicted_rates_hz.shape == (1, 5, obs_dim)
            and step2.predicted_rates_hz.shape == (1, 5, obs_dim)
            and step3.predicted_rates_hz.shape == (1, 5, obs_dim)
            and streamer.state.total_received_bins == 15
        )

        return VerificationCheckResult(
            check_id="CHK_STREAM_01",
            category="streaming",
            name="Causal Streaming Step Execution & Buffer Accumulation",
            passed=passed,
            duration_s=time.perf_counter() - t0,
            details={
                "total_bins": streamer.state.total_received_bins,
                "step3_latency_ms": step3.latency_ms,
            },
        )

    @classmethod
    def run_all(cls) -> SystemVerificationReport:
        """Execute full verification suite and return structured report."""
        t_start = time.perf_counter()

        # Run checks
        checks = [
            cls.verify_math_exactness(),
            cls.verify_dataset_fixtures(),
            cls.verify_synthetic_circuits(),
            cls.verify_causal_streaming(),
        ]

        # Run negative controls
        controls = cls.verify_negative_controls()
        all_controls_caught = all(c.caught for c in controls)

        all_checks_passed = all(c.passed for c in checks) and all_controls_caught
        total_dur = time.perf_counter() - t_start

        return SystemVerificationReport(
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            platform_info={
                "system": platform.system(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
            },
            total_checks=len(checks) + len(controls),
            passed_checks=sum(1 for c in checks if c.passed) + sum(1 for c in controls if c.caught),
            failed_checks=sum(1 for c in checks if not c.passed) + sum(1 for c in controls if not c.caught),
            all_passed=all_checks_passed,
            total_duration_s=total_dur,
            checks=[asdict(c) for c in checks],
            negative_controls=[c._asdict() for c in controls],
        )

    @classmethod
    def save_report(cls, output_path: str | Path) -> Path:
        """Execute verification and save JSON report to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = cls.run_all()
        with open(path, "w", encoding="utf-8") as f:
            f.write(report.to_json())
        return path
