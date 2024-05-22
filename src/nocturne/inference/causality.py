"""Causal prefix-invariance and future-data leakage verification engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class CausalAuditReport:
    """Summary report of causal prefix-invariance perturbation testing."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    max_rate_deviation: float
    max_latent_deviation: float
    is_strictly_causal: bool
    violations: List[Dict[str, Any]]


class CausalLeakageAuditor:
    """Rigorous perturbation test suite verifying zero information flow from future to past."""

    def __init__(self, rate_tolerance: float = 5e-5, latent_tolerance: float = 1e-5):
        self.rate_tolerance = rate_tolerance
        self.latent_tolerance = latent_tolerance

    def audit_model(
        self,
        model: nn.Module,
        num_cases: int = 1000,
        batch_size: int = 20,
        seq_len: int = 30,
        obs_dim: int = 16,
        seed: int = 42,
    ) -> CausalAuditReport:
        """Run perturbation test cases to verify causal prefix-invariance.

        For each case:
        1. Select random cutoff t* in [1, T - 2].
        2. Perturb future bins t' > t* with heavy spike injections or zeroing.
        3. Verify that predictions at t <= t* remain numerically identical.

        Args:
            model: PyTorch model supporting causal execution.
            num_cases: Number of distinct perturbation cases to evaluate (must be >= 1000 for BA07).
            batch_size: Batch size for vectorized evaluation.
            seq_len: Temporal sequence length T.
            obs_dim: Number of neural channels N.
            seed: Random seed for deterministic perturbation generation.

        Returns:
            CausalAuditReport detailing maximum observed deviations and violation log.
        """
        model.eval()
        torch.manual_seed(seed)
        np.random.seed(seed)

        passed = 0
        violations: List[Dict[str, Any]] = []
        max_rate_dev = 0.0
        max_latent_dev = 0.0

        remaining = num_cases
        case_idx = 0

        while remaining > 0:
            B = min(remaining, batch_size)
            T = seq_len
            N = obs_dim

            # Base counts
            base_counts = torch.randint(0, 5, (B, T, N), dtype=torch.float32)

            with torch.no_grad():
                rates_base, latents_base, _, _ = model(base_counts, causal=True)

            for b in range(B):
                # Pick random cutoff t*
                t_cutoff = np.random.randint(1, T - 1)

                # Clone and perturb future
                perturbed = base_counts[b : b + 1].clone()
                perturb_type = case_idx % 4

                if perturb_type == 0:
                    # Large additive spike burst
                    perturbed[:, t_cutoff:, :] += 20.0
                elif perturb_type == 1:
                    # Complete future silence
                    perturbed[:, t_cutoff:, :] = 0.0
                elif perturb_type == 2:
                    # Random noise injection
                    perturbed[:, t_cutoff:, :] = torch.randint(0, 25, (1, T - t_cutoff, N), dtype=torch.float32)
                else:
                    # High frequency alternating pattern
                    alt = torch.zeros((1, T - t_cutoff, N), dtype=torch.float32)
                    alt[:, ::2, :] = 15.0
                    perturbed[:, t_cutoff:, :] = alt

                with torch.no_grad():
                    rates_pert, latents_pert, _, _ = model(perturbed, causal=True)

                # Compare past and present predictions for t <= t_cutoff
                rate_diff = (
                    rates_base[b : b + 1, :t_cutoff, :] - rates_pert[:, :t_cutoff, :]
                ).abs().max().item()

                latent_diff = (
                    latents_base[b : b + 1, :t_cutoff, :] - latents_pert[:, :t_cutoff, :]
                ).abs().max().item()

                if rate_diff > max_rate_dev:
                    max_rate_dev = rate_diff
                if latent_diff > max_latent_dev:
                    max_latent_dev = latent_diff

                if rate_diff <= self.rate_tolerance and latent_diff <= self.latent_tolerance:
                    passed += 1
                else:
                    violations.append({
                        "case_id": case_idx,
                        "t_cutoff": int(t_cutoff),
                        "rate_deviation": float(rate_diff),
                        "latent_deviation": float(latent_diff),
                    })

                case_idx += 1

            remaining -= B

        return CausalAuditReport(
            total_cases=num_cases,
            passed_cases=passed,
            failed_cases=num_cases - passed,
            max_rate_deviation=max_rate_dev,
            max_latent_deviation=max_latent_dev,
            is_strictly_causal=(len(violations) == 0),
            violations=violations,
        )
