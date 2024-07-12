"""End-to-end scientific workflow: Deterministic bitwise reproducibility and numerical audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from nocturne.models.attention import AttentionConfig, SpatiotemporalAttentionModel
from nocturne.synthetic.generator import SyntheticDatasetGenerator


@dataclass(frozen=True)
class ReproducibilityAuditResult:
    """Quantitative outcome of cross-run determinism and bitwise reproducibility verification."""

    seed: int
    num_steps: int
    is_bitwise_identical: bool
    max_weight_discrepancy: float
    max_loss_discrepancy: float
    sha256_run1: str
    sha256_run2: str
    cross_seed_divergence: float
    is_non_trivial: bool
    audit_passed: bool
    summary_report: str


def _compute_state_hash(model: nn.Module) -> str:
    """Compute deterministic SHA-256 checksum of model state dictionary."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _run_single_deterministic_trial(
    seed: int,
    num_neurons: int,
    latent_dim: int,
    num_steps: int,
    seq_len: int,
) -> tuple[nn.Module, list[float]]:
    """Execute single training run with strict deterministic seed controls."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = AttentionConfig(
        obs_dim=num_neurons,
        d_model=16,
        num_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=max(64, seq_len),
        causal=True,
    )
    model = SpatiotemporalAttentionModel(config, latent_dim=latent_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    counts = torch.poisson(torch.full((2, seq_len, num_neurons), 1.2))
    losses: list[float] = []

    for _ in range(num_steps):
        optimizer.zero_grad()
        rates, latents, _, _ = model(counts)
        # Poisson deviance objective
        loss = torch.mean(rates * 0.02 - counts * torch.log(torch.clamp(rates * 0.02, min=1e-7)))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return model, losses


def run_reproducibility_audit(
    seed: int = 42,
    num_neurons: int = 8,
    latent_dim: int = 2,
    num_steps: int = 5,
    seq_len: int = 16,
) -> ReproducibilityAuditResult:
    """Execute reproducibility audit verifying bitwise identical state replication.

    Args:
        seed: Canonical random seed evaluated across repeated executions.
        num_neurons: Neural population dimension.
        latent_dim: Bottleneck latent dimension.
        num_steps: Number of forward-backward optimization steps.
        seq_len: Sequence length in temporal bins.

    Returns:
        ReproducibilityAuditResult containing discrepancy tolerances, hashes, and pass verdict.
    """
    # 1. Run 1 with canonical seed
    model_1, losses_1 = _run_single_deterministic_trial(
        seed=seed,
        num_neurons=num_neurons,
        latent_dim=latent_dim,
        num_steps=num_steps,
        seq_len=seq_len,
    )
    hash_1 = _compute_state_hash(model_1)

    # 2. Run 2 with identical canonical seed
    model_2, losses_2 = _run_single_deterministic_trial(
        seed=seed,
        num_neurons=num_neurons,
        latent_dim=latent_dim,
        num_steps=num_steps,
        seq_len=seq_len,
    )
    hash_2 = _compute_state_hash(model_2)

    # 3. Control Run with different seed (verifying non-trivial variance)
    model_ctrl, _ = _run_single_deterministic_trial(
        seed=seed + 999,
        num_neurons=num_neurons,
        latent_dim=latent_dim,
        num_steps=num_steps,
        seq_len=seq_len,
    )

    # 4. Compare weights between Run 1 and Run 2
    max_weight_diff = 0.0
    for (name1, p1), (name2, p2) in zip(model_1.named_parameters(), model_2.named_parameters()):
        diff = float(torch.max(torch.abs(p1 - p2)).item())
        if diff > max_weight_diff:
            max_weight_diff = diff

    # Compare losses
    loss_diffs = [abs(l1 - l2) for l1, l2 in zip(losses_1, losses_2)]
    max_loss_diff = max(loss_diffs) if loss_diffs else 0.0

    # Cross-seed divergence
    ctrl_diffs = [
        float(torch.norm(p1 - pctrl).item())
        for (name1, p1), (_, pctrl) in zip(model_1.named_parameters(), model_ctrl.named_parameters())
    ]
    cross_seed_div = float(np.mean(ctrl_diffs))

    is_bitwise = (hash_1 == hash_2) and (max_weight_diff < 1e-7) and (max_loss_diff < 1e-7)
    is_non_trivial = cross_seed_div > 1e-3
    passed = is_bitwise and is_non_trivial

    report_lines = [
        "=== Nocturne Reproducibility & Numerical Determinism Audit ===",
        f"Evaluated Seed                : {seed}",
        f"Optimization Steps            : {num_steps}",
        f"Run 1 SHA-256 Checksum        : {hash_1}",
        f"Run 2 SHA-256 Checksum        : {hash_2}",
        f"Max Parameter Discrepancy     : {max_weight_diff:.2e}",
        f"Max Loss Discrepancy          : {max_loss_diff:.2e}",
        f"Cross-Seed Divergence Norm    : {cross_seed_div:.4f}",
        f"Bitwise Identity Verified     : {is_bitwise}",
        f"Non-Trivial Dynamics Verified : {is_non_trivial}",
        f"Final Audit Verdict           : {'PASSED' if passed else 'FAILED'}",
    ]
    summary_text = "\n".join(report_lines)

    return ReproducibilityAuditResult(
        seed=seed,
        num_steps=num_steps,
        is_bitwise_identical=is_bitwise,
        max_weight_discrepancy=max_weight_diff,
        max_loss_discrepancy=max_loss_diff,
        sha256_run1=hash_1,
        sha256_run2=hash_2,
        cross_seed_divergence=cross_seed_div,
        is_non_trivial=is_non_trivial,
        audit_passed=passed,
        summary_report=summary_text,
    )
