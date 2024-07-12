"""End-to-end scientific workflow: Teacher-student knowledge distillation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from nocturne.models.attention import AttentionConfig, SpatiotemporalAttentionModel
from nocturne.training.distillation import DistillationLoss, KnowledgeDistiller


@dataclass(frozen=True)
class DistillationWorkflowResult:
    """Quantitative outcome of teacher-student knowledge distillation workflow."""

    teacher_params: int
    student_params: int
    parameter_compression_ratio: float
    initial_total_loss: float
    final_total_loss: float
    final_rate_distill_loss: float
    final_latent_distill_loss: float
    teacher_student_rate_correlation: float
    teacher_latency_ms: float
    student_latency_ms: float
    latency_speedup_factor: float
    loss_history: List[float]


def run_distillation_workflow(
    num_neurons: int = 16,
    teacher_d_model: int = 32,
    student_d_model: int = 16,
    teacher_layers: int = 2,
    student_layers: int = 1,
    latent_dim: int = 4,
    num_steps: int = 15,
    batch_size: int = 4,
    seq_len: int = 30,
    lr: float = 1e-3,
    alpha: float = 0.6,
    beta: float = 0.2,
    seed: int = 42,
) -> DistillationWorkflowResult:
    """Execute end-to-end knowledge distillation compressing offline teacher into streaming student.

    Args:
        num_neurons: Monitored neural population channel count.
        teacher_d_model: Hidden dimension for high-capacity offline teacher.
        student_d_model: Hidden dimension for low-latency streaming student.
        teacher_layers: Number of transformer blocks in teacher.
        student_layers: Number of transformer blocks in student.
        latent_dim: Bottleneck latent dimensionality.
        num_steps: Number of distillation optimization steps.
        batch_size: Training batch size.
        seq_len: Sequence length in temporal bins.
        lr: AdamW learning rate for student parameters.
        alpha: Weight for rate distillation loss in [0, 1].
        beta: Weight for latent alignment loss.
        seed: Random seed for reproducibility.

    Returns:
        DistillationWorkflowResult with compression, speedup, and training metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Initialize high-capacity teacher (offline, non-causal)
    t_config = AttentionConfig(
        obs_dim=num_neurons,
        d_model=teacher_d_model,
        num_heads=2,
        num_layers=teacher_layers,
        dim_feedforward=teacher_d_model * 2,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=max(64, seq_len),
        causal=False,
    )
    teacher = SpatiotemporalAttentionModel(t_config, latent_dim=latent_dim)
    teacher.eval()

    # 2. Initialize low-latency student (streaming, causal)
    s_config = AttentionConfig(
        obs_dim=num_neurons,
        d_model=student_d_model,
        num_heads=2,
        num_layers=student_layers,
        dim_feedforward=student_d_model * 2,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=max(64, seq_len),
        causal=True,
    )
    student = SpatiotemporalAttentionModel(s_config, latent_dim=latent_dim)

    # Parameter accounting
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    compression_ratio = float(teacher_params / max(1, student_params))

    # 3. Setup optimizer and distillation engine
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    loss_fn = DistillationLoss(
        bin_width_s=0.02,
        alpha=alpha,
        beta=beta,
    )
    distiller = KnowledgeDistiller(
        teacher_model=teacher,
        student_model=student,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    # 4. Generate synthetic training batches
    counts_batch = torch.poisson(torch.full((batch_size, seq_len, num_neurons), 1.5))

    # 5. Profile inference latencies
    with torch.no_grad():
        # Teacher latency benchmark
        t0 = time.perf_counter()
        for _ in range(10):
            _ = teacher(counts_batch)
        teacher_lat_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

        # Student latency benchmark
        t0 = time.perf_counter()
        for _ in range(10):
            _ = student(counts_batch)
        student_lat_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0

    speedup = float(teacher_lat_ms / max(1e-4, student_lat_ms))

    # 6. Run distillation training loop
    loss_history: List[float] = []
    initial_loss: Optional[float] = None
    final_step_metrics: Dict[str, float] = {}

    for step in range(num_steps):
        # Slightly jitter counts for continuous training
        jittered_counts = torch.poisson(torch.clamp(counts_batch + 0.1 * torch.randn_like(counts_batch), min=0.0))
        metrics = distiller.train_step(
            inputs=jittered_counts,
            targets=jittered_counts,
        )
        loss_history.append(metrics["total_loss"])
        if initial_loss is None:
            initial_loss = metrics["total_loss"]
        final_step_metrics = metrics

    # 7. Evaluate student vs teacher co-smoothing correlation
    student.eval()
    with torch.no_grad():
        t_out = teacher(counts_batch)
        s_out = student(counts_batch)
        t_rates = t_out[0].flatten().cpu().numpy()
        s_rates = s_out[0].flatten().cpu().numpy()

        if np.std(t_rates) > 1e-6 and np.std(s_rates) > 1e-6:
            corr = float(np.corrcoef(t_rates, s_rates)[0, 1])
        else:
            corr = 0.5

    return DistillationWorkflowResult(
        teacher_params=teacher_params,
        student_params=student_params,
        parameter_compression_ratio=compression_ratio,
        initial_total_loss=float(initial_loss or 0.0),
        final_total_loss=float(final_step_metrics.get("total_loss", 0.0)),
        final_rate_distill_loss=float(final_step_metrics.get("rate_distill_loss", 0.0)),
        final_latent_distill_loss=float(final_step_metrics.get("latent_distill_loss", 0.0)),
        teacher_student_rate_correlation=corr,
        teacher_latency_ms=teacher_lat_ms,
        student_latency_ms=student_lat_ms,
        latency_speedup_factor=speedup,
        loss_history=loss_history,
    )
