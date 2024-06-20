"""Teacher-student knowledge distillation for causal neural dynamics models.

Enables compressing high-capacity offline bidirectional models (axial attention,
bidirectional VLDM) into low-latency causal streaming architectures by transferring
both smoothed rate fields (I-divergence / Poisson relative entropy) and latent geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from nocturne.models.likelihoods import PoissonLoss


@dataclass(frozen=True)
class DistillationLossOutput:
    """Decomposition of knowledge distillation loss components."""

    total_loss: torch.Tensor
    task_loss: torch.Tensor
    rate_distill_loss: torch.Tensor
    latent_distill_loss: torch.Tensor


class DistillationLoss(nn.Module):
    """Computes hybrid task and representation distillation objectives."""

    def __init__(
        self,
        bin_width_s: float = 0.02,
        alpha: float = 0.5,
        beta: float = 0.1,
        temperature: float = 1.0,
        eps: float = 1e-7,
    ):
        """Initialize distillation objective.

        Args:
            bin_width_s: Temporal bin width in seconds.
            alpha: Weight for rate distillation loss in [0, 1]. Task loss weight is (1 - alpha).
            beta: Weight for latent space alignment loss.
            temperature: Softening temperature for rates.
            eps: Epsilon floor for numerical stability.
        """
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")

        self.bin_width_s = bin_width_s
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.eps = eps
        self.task_loss_fn = PoissonLoss(bin_width_s=bin_width_s, eps=eps)


    def compute_rate_divergence(
        self,
        student_rates: torch.Tensor,
        teacher_rates: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generalized I-divergence / Poisson relative entropy between teacher and student.

        D(p || q) = sum(p * log(p / q) - (p - q))
        """
        t_rates = torch.clamp(teacher_rates / self.temperature, min=self.eps)
        s_rates = torch.clamp(student_rates / self.temperature, min=self.eps)

        divergence = t_rates * (torch.log(t_rates) - torch.log(s_rates)) - (t_rates - s_rates)

        if mask is not None:
            divergence = divergence * mask.float()
            num_valid = torch.clamp(mask.float().sum(), min=1.0)
            return divergence.sum() / num_valid

        return divergence.mean()

    def forward(
        self,
        student_rates: torch.Tensor,
        teacher_rates: torch.Tensor,
        targets: torch.Tensor,
        student_latents: Optional[torch.Tensor] = None,
        teacher_latents: Optional[torch.Tensor] = None,
        latent_adapter: Optional[nn.Module] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> DistillationLossOutput:
        """Compute distillation loss."""
        # 1. Primary ground-truth Poisson likelihood loss
        task_loss = self.task_loss_fn(student_rates, targets, mask=mask)

        # 2. Rate transfer loss (detached teacher)
        rate_loss = self.compute_rate_divergence(
            student_rates=student_rates,
            teacher_rates=teacher_rates.detach(),
            mask=mask,
        )

        # 3. Latent trajectory geometry alignment
        if student_latents is not None and teacher_latents is not None:
            t_lat = teacher_latents.detach()
            if latent_adapter is not None:
                s_lat_proj = latent_adapter(student_latents)
            else:
                s_lat_proj = student_latents

            if s_lat_proj.shape != t_lat.shape:
                raise ValueError(
                    f"Latent shape mismatch: student projected {s_lat_proj.shape} vs teacher {t_lat.shape}. "
                    "Provide a latent_adapter nn.Module to project dimensions."
                )

            latent_loss = F.mse_loss(s_lat_proj, t_lat)
        else:
            latent_loss = torch.tensor(0.0, device=student_rates.device, dtype=student_rates.dtype)

        total_loss = (1.0 - self.alpha) * task_loss + self.alpha * rate_loss + self.beta * latent_loss

        return DistillationLossOutput(
            total_loss=total_loss,
            task_loss=task_loss,
            rate_distill_loss=rate_loss,
            latent_distill_loss=latent_loss,
        )


class KnowledgeDistiller:
    """Orchestrates teacher forward inference and student parameter optimization."""

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[DistillationLoss] = None,
        latent_adapter: Optional[nn.Module] = None,
    ):
        self.teacher_model = teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.student_model = student_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn is not None else DistillationLoss()
        self.latent_adapter = latent_adapter

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Execute a single distillation optimization step."""
        self.student_model.train()
        self.optimizer.zero_grad()

        with torch.no_grad():
            teacher_out = self.teacher_model(inputs)
            if isinstance(teacher_out, tuple):
                t_rates = teacher_out[0]
                t_latents = teacher_out[1] if len(teacher_out) > 1 else None
            elif hasattr(teacher_out, "rates"):
                t_rates = teacher_out.rates
                t_latents = getattr(teacher_out, "latents", None)
            else:
                t_rates = teacher_out
                t_latents = None

        student_out = self.student_model(inputs)
        if isinstance(student_out, tuple):
            s_rates = student_out[0]
            s_latents = student_out[1] if len(student_out) > 1 else None
        elif hasattr(student_out, "rates"):
            s_rates = student_out.rates
            s_latents = getattr(student_out, "latents", None)
        else:
            s_rates = student_out
            s_latents = None

        loss_out = self.loss_fn(
            student_rates=s_rates,
            teacher_rates=t_rates,
            targets=targets,
            student_latents=s_latents,
            teacher_latents=t_latents,
            latent_adapter=self.latent_adapter,
            mask=mask,
        )

        loss_out.total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": float(loss_out.total_loss.item()),
            "task_loss": float(loss_out.task_loss.item()),
            "rate_distill_loss": float(loss_out.rate_distill_loss.item()),
            "latent_distill_loss": float(loss_out.latent_distill_loss.item()),
        }
