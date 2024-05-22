"""Optimization routines, parameter grouping, learning rate schedules, and gradient clipping."""

from __future__ import annotations

import math
from typing import Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from nocturne.training.config import TrainingConfig


def create_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    """Create AdamW optimizer with decoupled weight decay for 1D parameters.

    Excludes 1D tensors (biases, normalization scale/bias) from weight decay.

    Args:
        model: PyTorch model to optimize.
        config: Training configuration.

    Returns:
        Configured AdamW optimizer.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    total_steps: int,
) -> LambdaLR:
    """Create learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: PyTorch optimizer.
        config: Training configuration.
        total_steps: Total number of training steps across all epochs.

    Returns:
        LambdaLR scheduler.
    """
    warmup_steps = config.warmup_steps
    min_ratio = config.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip parameter gradients by L2 norm.

    Args:
        model: PyTorch model.
        max_norm: Maximum gradient norm threshold.

    Returns:
        Total computed gradient norm before clipping.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=2.0,
    )
    return float(total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm)
