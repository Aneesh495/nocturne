"""Training engine, batch collation, optimization routines, and checkpoint recovery."""

from __future__ import annotations

from nocturne.training.config import TrainingConfig
from nocturne.training.collator import NeuralBatch, NeuralBatchCollator
from nocturne.training.optimizer import (
    create_optimizer,
    create_lr_scheduler,
    clip_gradients,
)
from nocturne.training.checkpoint import (
    CheckpointManager,
    capture_rng_states,
    restore_rng_states,
)
from nocturne.training.history import TrainingHistory
from nocturne.training.recovery import TrainingRecovery
from nocturne.training.trainer import NeuralDynamicsTrainer
from nocturne.training.curriculum import CurriculumScheduler, CurriculumState
from nocturne.training.distillation import (
    DistillationLoss,
    DistillationLossOutput,
    KnowledgeDistiller,
)

__all__ = [
    "TrainingConfig",
    "NeuralBatch",
    "NeuralBatchCollator",
    "create_optimizer",
    "create_lr_scheduler",
    "clip_gradients",
    "CheckpointManager",
    "capture_rng_states",
    "restore_rng_states",
    "TrainingHistory",
    "TrainingRecovery",
    "NeuralDynamicsTrainer",
    "CurriculumScheduler",
    "CurriculumState",
    "DistillationLoss",
    "DistillationLossOutput",
    "KnowledgeDistiller",
]


