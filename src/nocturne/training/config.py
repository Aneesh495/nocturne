"""Configuration schema for neural population dynamics training pipelines."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    """Hyperparameters and execution configuration for model optimization."""

    model_config = ConfigDict(frozen=True)

    # Optimization parameters
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Peak learning rate")
    weight_decay: float = Field(default=1e-4, ge=0.0, description="Decoupled weight decay parameter")
    clip_grad_norm: float = Field(default=1.0, gt=0.0, description="Maximum L2 gradient norm for clipping")
    warmup_steps: int = Field(default=100, ge=0, description="Number of linear warmup optimization steps")
    min_lr_ratio: float = Field(default=0.01, gt=0.0, le=1.0, description="Ratio of minimum lr to peak lr")

    # Training schedule
    max_epochs: int = Field(default=50, ge=1, description="Maximum training epochs")
    patience: int = Field(default=10, ge=1, description="Early stopping patience in epochs")
    batch_size: int = Field(default=32, ge=1, description="Mini-batch size")
    eval_every_epochs: int = Field(default=1, ge=1, description="Validation frequency in epochs")

    # Masked objective configuration
    mask_rate: float = Field(default=0.15, ge=0.0, le=0.9, description="Fraction of tokens/channels masked during training")
    mask_strategy: Literal["token", "channel", "temporal_tail"] = Field(
        default="token",
        description="Masking pattern for self-supervised training",
    )

    # Hardware & reproducibility
    device: str = Field(default="auto", description="Execution device ('auto', 'cpu', 'mps', 'cuda')")
    seed: int = Field(default=42, description="Global random seed for reproducibility")
    checkpoint_dir: str = Field(default="checkpoints", description="Directory for saving atomic checkpoints")
    keep_top_k: int = Field(default=3, ge=1, description="Number of best checkpoints to retain")
