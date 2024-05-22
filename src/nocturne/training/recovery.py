"""Checkpoint integrity validation and deterministic resumption verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn

from nocturne.training.checkpoint import CheckpointManager


class TrainingRecovery:
    """Verifies checkpoint completeness, finite weights, and deterministic recovery."""

    REQUIRED_KEYS = (
        "epoch",
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "metrics",
        "rng_states",
    )

    @classmethod
    def verify_checkpoint_integrity(cls, checkpoint_path: str | Path) -> Dict[str, Any]:
        """Validate that a checkpoint file is uncorrupted and contains all required state.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Loaded checkpoint dictionary.

        Raises:
            ValueError: If any key is missing or model parameters contain NaNs/Infs.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        for key in cls.REQUIRED_KEYS:
            if key not in checkpoint:
                raise ValueError(f"Corrupt checkpoint {path}: missing required key '{key}'")

        # Verify all parameters in state dict are finite
        state_dict = checkpoint["model_state_dict"]
        for param_name, tensor in state_dict.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Corrupt checkpoint {path}: non-finite values detected in parameter '{param_name}'"
                )

        return checkpoint

    @classmethod
    def resume_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> Tuple[int, int, Dict[str, float]]:
        """Safely restore model, optimizer, scheduler, and RNG state from verified checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            model: PyTorch model.
            optimizer: Optimizer to restore.
            scheduler: Optional learning rate scheduler.

        Returns:
            Tuple of (resumed_epoch, resumed_step, metrics).
        """
        checkpoint = cls.verify_checkpoint_integrity(checkpoint_path)
        manager = CheckpointManager(Path(checkpoint_path).parent)
        data = manager.load(
            checkpoint_path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,
        )
        return data["epoch"], data["step"], data["metrics"]
