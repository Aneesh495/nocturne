"""Atomic checkpoint management and complete state serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


def capture_rng_states() -> Dict[str, Any]:
    """Capture random number generator states across Python, NumPy, and PyTorch.

    Returns:
        Dictionary containing serialized RNG states.
    """
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: Dict[str, Any]) -> None:
    """Restore random number generator states from a captured snapshot.

    Args:
        states: Dictionary previously captured via capture_rng_states.
    """
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


class CheckpointManager:
    """Manages atomic checkpoint saving, loading, retention pruning, and recovery."""

    def __init__(self, checkpoint_dir: str | Path, keep_top_k: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self._checkpoints: List[Tuple[float, Path]] = []  # (val_loss, path)

    def save(
        self,
        epoch: int,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> Path:
        """Atomically save checkpoint to disk using a temporary file and atomic replace.

        Args:
            epoch: Current training epoch index.
            step: Global optimization step index.
            model: PyTorch model.
            optimizer: Optimizer instance.
            scheduler: Optional learning rate scheduler.
            metrics: Validation and training metrics dictionary.
            is_best: Whether this checkpoint achieved the best validation score.

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}_step_{step:06d}.pt"
        temp_path = self.checkpoint_dir / f"tmp_checkpoint_{os.getpid()}_{step}.pt"

        state = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "rng_states": capture_rng_states(),
        }

        # Step 1: Write to temporary file
        torch.save(state, temp_path)

        # Step 2: Atomic rename
        os.replace(temp_path, checkpoint_path)

        # Step 3: Update latest symlink/copy
        latest_path = self.checkpoint_dir / "latest.pt"
        temp_latest = self.checkpoint_dir / f"tmp_latest_{os.getpid()}.pt"
        torch.save(state, temp_latest)
        os.replace(temp_latest, latest_path)

        # Step 4: If best, save best copy
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            temp_best = self.checkpoint_dir / f"tmp_best_{os.getpid()}.pt"
            torch.save(state, temp_best)
            os.replace(temp_best, best_path)

        # Step 5: Prune old checkpoints
        val_loss = metrics.get("val_loss", float("inf"))
        self._checkpoints.append((val_loss, checkpoint_path))
        self._checkpoints.sort(key=lambda x: x[0])  # Sort ascending by loss

        if len(self._checkpoints) > self.keep_top_k:
            # Remove worst checkpoints exceeding retention limit
            surplus = self._checkpoints[self.keep_top_k :]
            self._checkpoints = self._checkpoints[: self.keep_top_k]
            for _, path_to_remove in surplus:
                if path_to_remove.exists() and path_to_remove != checkpoint_path:
                    try:
                        path_to_remove.unlink()
                    except OSError:
                        pass

        return checkpoint_path

    def load(
        self,
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        """Load state from checkpoint into model, optimizer, scheduler, and RNGs.

        Args:
            checkpoint_path: Path to checkpoint file.
            model: PyTorch model to populate.
            optimizer: Optional optimizer to restore.
            scheduler: Optional scheduler to restore.
            restore_rng: Whether to restore random number generator states.

        Returns:
            Checkpoint dictionary containing epoch, step, metrics, and metadata.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if restore_rng and checkpoint.get("rng_states") is not None:
            restore_rng_states(checkpoint["rng_states"])

        return checkpoint
