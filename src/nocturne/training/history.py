"""Structured training and validation metric history tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class TrainingHistory:
    """Tracks epoch-level metrics and serializes training curves to disk."""

    def __init__(self):
        self.epochs: List[int] = []
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.learning_rates: List[float] = []
        self.grad_norms: List[float] = []
        self.extra_metrics: List[Dict[str, float]] = []

    def record(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        grad_norm: float,
        extra: Dict[str, float] | None = None,
    ) -> None:
        """Record metrics for an epoch.

        Args:
            epoch: Epoch index.
            train_loss: Mean training loss.
            val_loss: Validation loss.
            lr: Current learning rate.
            grad_norm: Gradient norm before clipping.
            extra: Additional diagnostic metrics.
        """
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.learning_rates.append(lr)
        self.grad_norms.append(grad_norm)
        self.extra_metrics.append(extra or {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to serializable dictionary."""
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "learning_rates": self.learning_rates,
            "grad_norms": self.grad_norms,
            "extra_metrics": self.extra_metrics,
        }

    def save(self, file_path: str | Path) -> None:
        """Save history to JSON file.

        Args:
            file_path: Output JSON path.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str | Path) -> TrainingHistory:
        """Load history from JSON file.

        Args:
            file_path: Input JSON path.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        hist = cls()
        hist.epochs = data["epochs"]
        hist.train_loss = data["train_loss"]
        hist.val_loss = data["val_loss"]
        hist.learning_rates = data["learning_rates"]
        hist.grad_norms = data["grad_norms"]
        hist.extra_metrics = data.get("extra_metrics", [])
        return hist
