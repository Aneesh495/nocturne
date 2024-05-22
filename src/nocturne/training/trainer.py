"""Training coordinator and validation loop for neural population models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from nocturne.training.config import TrainingConfig
from nocturne.training.collator import NeuralBatch, NeuralBatchCollator
from nocturne.training.optimizer import create_optimizer, create_lr_scheduler, clip_gradients
from nocturne.training.checkpoint import CheckpointManager
from nocturne.training.history import TrainingHistory
from nocturne.binning.likelihood_math import poisson_bits_per_spike

logger = logging.getLogger(__name__)


class TrialsDataset(Dataset):
    """Simple dataset wrapping a list of (counts, kinematics) pairs."""

    def __init__(self, trials: List[Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]]):
        self.trials = trials

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int):
        return self.trials[idx]


def resolve_device(device_pref: str) -> torch.device:
    """Resolve target torch device based on preference and hardware availability."""
    if device_pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_pref)


class NeuralDynamicsTrainer:
    """Coordinated training loop with early stopping, checkpointing, and metric logging."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_data: List[Tuple[Any, Any]],
        val_data: List[Tuple[Any, Any]],
        bin_width_s: float = 0.02,
    ):
        self.model = model
        self.config = config
        self.bin_width_s = bin_width_s

        # Hardware dispatch
        self.device = resolve_device(config.device)
        self.model.to(self.device)

        # Datasets & Loaders
        self.collator = NeuralBatchCollator(
            mask_rate=config.mask_rate,
            mask_strategy=config.mask_strategy,
        )
        self.val_collator = NeuralBatchCollator(
            mask_rate=config.mask_rate,
            mask_strategy=config.mask_strategy,
        )

        self.train_loader = DataLoader(
            TrialsDataset(train_data),
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )
        self.val_loader = DataLoader(
            TrialsDataset(val_data),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=self.val_collator,
        )

        # Optimization
        total_steps = len(self.train_loader) * config.max_epochs
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_lr_scheduler(self.optimizer, config, total_steps)

        # Management
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            keep_top_k=config.keep_top_k,
        )
        self.history = TrainingHistory()
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def _train_epoch(self) -> Tuple[float, float, float]:
        """Execute one training epoch.

        Returns:
            Tuple of (mean_loss, mean_grad_norm, final_lr).
        """
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = len(self.train_loader)

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            counts = batch.counts.to(self.device)
            obs_mask = batch.observed_mask.to(self.device)
            loss_mask = batch.loss_mask.to(self.device)
            padding_mask = batch.padding_mask.to(self.device)

            # Model forward call
            # Supports both VLDM and Attention model signatures
            rates_hz, latents, loss, metrics = self.model(
                counts=counts,
                observed_mask=obs_mask,
                target_counts=counts,
                loss_mask=loss_mask,
                key_padding_mask=padding_mask,
            )

            if loss is None:
                raise RuntimeError("Model returned None loss during training step")

            loss.backward()
            grad_norm = clip_gradients(self.model, self.config.clip_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += float(loss.item())
            total_grad_norm += grad_norm
            self.global_step += 1

        current_lr = float(self.optimizer.param_groups[0]["lr"])
        mean_loss = total_loss / max(1, num_batches)
        mean_grad = total_grad_norm / max(1, num_batches)
        return mean_loss, mean_grad, current_lr

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run validation evaluation across val_loader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_spikes = 0.0
        total_pred_rates = 0.0
        num_batches = len(self.val_loader)

        for batch in self.val_loader:
            counts = batch.counts.to(self.device)
            obs_mask = batch.observed_mask.to(self.device)
            loss_mask = batch.loss_mask.to(self.device)
            padding_mask = batch.padding_mask.to(self.device)

            rates_hz, latents, loss, metrics = self.model(
                counts=counts,
                observed_mask=obs_mask,
                target_counts=counts,
                loss_mask=loss_mask,
                key_padding_mask=padding_mask,
            )

            if loss is not None:
                total_loss += float(loss.item())
            total_spikes += float((counts * loss_mask.float()).sum().item())
            total_pred_rates += float(rates_hz.mean().item())

        val_loss = total_loss / max(1, num_batches)
        mean_rate = total_pred_rates / max(1, num_batches)

        return {
            "val_loss": val_loss,
            "total_val_spikes": total_spikes,
            "mean_val_rate": mean_rate,
        }

    def fit(self) -> Tuple[TrainingHistory, Optional[Path]]:
        """Run full training loop until max_epochs or early stopping triggers.

        Returns:
            Tuple of (training_history, path_to_best_checkpoint).
        """
        best_path: Optional[Path] = None

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss, grad_norm, lr = self._train_epoch()

            if epoch % self.config.eval_every_epochs == 0:
                val_metrics = self.evaluate()
                val_loss = val_metrics["val_loss"]

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Save checkpoint
                saved_path = self.checkpoint_manager.save(
                    epoch=epoch,
                    step=self.global_step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics={"train_loss": train_loss, **val_metrics},
                    is_best=is_best,
                )
                if is_best:
                    best_path = saved_path

                self.history.record(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=lr,
                    grad_norm=grad_norm,
                    extra=val_metrics,
                )

                if self.patience_counter >= self.config.patience:
                    break

        return self.history, best_path
