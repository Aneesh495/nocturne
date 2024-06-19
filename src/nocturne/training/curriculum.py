"""Curriculum learning schedules for self-supervised neural dynamics training.

Controls progressive difficulty scaling across training epochs, including:
1. Masking density ramping (linear, cosine, or exponential).
2. Masking strategy sequencing (token reconstruction -> channel co-smoothing -> causal forecasting).
3. Temporal context window expansion for stabilized recurrent and attention training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CurriculumState:
    """Snapshot of curriculum parameters for a given training epoch."""

    epoch: int
    mask_rate: float
    mask_strategy: str
    max_sequence_length: Optional[int] = None
    teacher_forcing_ratio: float = 1.0


class CurriculumScheduler:
    """Manages epoch-dependent training difficulty progression."""

    def __init__(
        self,
        total_epochs: int = 100,
        warmup_epochs: int = 10,
        initial_mask_rate: float = 0.05,
        target_mask_rate: float = 0.30,
        schedule_type: str = "cosine",
        strategy_schedule: Optional[Dict[int, str]] = None,
        min_seq_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
    ):
        if total_epochs <= 0:
            raise ValueError(f"total_epochs must be > 0, got {total_epochs}")
        if warmup_epochs < 0 or warmup_epochs > total_epochs:
            raise ValueError(f"Invalid warmup_epochs: {warmup_epochs}")
        if not (0.0 <= initial_mask_rate <= 1.0 and 0.0 <= target_mask_rate <= 1.0):
            raise ValueError("Mask rates must be within [0, 1]")
        if schedule_type not in ("linear", "cosine", "constant", "exponential"):
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.initial_mask_rate = initial_mask_rate
        self.target_mask_rate = target_mask_rate
        self.schedule_type = schedule_type

        # Strategy schedule: epoch -> strategy
        # Default: start with 'token', shift to 'channel' at 30%, shift to 'temporal_tail' at 70%
        if strategy_schedule is None:
            e1 = int(total_epochs * 0.35)
            e2 = int(total_epochs * 0.70)
            self.strategy_schedule = {
                0: "token",
                e1: "channel",
                e2: "temporal_tail",
            }
        else:
            self.strategy_schedule = dict(sorted(strategy_schedule.items()))

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len

    def get_mask_rate(self, epoch: int) -> float:
        """Calculate the scheduled mask rate for a given epoch."""
        if epoch < 0:
            epoch = 0

        if self.schedule_type == "constant" or self.total_epochs == self.warmup_epochs:
            return self.target_mask_rate

        if epoch < self.warmup_epochs:
            # Linear warmup from initial_mask_rate
            if self.warmup_epochs == 0:
                progress = 1.0
            else:
                progress = epoch / float(self.warmup_epochs)
            return self.initial_mask_rate * progress

        effective_epoch = epoch - self.warmup_epochs
        remaining_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, effective_epoch / float(remaining_epochs))

        if self.schedule_type == "linear":
            rate = self.initial_mask_rate + progress * (self.target_mask_rate - self.initial_mask_rate)
        elif self.schedule_type == "cosine":
            cosine_factor = 0.5 * (1.0 - math.cos(math.pi * progress))
            rate = self.initial_mask_rate + cosine_factor * (self.target_mask_rate - self.initial_mask_rate)
        elif self.schedule_type == "exponential":
            rate = self.initial_mask_rate * ((self.target_mask_rate / max(1e-4, self.initial_mask_rate)) ** progress)
        else:
            rate = self.target_mask_rate

        return float(np_clip(rate, 0.0, 1.0))

    def get_mask_strategy(self, epoch: int) -> str:
        """Determine active masking strategy for epoch according to stage transitions."""
        active_strategy = "token"
        for thresh_epoch, strat in sorted(self.strategy_schedule.items()):
            if epoch >= thresh_epoch:
                active_strategy = strat
            else:
                break
        return active_strategy

    def get_sequence_length(self, epoch: int) -> Optional[int]:
        """Interpolate maximum sequence length window if sequence curriculum is active."""
        if self.min_seq_len is None or self.max_seq_len is None:
            return None

        if epoch >= self.total_epochs:
            return self.max_seq_len

        progress = min(1.0, max(0.0, epoch / float(self.total_epochs)))
        cur_len = self.min_seq_len + int(progress * (self.max_seq_len - self.min_seq_len))
        return int(cur_len)

    def step(self, epoch: int) -> CurriculumState:
        """Compute full curriculum parameter state for the designated epoch."""
        mask_rate = self.get_mask_rate(epoch)
        mask_strategy = self.get_mask_strategy(epoch)
        seq_len = self.get_sequence_length(epoch)

        # Teacher forcing ratio decays linearly from 1.0 to 0.0 over epochs
        tf_ratio = max(0.0, 1.0 - (epoch / float(self.total_epochs)))

        return CurriculumState(
            epoch=epoch,
            mask_rate=mask_rate,
            mask_strategy=mask_strategy,
            max_sequence_length=seq_len,
            teacher_forcing_ratio=tf_ratio,
        )

    def state_dict(self) -> Dict[str, Any]:
        """Serialize configuration parameters for checkpointing."""
        return {
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "initial_mask_rate": self.initial_mask_rate,
            "target_mask_rate": self.target_mask_rate,
            "schedule_type": self.schedule_type,
            "strategy_schedule": self.strategy_schedule,
            "min_seq_len": self.min_seq_len,
            "max_seq_len": self.max_seq_len,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> CurriculumScheduler:
        """Instantiate scheduler from serialized dictionary."""
        return cls(
            total_epochs=data["total_epochs"],
            warmup_epochs=data["warmup_epochs"],
            initial_mask_rate=data["initial_mask_rate"],
            target_mask_rate=data["target_mask_rate"],
            schedule_type=data["schedule_type"],
            strategy_schedule={int(k): v for k, v in data["strategy_schedule"].items()},
            min_seq_len=data.get("min_seq_len"),
            max_seq_len=data.get("max_seq_len"),
        )


def np_clip(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))
