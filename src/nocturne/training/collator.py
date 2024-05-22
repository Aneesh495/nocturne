"""Batch collation, padding, and masked objective token generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch


@dataclass(frozen=True)
class NeuralBatch:
    """Collated mini-batch of neural recordings with observation and padding masks."""

    counts: torch.Tensor          # [B, T_max, N] float32 spike counts
    observed_mask: torch.Tensor   # [B, T_max, N] bool: True indicates visible to encoder
    loss_mask: torch.Tensor       # [B, T_max, N] bool: True indicates evaluated in loss
    padding_mask: torch.Tensor    # [B, T_max] bool: True indicates padded temporal steps
    lengths: torch.Tensor         # [B] int64: unpadded length T_i for each trial
    kinematics: Optional[torch.Tensor] = None  # [B, T_max, K] float32 behavioral state


class NeuralBatchCollator:
    """Pads variable-length neural trials and constructs self-supervised training masks."""

    def __init__(
        self,
        mask_rate: float = 0.15,
        mask_strategy: str = "token",
    ):
        if not (0.0 <= mask_rate <= 1.0):
            raise ValueError(f"mask_rate must be in [0, 1], got {mask_rate}")
        if mask_strategy not in ("token", "channel", "temporal_tail"):
            raise ValueError(f"Unknown mask_strategy: {mask_strategy}")

        self.mask_rate = mask_rate
        self.mask_strategy = mask_strategy

    def __call__(
        self,
        batch: List[Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]],
    ) -> NeuralBatch:
        """Collate a list of (counts_i, kinematics_i) pairs into a padded NeuralBatch.

        Args:
            batch: List of tuples where counts_i has shape [T_i, N] and optional kinematics_i has shape [T_i, K].

        Returns:
            Collated NeuralBatch.
        """
        B = len(batch)
        if B == 0:
            raise ValueError("Cannot collate an empty batch")

        # Determine shapes
        counts_list = []
        kinematics_list = []
        lengths = []

        for item in batch:
            c = torch.as_tensor(item[0], dtype=torch.float32)
            k = torch.as_tensor(item[1], dtype=torch.float32) if item[1] is not None else None
            counts_list.append(c)
            kinematics_list.append(k)
            lengths.append(c.shape[0])

        T_max = max(lengths)
        N = counts_list[0].shape[1]

        # Allocate padded tensors
        padded_counts = torch.zeros((B, T_max, N), dtype=torch.float32)
        padding_mask = torch.ones((B, T_max), dtype=torch.bool)
        valid_mask = torch.zeros((B, T_max, N), dtype=torch.bool)

        has_kinematics = kinematics_list[0] is not None
        padded_kinematics: Optional[torch.Tensor] = None
        if has_kinematics:
            K = kinematics_list[0].shape[1]
            padded_kinematics = torch.zeros((B, T_max, K), dtype=torch.float32)

        for b in range(B):
            T_b = lengths[b]
            padded_counts[b, :T_b, :] = counts_list[b]
            padding_mask[b, :T_b] = False  # False means valid (standard PyTorch key_padding_mask convention)
            valid_mask[b, :T_b, :] = True
            if has_kinematics and kinematics_list[b] is not None:
                padded_kinematics[b, :T_b, :] = kinematics_list[b]

        # Construct observed and loss masks according to mask strategy
        observed_mask = valid_mask.clone()
        loss_mask = valid_mask.clone()

        if self.mask_rate > 0.0:
            if self.mask_strategy == "token":
                # Random token-level masking
                rand_draw = torch.rand((B, T_max, N))
                masked_positions = (rand_draw < self.mask_rate) & valid_mask
                observed_mask = valid_mask & (~masked_positions)
                loss_mask = masked_positions

            elif self.mask_strategy == "channel":
                # Mask entire neural channels per trial (co-smoothing objective)
                num_masked_channels = max(1, int(N * self.mask_rate))
                for b in range(B):
                    perm = torch.randperm(N)
                    masked_ch = perm[:num_masked_channels]
                    observed_mask[b, :, masked_ch] = False
                    loss_mask[b, :, :] = False
                    loss_mask[b, :lengths[b], masked_ch] = True

            elif self.mask_strategy == "temporal_tail":
                # Mask trailing time bins (forecasting objective)
                for b in range(B):
                    T_b = lengths[b]
                    horizon = max(1, int(T_b * self.mask_rate))
                    split_pt = T_b - horizon
                    observed_mask[b, split_pt:T_b, :] = False
                    loss_mask[b, :, :] = False
                    loss_mask[b, split_pt:T_b, :] = True

        return NeuralBatch(
            counts=padded_counts,
            observed_mask=observed_mask,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            lengths=torch.tensor(lengths, dtype=torch.int64),
            kinematics=padded_kinematics,
        )
