"""Inferred external drive subnetwork and regularization penalty."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn

from nocturne.models.vldm.config import VLDMConfig


class InferredDriveNetwork(nn.Module):
    """Inferred time-varying external drive network u_t in R^M.

    If use_inferred_drive is False, strictly emits zero drive u_t = 0, enforcing
    purely autonomous vector field dynamics.
    If use_inferred_drive is True, estimates external drive and penalizes its L2 magnitude.
    """

    def __init__(self, config: VLDMConfig):
        super().__init__()
        self.config = config
        self.use_drive = config.use_inferred_drive
        self.drive_dim = config.drive_dim
        self.latent_dim = config.latent_dim
        self.l2_weight = config.drive_l2_weight

        if self.use_drive:
            self.drive_gru = nn.GRU(
                input_size=config.obs_dim,
                hidden_size=config.encoder_hidden_dim,
                batch_first=True,
            )
            self.drive_head = nn.Linear(config.encoder_hidden_dim, self.drive_dim)
            self.input_matrix = nn.Linear(self.drive_dim, self.latent_dim, bias=False)

    def forward(
        self,
        counts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Infer external driving trajectory u_{1:T} and compute penalty.

        Args:
            counts: Observed spike counts [B, T, N].
            mask: Optional observation mask [B, T, N].

        Returns:
            Tuple of (drive [B, T, M] or None, drive_penalty scalar).
        """
        if not self.use_drive:
            # Enforce strictly autonomous flow: no drive
            return None, torch.tensor(0.0, device=counts.device, dtype=counts.dtype)

        if mask is not None:
            counts_in = counts * mask.float()
        else:
            counts_in = counts

        h, _ = self.drive_gru(counts_in)  # [B, T, hidden]
        u = self.drive_head(h)            # [B, T, drive_dim]

        # L2 penalty encouraging minimal drive: ||u||^2
        penalty = self.l2_weight * u.pow(2).mean()

        return u, penalty

    def project_to_latent_velocity(self, u: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Project drive u_t into latent velocity space B * u_t."""
        if not self.use_drive or u is None:
            return None
        return self.input_matrix(u)
