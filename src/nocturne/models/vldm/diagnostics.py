"""Diagnostics for evaluating inferred drive versus autonomous vector field dynamics."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field
import torch

from nocturne.models.vldm.model import VariationalLatentDynamicsModel


class DynamicsDriveAudit(BaseModel):
    """Audit report quantifying the balance between autonomous flow and inferred external drive."""

    model_config = ConfigDict(frozen=True)

    autonomous_flow_norm: float = Field(..., ge=0.0, description="Mean L2 norm of autonomous velocity f(z)")
    inferred_drive_norm: float = Field(..., ge=0.0, description="Mean L2 norm of projected drive B * u")
    autonomous_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Ratio of autonomous flow to total velocity: ||f|| / (||f|| + ||Bu||)"
    )
    is_drive_dominated: bool = Field(
        ..., description="True if inferred drive dominates (>75% of total velocity), indicating failure to learn autonomous flow"
    )
    diagnostic_message: str = Field(..., description="Human-readable assessment of dynamical identifiability")


class DynamicsAuditor:
    """Evaluates whether VLDM learned genuine autonomous vector fields or collapsed to drive-fitting."""

    @staticmethod
    def audit_model(
        model: VariationalLatentDynamicsModel,
        test_counts: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> DynamicsDriveAudit:
        """Audit the balance between autonomous flow and external drive across test trials."""
        model.eval()
        with torch.no_grad():
            B, T, N = test_counts.shape
            mu_0, log_var_0 = model.encoder(test_counts, mask=observed_mask)
            drive_u, _ = model.drive_net(test_counts, mask=observed_mask)
            drive_proj = model.drive_net.project_to_latent_velocity(drive_u)  # [B, T, D] or None

            latents = model.rollout_latents(mu_0, num_time_bins=T, inferred_drive=drive_u)

            # Evaluate autonomous velocity field f_theta(z)
            auto_vel = model.transitions.net(latents)  # [B, T, D]
            auto_norm = float(torch.norm(auto_vel, dim=-1).mean().item())

            if drive_proj is not None:
                drive_norm = float(torch.norm(drive_proj, dim=-1).mean().item())
            else:
                drive_norm = 0.0

            total_norm = auto_norm + drive_norm + 1e-8
            ratio = auto_norm / total_norm
            is_dominated = ratio < 0.25 and drive_norm > 0.1

            if not model.config.use_inferred_drive:
                msg = "Strictly autonomous dynamics mode enforced (inferred drive disabled)."
            elif is_dominated:
                msg = "WARNING: Model collapsed to drive-dominated mode. Inferred drive explains trajectories without learning autonomous vector field."
            else:
                msg = f"Autonomous dynamics dominant ({ratio*100:.1f}% autonomous flow)."

            return DynamicsDriveAudit(
                autonomous_flow_norm=auto_norm,
                inferred_drive_norm=drive_norm,
                autonomous_ratio=ratio,
                is_drive_dominated=is_dominated,
                diagnostic_message=msg,
            )
