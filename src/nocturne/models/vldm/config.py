"""Configuration schema for Variational Nonlinear Latent Dynamics (VLDM)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VLDMConfig(BaseModel):
    """Hyperparameter and architectural configuration for VLDM."""

    model_config = ConfigDict(frozen=True)

    obs_dim: int = Field(..., ge=1, description="Number of observed/total neural channels N")
    latent_dim: int = Field(default=8, ge=1, description="Dimensionality of latent state space D")
    encoder_hidden_dim: int = Field(default=64, ge=8, description="Encoder recurrent hidden state dimension")
    decoder_hidden_dim: int = Field(default=64, ge=8, description="Decoder feedforward hidden state dimension")
    bin_width_s: float = Field(default=0.02, gt=0.0, description="Temporal bin width in seconds")

    # Inferred drive vs autonomous dynamics switch
    use_inferred_drive: bool = Field(
        default=False,
        description="Whether to infer external time-varying inputs u_t or enforce autonomous dynamics",
    )
    drive_dim: int = Field(default=2, ge=1, description="Dimensionality of inferred external drive u_t")
    drive_l2_weight: float = Field(default=0.01, ge=0.0, description="L2 regularization penalty on inferred drive")

    # Variational terms
    kl_weight: float = Field(default=1.0, ge=0.0, description="Weight multiplier beta for initial state KL divergence")
    transition_noise_std: float = Field(default=0.1, gt=0.0, description="Innovation noise standard deviation")
    softplus_beta: float = Field(default=1.0, gt=0.0, description="Beta parameter for softplus rate readout")
