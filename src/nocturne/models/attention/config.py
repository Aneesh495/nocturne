"""Configuration schema for Axial Spatiotemporal Masked Attention Model (STA)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AttentionConfig(BaseModel):
    """Architectural configuration for axial spatiotemporal attention."""

    model_config = ConfigDict(frozen=True)

    obs_dim: int = Field(..., ge=1, description="Number of neural channels N")
    d_model: int = Field(default=64, ge=8, description="Token representation embedding dimension")
    num_heads: int = Field(default=4, ge=1, description="Number of multi-head attention heads")
    num_layers: int = Field(default=3, ge=1, description="Number of stacked axial transformer blocks")
    dim_feedforward: int = Field(default=128, ge=16, description="Dimension of feedforward expansion layer")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout probability")
    bin_width_s: float = Field(default=0.02, gt=0.0, description="Bin duration in seconds")
    max_seq_len: int = Field(default=512, ge=16, description="Maximum temporal sequence length")
    causal: bool = Field(default=False, description="Strict causal temporal attention masking flag")
    softplus_beta: float = Field(default=1.0, gt=0.0, description="Beta parameter for softplus rate readout")
