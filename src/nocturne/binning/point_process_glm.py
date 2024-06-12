"""Point-process Generalized Linear Model (GLM) with raised-cosine history filters.

Implements classical post-spike refractory kernels and cross-neuronal coupling
convolutions based on Truccolo et al. (2005) and Pillow et al. (2008).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nocturne.binning.likelihood_math import get_log_factorial


class RaisedCosineBasis:
    """Generates non-linearly scaled raised-cosine temporal basis functions.

    Basis functions are logarithmically spaced to provide high temporal resolution
    immediately following a spike (absolute/relative refractory period) and coarse
    resolution at longer temporal lags.
    """

    def __init__(
        self,
        num_bases: int = 5,
        history_length_bins: int = 20,
        c: float = 0.5,
    ):
        if num_bases < 2:
            raise ValueError(f"num_bases must be >= 2, got {num_bases}")
        if history_length_bins < num_bases:
            raise ValueError(f"history_length_bins must be >= num_bases, got {history_length_bins}")

        self.num_bases = num_bases
        self.history_length = history_length_bins
        self.c = c
        self.basis_matrix = self._construct_basis()

    def _construct_basis(self) -> np.ndarray:
        # Time steps from 1 to history_length (causal lag strictly >= 1)
        t = np.arange(1, self.history_length + 1, dtype=np.float64)
        log_t = np.log(t + self.c)
        min_log = log_t[0]
        max_log = log_t[-1]

        # Equispaced peaks in log-time
        peaks = np.linspace(min_log, max_log, self.num_bases)
        step = peaks[1] - peaks[0] if len(peaks) > 1 else 1.0

        basis = np.zeros((self.history_length, self.num_bases), dtype=np.float64)
        for i, peak in enumerate(peaks):
            dist = (log_t - peak) / step
            # Non-zero for |dist| < 1
            mask = np.abs(dist) < 1.0
            basis[mask, i] = 0.5 * (1.0 + np.cos(math.pi * dist[mask]))

        # Normalize each basis function to unit Frobenius norm
        norms = np.linalg.norm(basis, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        return basis / norms

    def to_torch(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Convert basis matrix to torch Tensor of shape [history_length, num_bases]."""
        return torch.tensor(self.basis_matrix, dtype=torch.float32, device=device)


class PointProcessGLM(nn.Module):
    """Neural point-process Generalized Linear Model (GLM).

    Model equation:
        lambda_i(t) = softplus(b_i + sum_j (h_{ij} * y_j)(t) + (w_i * x)(t))
    where h_{ij} is parameterized over raised-cosine temporal basis functions.
    """

    def __init__(
        self,
        num_channels: int,
        num_bases: int = 5,
        history_length_bins: int = 20,
        include_cross_coupling: bool = False,
        bin_width_s: float = 0.02,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_bases = num_bases
        self.history_length = history_length_bins
        self.include_cross_coupling = include_cross_coupling
        self.bin_width_s = bin_width_s

        # Raised cosine basis buffer: [W, num_bases]
        basis_gen = RaisedCosineBasis(num_bases, history_length_bins)
        basis_tensor = basis_gen.to_torch()
        self.register_buffer("basis", basis_tensor)

        # Baseline firing rate log-bias per channel
        self.bias = nn.Parameter(torch.zeros(num_channels))

        # Post-spike history weights
        if include_cross_coupling:
            # Full coupling matrix: [N_target, N_source, num_bases]
            self.history_weights = nn.Parameter(
                torch.randn(num_channels, num_channels, num_bases) * 0.01
            )
        else:
            # Self-history only: [N, num_bases]
            self.history_weights = nn.Parameter(
                torch.randn(num_channels, num_bases) * 0.01
            )

    def get_effective_filters(self) -> torch.Tensor:
        """Expand basis weights to full temporal filters.

        Returns:
            Tensor of shape [N, N, W] if cross-coupling, else [N, W].
        """
        if self.include_cross_coupling:
            # history_weights: [N, N, B], basis: [W, B]
            # filters: [N, N, W] = sum_b w_{ijb} * basis_{wb}
            return torch.einsum("ijb,wb->ijw", self.history_weights, self.basis)
        # history_weights: [N, B], basis: [W, B] -> [N, W]
        return torch.einsum("ib,wb->iw", self.history_weights, self.basis)

    def forward(
        self,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute instantaneous predicted firing rates in Hz.

        Args:
            counts: Input spike count tensor [B, T, N].

        Returns:
            Predicted firing rate tensor in Hz [B, T, N].
        """
        B, T, N = counts.shape
        W = self.history_length

        # Causal padding on time dimension: pad left by W, pad right by 0
        # Permute to [B, N, T] for 1D convolution
        counts_perm = counts.transpose(1, 2)  # [B, N, T]
        padded = F.pad(counts_perm, (W, 0))  # [B, N, T + W]

        if self.include_cross_coupling:
            filters = self.get_effective_filters()  # [N_out, N_in, W]
            # Conv1d expects weight: [out_channels, in_channels, kernel_size]
            # Flip filter horizontally for causal correlation
            kernel = torch.flip(filters, dims=[-1])
            # Convolve padded sequence: output length is T + 1
            conv = F.conv1d(padded, kernel)  # [B, N, T + 1]
            # Strictly causal: slice lag >= 1 (drop the last step which aligns with current lag 0)
            hist_effect = conv[..., :T].transpose(1, 2)  # [B, T, N]
        else:
            filters = self.get_effective_filters()  # [N, W]
            kernel = torch.flip(filters.unsqueeze(1), dims=[-1])  # [N, 1, W]
            # Depthwise convolution per channel
            conv = F.conv1d(padded, kernel, groups=N)  # [B, N, T + 1]
            hist_effect = conv[..., :T].transpose(1, 2)  # [B, T, N]

        # Rate readout via softplus: lambda = softplus(bias + hist_effect) / dt
        total_input = self.bias.view(1, 1, N) + hist_effect
        rates_hz = F.softplus(total_input) / self.bin_width_s
        return torch.clamp(rates_hz, min=1e-5)

    def compute_nll(
        self,
        counts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Poisson negative log-likelihood loss for training."""
        rates_hz = self.forward(counts)
        dt = self.bin_width_s
        expected_counts = torch.clamp(rates_hz * dt, min=1e-8)
        log_fact = torch.tensor(
            get_log_factorial(counts.detach().cpu().numpy()),
            dtype=expected_counts.dtype,
            device=counts.device,
        )
        # NLL element: lambda * dt - y * ln(lambda * dt) + ln(y!)
        nll_element = expected_counts - (counts * torch.log(expected_counts)) + log_fact
        if mask is not None:
            nll_element = nll_element * mask.float()
            denom = torch.clamp(mask.float().sum(), min=1.0)
            return nll_element.sum() / denom
        return nll_element.mean()
