"""Multitaper spectral density and point-process coherence estimation.

Computes power spectral density and cross-channel coherence for spike trains
using Discrete Prolate Spheroidal Sequences (DPSS / Slepian tapers).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.signal.windows import dpss
import torch


@dataclass(frozen=True)
class SpectralDensityResult:
    """Frequency-domain spectral estimation and coherence result."""

    frequencies_hz: List[float]
    psd: np.ndarray  # [N, F]
    coherence_matrix: Optional[np.ndarray]  # [N, N, F]
    peak_frequencies_hz: List[float]  # [N]
    nyquist_freq_hz: float
    df: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to dictionary with serializable arrays."""
        d = asdict(self)
        d["psd"] = self.psd.tolist()
        if self.coherence_matrix is not None:
            d["coherence_matrix"] = self.coherence_matrix.tolist()
        return d


class MultitaperSpectralEstimator:
    """Estimates power spectra and coherence for neural point-process count sequences."""

    def __init__(
        self,
        bin_width_s: float = 0.02,
        time_half_bandwidth: float = 3.0,
        num_tapers: Optional[int] = None,
    ):
        if bin_width_s <= 0.0:
            raise ValueError(f"bin_width_s must be > 0, got {bin_width_s}")
        if time_half_bandwidth <= 1.0:
            raise ValueError(f"time_half_bandwidth must be > 1.0, got {time_half_bandwidth}")

        self.bin_width_s = bin_width_s
        self.sampling_rate_hz = 1.0 / bin_width_s
        self.nyquist_freq_hz = self.sampling_rate_hz / 2.0
        self.time_half_bandwidth = time_half_bandwidth
        self.num_tapers = num_tapers or int(2 * time_half_bandwidth - 1)

    def compute_psd(
        self,
        counts: np.ndarray | torch.Tensor,
        n_fft: Optional[int] = None,
        compute_coherence: bool = False,
    ) -> SpectralDensityResult:
        """Compute multitaper power spectral density across channels.

        Args:
            counts: Array of shape [T, N] or batched [B, T, N] of spike counts.
            n_fft: Optional FFT length (defaults to next power of 2 >= T).
            compute_coherence: If True, computes all pairwise cross-channel coherences.

        Returns:
            SpectralDensityResult containing frequency grid, PSD, and coherence.
        """
        if isinstance(counts, torch.Tensor):
            arr = counts.detach().cpu().numpy().astype(np.float64)
        else:
            arr = np.asarray(counts, dtype=np.float64)

        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # [1, T, N]

        B, T, N = arr.shape
        if T < 4:
            raise ValueError(f"Sequence length T={T} is too short for spectral analysis")

        if n_fft is None:
            n_fft = max(64, int(2 ** math.ceil(math.log2(T))))

        # Positive one-sided frequencies up to Nyquist
        freqs = np.fft.rfftfreq(n_fft, d=self.bin_width_s)  # [F]
        num_freqs = len(freqs)
        df = float(freqs[1] - freqs[0]) if num_freqs > 1 else 0.0

        # Compute Slepian tapers: shape [K, T]
        tapers = dpss(T, NW=self.time_half_bandwidth, Kmax=self.num_tapers, sym=False)  # [K, T]
        K = tapers.shape[0]

        # Centered signals: remove mean firing rate per trial and channel
        centered = arr - np.mean(arr, axis=1, keepdims=True)  # [B, T, N]

        # Tapered Fourier transforms: J_{b, k, n}(f)
        # Reshape for broadcasting: centered [B, 1, T, N], tapers [1, K, T, 1]
        tapered_signal = centered[:, np.newaxis, :, :] * tapers[np.newaxis, :, :, np.newaxis]
        # FFT along time axis: [B, K, F, N]
        J = np.fft.rfft(tapered_signal, n=n_fft, axis=2)

        # Power spectral density per channel: S(f) = (1 / K) sum_k |J_k(f)|^2 * dt
        # Average over trials B and tapers K: [F, N]
        psd_raw = np.mean(np.abs(J)**2, axis=(0, 1)) * self.bin_width_s  # [F, N]
        psd = psd_raw.T  # [N, F]

        # Find peak frequencies per channel (excluding DC zero frequency)
        peak_freqs: List[float] = []
        for n in range(N):
            if num_freqs > 1:
                peak_idx = 1 + np.argmax(psd[n, 1:])
                peak_freqs.append(float(freqs[peak_idx]))
            else:
                peak_freqs.append(float(freqs[0]))

        # Compute pairwise cross-channel coherence if requested
        coherence_matrix: Optional[np.ndarray] = None
        if compute_coherence and N > 1:
            # Cross-spectrum: S_{ij}(f) = mean_{b,k} [J_i(f) * conj(J_j(f))]
            # Reshape J: [B * K, F, N]
            J_flat = J.reshape(-1, num_freqs, N)  # [M, F, N]
            coherence = np.zeros((N, N, num_freqs), dtype=np.float64)

            # Auto-spectra for normalization
            auto_specs = np.mean(np.abs(J_flat)**2, axis=0)  # [F, N]

            for i in range(N):
                coherence[i, i, :] = 1.0
                for j in range(i + 1, N):
                    cross_spec = np.mean(J_flat[:, :, i] * np.conj(J_flat[:, :, j]), axis=0)
                    denom = auto_specs[:, i] * auto_specs[:, j] + 1e-12
                    coh = (np.abs(cross_spec)**2) / denom
                    coherence[i, j, :] = coh
                    coherence[j, i, :] = coh

            coherence_matrix = coherence

        return SpectralDensityResult(
            frequencies_hz=[float(f) for f in freqs],
            psd=psd,
            coherence_matrix=coherence_matrix,
            peak_frequencies_hz=peak_freqs,
            nyquist_freq_hz=float(self.nyquist_freq_hz),
            df=df,
        )
