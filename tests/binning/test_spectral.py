"""Unit tests for multitaper spectral density and point-process coherence."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from nocturne.binning.spectral import (
    MultitaperSpectralEstimator,
    SpectralDensityResult,
)


def test_pure_frequency_peak_recovery() -> None:
    dt = 0.01  # 100 Hz sampling rate, 50 Hz Nyquist
    T = 200
    t = np.arange(T) * dt
    f0 = 12.0  # 12 Hz oscillation

    # Channel 0 has strong 12 Hz oscillation
    sig = 2.0 * np.sin(2.0 * math.pi * f0 * t) + 0.1 * np.random.randn(T)
    # Channel 1 has 25 Hz oscillation
    sig2 = 2.0 * np.sin(2.0 * math.pi * 25.0 * t) + 0.1 * np.random.randn(T)

    data = np.column_stack([sig, sig2])  # [T, 2]

    estimator = MultitaperSpectralEstimator(bin_width_s=dt, time_half_bandwidth=3.0)
    res = estimator.compute_psd(data, n_fft=256)

    assert isinstance(res, SpectralDensityResult)
    assert res.nyquist_freq_hz == 50.0
    assert len(res.peak_frequencies_hz) == 2
    # Verify channel 0 peak is near 12 Hz
    assert pytest.approx(res.peak_frequencies_hz[0], abs=1.0) == 12.0
    # Verify channel 1 peak is near 25 Hz
    assert pytest.approx(res.peak_frequencies_hz[1], abs=1.0) == 25.0


def test_cross_channel_coherence() -> None:
    dt = 0.02
    T = 150
    t = np.arange(T) * dt
    f_shared = 8.0

    # Synchronous shared oscillation + independent noise across 4 trials
    B = 4
    shared_wave = np.sin(2.0 * math.pi * f_shared * t)
    sig_a = np.tile(shared_wave, (B, 1)) + 0.1 * np.random.randn(B, T)
    sig_b = np.tile(shared_wave, (B, 1)) + 0.1 * np.random.randn(B, T)

    counts = np.stack([sig_a, sig_b], axis=-1)  # [B, T, 2]

    estimator = MultitaperSpectralEstimator(bin_width_s=dt)
    res = estimator.compute_psd(counts, compute_coherence=True)

    assert res.coherence_matrix is not None
    assert res.coherence_matrix.shape == (2, 2, len(res.frequencies_hz))
    # Diagonal coherence is exactly 1.0
    assert np.allclose(res.coherence_matrix[0, 0, :], 1.0)
    assert np.allclose(res.coherence_matrix[1, 1, :], 1.0)

    # Coherence at the shared 8 Hz frequency should be very high (> 0.85)
    freq_idx = np.argmin(np.abs(np.array(res.frequencies_hz) - f_shared))
    shared_coh = res.coherence_matrix[0, 1, freq_idx]
    assert shared_coh > 0.85


def test_pytorch_tensor_and_short_sequence_error() -> None:
    estimator = MultitaperSpectralEstimator(bin_width_s=0.02)
    tensor_data = torch.randn(2, 50, 4)
    res = estimator.compute_psd(tensor_data)
    assert res.psd.shape == (4, len(res.frequencies_hz))

    short_data = np.ones((2, 3, 4))
    with pytest.raises(ValueError, match="too short"):
        estimator.compute_psd(short_data)
