"""Tests for adversarial attractor falsification and surrogate null testing."""

import numpy as np
import pytest
from nocturne.verification.synthetic_falsification import (
    AdversarialAttractorFalsifier,
    FalsificationResult,
    FalsificationReport,
)


def test_falsifier_ou_generation_shapes():
    falsifier = AdversarialAttractorFalsifier(dt=0.01)
    traj = falsifier.generate_ornstein_uhlenbeck(T=200, D=3)
    assert traj.shape == (200, 3)
    assert np.all(np.isfinite(traj))


def test_falsifier_fourier_surrogates():
    falsifier = AdversarialAttractorFalsifier(dt=0.01)
    orig = falsifier.generate_ornstein_uhlenbeck(T=256, D=2)
    surr = falsifier.generate_fourier_surrogates(orig)

    assert surr.shape == orig.shape
    # Power spectra amplitudes should be identical within numerical precision
    fft_orig = np.abs(np.fft.rfft(orig, axis=0))
    fft_surr = np.abs(np.fft.rfft(surr, axis=0))
    np.testing.assert_allclose(fft_orig, fft_surr, atol=1e-5)


def test_linear_sink_falsification_passes():
    falsifier = AdversarialAttractorFalsifier(num_seeds=15)
    res: FalsificationResult = falsifier.test_linear_sink_falsification()

    assert res.regime_name == "linear_spiral_sink"
    assert res.passed is True
    assert res.spurious_attractor_detected is False
    assert res.detected_fixed_points_count <= 1


def test_diffusion_surrogate_test_passes():
    falsifier = AdversarialAttractorFalsifier(surrogate_shuffles=25, random_seed=42)
    res: FalsificationResult = falsifier.test_stochastic_diffusion_surrogate()

    assert res.regime_name == "ornstein_uhlenbeck_diffusion"
    assert res.passed is True
    assert res.spurious_attractor_detected is False
    assert res.surrogate_p_value >= 0.05


def test_full_falsification_suite():
    falsifier = AdversarialAttractorFalsifier(num_seeds=10, surrogate_shuffles=20, random_seed=42)
    report: FalsificationReport = falsifier.run_falsification_suite()

    assert report.all_passed is True
    assert len(report.results) == 2
    for r in report.results:
        assert r.passed is True
