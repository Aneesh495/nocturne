"""Tests for JacobianAnalyzer eigenspectra and topological stability classification."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from nocturne.geometry import JacobianAnalyzer


def test_jacobian_analyzer_stable_spiral():
    # Linear damped oscillator:
    # dz1/dt = z2
    # dz2/dt = -omega^2 z1 - 2 gamma z2
    # J = [[0, 1], [-omega^2, -2 gamma]]
    omega = 2.0
    gamma = 0.5  # underdamped -> complex eigenvalues with negative real part

    def oscillator_vf(z: torch.Tensor) -> torch.Tensor:
        z1 = z[:, 0:1]
        z2 = z[:, 1:2]
        dz1 = z2
        dz2 = -(omega ** 2) * z1 - 2.0 * gamma * z2
        return torch.cat([dz1, dz2], dim=-1)

    fp_origin = np.array([0.0, 0.0])
    diag = JacobianAnalyzer.analyze_fixed_point(oscillator_vf, fp_origin)

    assert diag.is_stable
    assert diag.topological_type == "stable_focus"
    assert diag.max_real_eigenvalue == pytest.approx(-gamma, abs=1e-3)
    assert np.any(np.abs(np.imag(diag.eigenvalues)) > 0.0)


def test_jacobian_analyzer_saddle():
    # Saddle system: dz1/dt = z1, dz2/dt = -z2
    def saddle_vf(z: torch.Tensor) -> torch.Tensor:
        z1 = z[:, 0:1]
        z2 = z[:, 1:2]
        return torch.cat([z1, -z2], dim=-1)

    fp_origin = np.array([0.0, 0.0])
    diag = JacobianAnalyzer.analyze_fixed_point(saddle_vf, fp_origin)

    assert not diag.is_stable
    assert diag.topological_type == "saddle"
    assert diag.max_real_eigenvalue > 0.0
