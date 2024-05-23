"""Tests for OrthogonalProcrustesAligner and out-of-sample gauge transformation."""

from __future__ import annotations

import numpy as np
import pytest

from nocturne.geometry import (
    OrthogonalProcrustesAligner,
    ProcrustesAlignmentResult,
)


def test_procrustes_recovers_known_rotation_and_translation():
    np.random.seed(42)
    N, D = 100, 3

    # Generate source points
    Z1 = np.random.randn(N, D)

    # True rotation: 90 degrees around z-axis
    theta = np.pi / 2.0
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation_true = np.array([1.5, -2.0, 3.0])
    scale_true = 1.0

    # Target points Z2 = s * Z1 * R + t
    Z2 = scale_true * (Z1 @ R_true) + translation_true

    aligner = OrthogonalProcrustesAligner()
    res = aligner.evaluate(Z1, Z2)

    assert res.disparity < 1e-6
    assert np.allclose(res.R, R_true, atol=1e-5)
    assert np.allclose(res.translation, translation_true, atol=1e-5)


def test_procrustes_train_only_out_of_sample_generalization():
    np.random.seed(123)
    D = 4
    Z_train = np.random.randn(200, D)
    Z_test = np.random.randn(50, D)

    # Random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(D, D))
    Z_train_rot = Z_train @ Q
    Z_test_rot = Z_test @ Q

    aligner = OrthogonalProcrustesAligner()
    aligner.fit(Z_train, Z_train_rot)

    # Apply strictly out-of-sample
    Z_test_aligned = aligner.transform(Z_test)
    test_disparity = np.linalg.norm(Z_test_aligned - Z_test_rot) / np.linalg.norm(Z_test_rot)

    assert test_disparity < 1e-6
