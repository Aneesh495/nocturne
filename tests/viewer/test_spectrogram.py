"""Unit tests for vector SVG spectrogram visualizer."""

from __future__ import annotations

import re
import numpy as np
import pytest
import torch

from nocturne.viewer.spectrogram import (
    SpectrogramVisualizer,
    _viridis_colormap,
)


def test_viridis_colormap_hex_format() -> None:
    hex_pattern = re.compile(r"^#[0-9a-f]{6}$")
    for val in [0.0, 0.25, 0.5, 0.75, 1.0, -0.5, 1.5]:
        col = _viridis_colormap(val)
        assert hex_pattern.match(col) is not None


def test_spectrogram_render_svg_validity() -> None:
    # 16 frequency rows, 30 time columns
    F_bins, T_bins = 16, 30
    freqs = list(np.linspace(2.0, 50.0, F_bins))
    power = np.random.exponential(scale=1.0, size=(F_bins, T_bins))

    viz = SpectrogramVisualizer(bin_width_s=0.02)
    svg = viz.render_svg(
        power_matrix=power,
        freq_grid_hz=freqs,
        title="Beta-Band Population Power",
        width=700,
        height=350,
    )

    assert svg.startswith("<svg")
    assert svg.strip().endswith("</svg>")
    assert 'viewBox="0 0 700 350"' in svg
    assert "Beta-Band Population Power" in svg
    assert "Hz" in svg
    assert "dB" in svg
    assert "<rect" in svg


def test_pytorch_tensor_input() -> None:
    viz = SpectrogramVisualizer()
    tensor_mat = torch.rand(10, 20)
    svg = viz.render_svg(tensor_mat)
    assert "</svg>" in svg


def test_invalid_shape_raises() -> None:
    viz = SpectrogramVisualizer()
    with pytest.raises(ValueError, match="Expected 2D matrix"):
        viz.render_svg(np.ones((2, 3, 4)))

    with pytest.raises(ValueError, match="at least 2"):
        viz.render_svg(np.ones((1, 5)))
