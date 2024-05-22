"""Tests for StreamingBufferState sliding ring buffer and checkpointing."""

from __future__ import annotations

import torch
import pytest

from nocturne.streaming.state import StreamingBufferState


def test_streaming_buffer_sliding_window():
    obs_dim = 8
    max_w = 10
    state = StreamingBufferState(obs_dim=obs_dim, max_window_bins=max_w)

    # Push 6 bins
    c1 = torch.ones((1, 6, obs_dim))
    w1, start1 = state.push(c1)
    assert w1.shape == (1, 6, obs_dim)
    assert start1 == 0
    assert state.total_received_bins == 6

    # Push 8 bins -> total 14 > 10, should slide to max_w = 10
    c2 = torch.ones((1, 8, obs_dim)) * 2.0
    w2, start2 = state.push(c2)
    assert w2.shape == (1, 10, obs_dim)
    assert start2 == 2
    assert state.total_received_bins == 14

    # The oldest 4 bins from c1 were discarded, so w2 contains 2 bins from c1 and 8 bins from c2
    assert (w2[:, :2, :] == 1.0).all()
    assert (w2[:, 2:, :] == 2.0).all()


def test_streaming_buffer_checkpoint_and_restore():
    obs_dim = 4
    state = StreamingBufferState(obs_dim=obs_dim, max_window_bins=15)

    c = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    state.push(c)

    ckpt = state.capture_checkpoint()
    assert ckpt["total_received_bins"] == 5

    # Create fresh state and restore
    new_state = StreamingBufferState(obs_dim=obs_dim, max_window_bins=15)
    new_state.restore_checkpoint(ckpt)

    assert new_state.total_received_bins == 5
    assert torch.equal(new_state.window_counts, state.window_counts)
