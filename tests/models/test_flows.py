"""Unit tests for planar and radial normalizing flow priors."""

from __future__ import annotations

import pytest
import torch

from nocturne.models.vldm.flows import (
    NormalizingFlowPrior,
    PlanarFlow,
    RadialFlow,
)


def test_planar_flow_forward_and_gradients() -> None:
    flow = PlanarFlow(dim=4)
    z = torch.randn(10, 4, requires_grad=True)

    z_next, log_det = flow(z)

    assert z_next.shape == (10, 4)
    assert log_det.shape == (10,)
    assert torch.all(torch.isfinite(z_next))
    assert torch.all(torch.isfinite(log_det))

    loss = torch.sum(z_next**2) + torch.sum(log_det)
    loss.backward()

    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))


def test_radial_flow_forward_and_gradients() -> None:
    flow = RadialFlow(dim=6)
    z = torch.randn(8, 6, requires_grad=True)

    z_next, log_det = flow(z)

    assert z_next.shape == (8, 6)
    assert log_det.shape == (8,)
    assert torch.all(torch.isfinite(z_next))
    assert torch.all(torch.isfinite(log_det))

    loss = torch.sum(z_next**2) + torch.sum(log_det)
    loss.backward()

    assert z.grad is not None
    assert torch.all(torch.isfinite(z.grad))


def test_normalizing_flow_prior_sampling() -> None:
    prior_planar = NormalizingFlowPrior(dim=4, num_layers=3, flow_type="planar")
    samples, log_probs = prior_planar.sample_with_log_prob(batch_size=16)

    assert samples.shape == (16, 4)
    assert log_probs.shape == (16,)
    assert torch.all(torch.isfinite(samples))
    assert torch.all(torch.isfinite(log_probs))

    prior_radial = NormalizingFlowPrior(dim=4, num_layers=2, flow_type="radial")
    samples_r, log_probs_r = prior_radial.sample_with_log_prob(batch_size=16)

    assert samples_r.shape == (16, 4)
    assert log_probs_r.shape == (16,)
    assert torch.all(torch.isfinite(samples_r))
    assert torch.all(torch.isfinite(log_probs_r))


def test_invalid_flow_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown flow_type"):
        NormalizingFlowPrior(dim=4, flow_type="unsupported_type")
