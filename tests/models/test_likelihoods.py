"""Unit tests for PoissonLoss PyTorch module and autograd gradient checks."""

import math
import numpy as np
import pytest
import torch

from nocturne.models.likelihoods import PoissonLoss


def test_poisson_loss_hand_value():
    dt = 0.02
    loss_fn = PoissonLoss(bin_width_s=dt, include_log_factorial=True)

    # 1 bin, 1 neuron
    # y = 2, rate = 100 Hz -> mu = 2.0
    rates = torch.tensor([[100.0]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[2.0]], dtype=torch.float64)

    loss = loss_fn.forward_from_rates(rates, targets)
    # Expected NLL = -ln P(2 | 2) = -(-1.3068528) = 1.3068528
    expected = -(math.log(2.0 * math.exp(-2.0)))
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_poisson_loss_gradient_exactness():
    dt = 0.02
    loss_fn = PoissonLoss(bin_width_s=dt, include_log_factorial=False)

    # For log_rates eta:
    # NLL = exp(eta + ln dt) - y * (eta + ln dt) = mu - y * eta - y * ln dt
    # d(NLL)/d(eta) = mu - y
    log_rates = torch.tensor([[math.log(50.0), math.log(150.0)]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[3.0, 1.0]], dtype=torch.float64)

    loss = loss_fn.forward_from_log_rates(log_rates, targets)
    loss.backward()

    # Expected gradients: (mu - y) / numel
    mu = torch.exp(log_rates + math.log(dt))
    expected_grad = (mu - targets) / 2.0  # mean reduction over 2 elements

    torch.testing.assert_close(log_rates.grad, expected_grad, rtol=1e-7, atol=1e-7)


def test_poisson_loss_masking():
    dt = 0.01
    loss_fn = PoissonLoss(bin_width_s=dt)

    rates = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    loss_masked = loss_fn.forward_from_rates(rates, targets, mask=mask)

    # Hand compute mean over elements [0, 0] and [1, 1]
    mu00 = 10.0 * dt
    nll00 = mu00 - 1.0 * math.log(mu00)
    mu11 = 40.0 * dt
    nll11 = mu11 - 4.0 * math.log(mu11)
    expected_mean = (nll00 + nll11) / 2.0

    assert loss_masked.item() == pytest.approx(expected_mean, rel=1e-5)
