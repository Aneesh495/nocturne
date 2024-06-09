"""Unit tests for adaptive Dormand-Prince and symplectic Verlet ODE solvers."""

from __future__ import annotations

import math
import pytest
import torch

from nocturne.models.vldm.neural_ode import (
    AdaptiveDormandPrince54Integrator,
    ContinuousLatentODESolver,
    ODESolverStats,
    SymplecticVerletIntegrator,
)


def test_dopri5_exponential_decay_accuracy() -> None:
    alpha = 1.5
    z0 = torch.tensor([2.0, -1.0])

    def decay_field(z: torch.Tensor) -> torch.Tensor:
        return -alpha * z

    integrator = AdaptiveDormandPrince54Integrator(rtol=1e-5, atol=1e-7)
    t_end = 2.0
    z_final, stats = integrator.integrate(decay_field, z0, (0.0, t_end))

    assert isinstance(stats, ODESolverStats)
    assert stats.accepted_steps > 0
    assert stats.rejected_steps >= 0

    # Analytic solution: z(t) = z_0 * exp(-alpha * t)
    analytic_final = z0 * math.exp(-alpha * t_end)
    assert torch.allclose(z_final, analytic_final, atol=1e-4, rtol=1e-4)


def test_symplectic_verlet_energy_conservation() -> None:
    k = 4.0  # Spring constant
    m_inv = 1.0  # Inverse mass
    dt = 0.01

    def spring_force(q: torch.Tensor) -> torch.Tensor:
        return -k * q

    def total_energy(q: torch.Tensor, p: torch.Tensor) -> float:
        kinetic = 0.5 * m_inv * torch.sum(p**2).item()
        potential = 0.5 * k * torch.sum(q**2).item()
        return kinetic + potential

    verlet = SymplecticVerletIntegrator(step_size=dt)
    q = torch.tensor([1.5])
    p = torch.tensor([0.0])

    initial_e = total_energy(q, p)

    # Integrate for 200 steps (2 seconds)
    for _ in range(200):
        q, p = verlet.step(q, p, force_fn=spring_force, mass_inv=m_inv)

    final_e = total_energy(q, p)

    # Symplectic Verlet guarantees bounded energy oscillation (no energy dissipation)
    rel_energy_drift = abs(final_e - initial_e) / initial_e
    assert rel_energy_drift < 1e-3  # Less than 0.1% energy fluctuation


def test_continuous_latent_ode_solver_rollout_and_gradients() -> None:
    solver = ContinuousLatentODESolver(solver_type="rk4", bin_width_s=0.02)

    # Parameterized linear velocity: dz/dt = A z
    A = torch.nn.Parameter(torch.randn(4, 4) * 0.1)

    def linear_field(z: torch.Tensor) -> torch.Tensor:
        return z @ A.T

    z0 = torch.randn(8, 4, requires_grad=True)  # [B=8, d=4]
    traj = solver.rollout(linear_field, z0, num_steps=15)

    assert traj.shape == (8, 15, 4)
    loss = torch.sum(traj**2)
    loss.backward()

    # Verify gradients flow cleanly back to both z0 and parameters A
    assert z0.grad is not None
    assert A.grad is not None
    assert torch.all(torch.isfinite(z0.grad))
    assert torch.all(torch.isfinite(A.grad))
