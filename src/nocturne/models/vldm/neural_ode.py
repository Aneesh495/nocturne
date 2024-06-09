"""Continuous-time neural ODE numerical integrators and solvers for latent state spaces.

Implements adaptive Dormand-Prince (RK45) step integration with embedded error estimation,
symplectic Verlet integration for energy-preserving Hamiltonian dynamics, and fixed-step RK4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ODESolverStats:
    """Diagnostic statistics from an adaptive numerical integration trajectory."""

    num_steps: int
    num_evals: int
    accepted_steps: int
    rejected_steps: int
    mean_step_size: float
    min_step_size: float
    max_step_size: float


class AdaptiveDormandPrince54Integrator:
    """Embedded Runge-Kutta 5(4) Dormand-Prince adaptive ODE solver.

    Provides 5th-order accurate trajectories with 4th-order embedded error
    estimation for automatic step-size control.
    """

    # Butcher tableau coefficients
    C = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0]

    A = [
        [],
        [1.0 / 5.0],
        [3.0 / 40.0, 9.0 / 40.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
        [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0],
        [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
        [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0],
    ]

    B5 = [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0]
    B4 = [
        5179.0 / 57600.0,
        0.0,
        7571.0 / 16695.0,
        393.0 / 640.0,
        -92097.0 / 339200.0,
        187.0 / 2100.0,
        1.0 / 40.0,
    ]

    def __init__(
        self,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        min_step: float = 1e-5,
        max_step: float = 0.5,
    ):
        self.rtol = rtol
        self.atol = atol
        self.safety = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.min_step = min_step
        self.max_step = max_step

    def integrate(
        self,
        vector_field: Callable[[torch.Tensor], torch.Tensor],
        z0: torch.Tensor,
        t_span: Tuple[float, float],
    ) -> Tuple[torch.Tensor, ODESolverStats]:
        """Integrate vector field from t_span[0] to t_span[1] with adaptive step size.

        Args:
            vector_field: Callable returning velocity dz/dt given z.
            z0: Initial latent tensor of shape [..., d].
            t_span: Tuple of (t_start, t_end).

        Returns:
            Tuple of (z_final, ODESolverStats).
        """
        t, t_end = t_span
        z = z0
        h = min(self.max_step, max(self.min_step, (t_end - t) / 10.0))

        accepted = 0
        rejected = 0
        evals = 0
        step_sizes: List[float] = []

        while t < t_end - 1e-9:
            if t + h > t_end:
                h = t_end - t

            # Stage 1
            k1 = vector_field(z)
            # Stage 2
            k2 = vector_field(z + h * (self.A[1][0] * k1))
            # Stage 3
            k3 = vector_field(z + h * (self.A[2][0] * k1 + self.A[2][1] * k2))
            # Stage 4
            k4 = vector_field(z + h * (self.A[3][0] * k1 + self.A[3][1] * k2 + self.A[3][2] * k3))
            # Stage 5
            k5 = vector_field(
                z + h * (self.A[4][0] * k1 + self.A[4][1] * k2 + self.A[4][2] * k3 + self.A[4][3] * k4)
            )
            # Stage 6
            k6 = vector_field(
                z + h * (
                    self.A[5][0] * k1 + self.A[5][1] * k2 + self.A[5][2] * k3
                    + self.A[5][3] * k4 + self.A[5][4] * k5
                )
            )

            # 5th-order candidate
            z_5 = z + h * (
                self.B5[0] * k1 + self.B5[2] * k3 + self.B5[3] * k4 + self.B5[4] * k5 + self.B5[5] * k6
            )
            # Stage 7 (FSAL: First Same As Last)
            k7 = vector_field(z_5)
            evals += 7

            # 4th-order candidate for error estimate
            z_4 = z + h * (
                self.B4[0] * k1 + self.B4[2] * k3 + self.B4[3] * k4
                + self.B4[4] * k5 + self.B4[5] * k6 + self.B4[6] * k7
            )

            # Truncation error
            err = torch.max(torch.abs(z_5 - z_4)).item()
            scale = self.atol + self.rtol * max(
                torch.max(torch.abs(z)).item(),
                torch.max(torch.abs(z_5)).item(),
            )
            err_ratio = max(1e-10, err / (scale + 1e-12))

            if err_ratio <= 1.0:
                # Accept step
                t += h
                z = z_5
                accepted += 1
                step_sizes.append(h)
                # Next step adaptation
                factor = min(self.max_factor, max(self.min_factor, self.safety * (1.0 / err_ratio) ** 0.2))
                h = min(self.max_step, max(self.min_step, h * factor))
            else:
                # Reject step
                rejected += 1
                factor = min(self.max_factor, max(self.min_factor, self.safety * (1.0 / err_ratio) ** 0.25))
                h = max(self.min_step, h * factor)

        stats = ODESolverStats(
            num_steps=accepted + rejected,
            num_evals=evals,
            accepted_steps=accepted,
            rejected_steps=rejected,
            mean_step_size=float(sum(step_sizes) / len(step_sizes)) if step_sizes else 0.0,
            min_step_size=float(min(step_sizes)) if step_sizes else 0.0,
            max_step_size=float(max(step_sizes)) if step_sizes else 0.0,
        )
        return z, stats


class SymplecticVerletIntegrator:
    """Symplectic velocity-Verlet numerical integrator for Hamiltonian neural dynamics.

    Preserves phase-space symplectic 2-form and total energy in conservative
    or lightly damped neural oscillator systems.
    """

    def __init__(self, step_size: float = 0.02):
        self.step_size = step_size

    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        force_fn: Callable[[torch.Tensor], torch.Tensor],
        mass_inv: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one symplectic Verlet update step.

        Equations:
            p_{t+1/2} = p_t + 0.5 * dt * F(q_t)
            q_{t+1}   = q_t + dt * mass_inv * p_{t+1/2}
            p_{t+1}   = p_{t+1/2} + 0.5 * dt * F(q_{t+1})

        Args:
            q: Generalized coordinate / position tensor [..., d/2].
            p: Generalized momentum tensor [..., d/2].
            force_fn: Negative potential gradient F(q) = -grad V(q).
            mass_inv: Inverse inertia / mass parameter.

        Returns:
            Tuple of updated (q_{t+1}, p_{t+1}).
        """
        dt = self.step_size

        # Half-step momentum
        f_q = force_fn(q)
        p_half = p + 0.5 * dt * f_q

        # Full-step position
        q_next = q + dt * mass_inv * p_half

        # Half-step momentum with new position force
        f_q_next = force_fn(q_next)
        p_next = p_half + 0.5 * dt * f_q_next

        return q_next, p_next


class ContinuousLatentODESolver(nn.Module):
    """Wrapper module integrating neural ODE vector fields across multi-step sequences."""

    def __init__(self, solver_type: str = "rk4", bin_width_s: float = 0.02):
        super().__init__()
        self.solver_type = solver_type
        self.bin_width_s = bin_width_s
        self.adaptive_dopri = AdaptiveDormandPrince54Integrator()
        self.symplectic = SymplecticVerletIntegrator(step_size=bin_width_s)

    def rollout(
        self,
        vector_field: Callable[[torch.Tensor], torch.Tensor],
        z0: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Roll out continuous latent state across num_steps uniform bin intervals.

        Args:
            vector_field: Velocity function dz/dt = f(z).
            z0: Initial latent state tensor [B, d].
            num_steps: Number of forward integration time steps T.

        Returns:
            Latent trajectory tensor [B, num_steps, d].
        """
        trajectory: List[torch.Tensor] = [z0]
        z_curr = z0
        dt = self.bin_width_s

        for step in range(1, num_steps):
            if self.solver_type == "rk4":
                # Fixed 4th-order Runge-Kutta
                k1 = vector_field(z_curr)
                k2 = vector_field(z_curr + 0.5 * dt * k1)
                k3 = vector_field(z_curr + 0.5 * dt * k2)
                k4 = vector_field(z_curr + dt * k3)
                z_curr = z_curr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            elif self.solver_type == "dopri5":
                t_start = (step - 1) * dt
                t_end = step * dt
                z_curr, _ = self.adaptive_dopri.integrate(
                    vector_field=vector_field,
                    z0=z_curr,
                    t_span=(t_start, t_end),
                )
            else:
                # Default Euler step
                z_curr = z_curr + dt * vector_field(z_curr)

            trajectory.append(z_curr)

        return torch.stack(trajectory, dim=1)
