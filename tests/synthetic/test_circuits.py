"""Unit tests for synthetic dynamical circuits, analytic eigenspectra, and fixed points."""

import math
import numpy as np
import pytest

from nocturne.synthetic.bistable_memory import BistableMemoryCircuit
from nocturne.synthetic.driven_recurrent import DrivenRecurrentCircuit
from nocturne.synthetic.linear_oscillator import LinearDampedOscillator
from nocturne.synthetic.nonlinear_oscillator import NonlinearLimitCycleOscillator


def test_linear_damped_oscillator_analytic_solution():
    gamma = 0.4
    omega = 2.0 * math.pi  # 1 Hz oscillation
    circuit = LinearDampedOscillator(damping_gamma=gamma, frequency_omega=omega)

    # Check analytic eigenvalues
    ev1, ev2 = circuit.analytic_eigenvalues
    assert ev1.real == pytest.approx(-gamma)
    assert ev1.imag == pytest.approx(omega)
    assert ev2.real == pytest.approx(-gamma)
    assert ev2.imag == pytest.approx(-omega)

    # Compare RK4 integration against analytic solution
    z0 = np.array([1.5, -0.5])
    duration = 2.0
    dt = 0.001
    traj = circuit.integrate_rk4(z0, duration_s=duration, dt_s=dt)

    analytic_states = circuit.analytic_solution(z0, traj.timestamps)
    # Numerical integration should match closed form within 1e-4
    np.testing.assert_allclose(traj.states, analytic_states, atol=1e-3)


def test_nonlinear_limit_cycle_oscillator_attractor():
    mu = 1.0  # Limit cycle at radius sqrt(1.0) = 1.0
    circuit = NonlinearLimitCycleOscillator(mu=mu, omega=2.0 * math.pi, b=0.0)
    assert circuit.limit_cycle_radius == pytest.approx(1.0)

    # Start inside limit cycle (r = 0.4)
    traj_inside = circuit.integrate_rk4(np.array([0.4, 0.0]), duration_s=4.0, dt_s=0.002)
    # Final state radius should approach 1.0
    final_r_inside = np.linalg.norm(traj_inside.states[-1])
    assert final_r_inside == pytest.approx(1.0, abs=0.05)

    # Start outside limit cycle (r = 1.8)
    traj_outside = circuit.integrate_rk4(np.array([1.8, 0.0]), duration_s=4.0, dt_s=0.002)
    final_r_outside = np.linalg.norm(traj_outside.states[-1])
    assert final_r_outside == pytest.approx(1.0, abs=0.05)

    # Test Jacobian against finite differences
    z_test = np.array([0.8, -0.6])
    J_analytic = circuit.jacobian_at(z_test)
    eps = 1e-6
    J_num = np.zeros((2, 2))
    for i in range(2):
        dz = np.zeros(2)
        dz[i] = eps
        f_plus = circuit.vector_field(z_test + dz)
        f_minus = circuit.vector_field(z_test - dz)
        J_num[:, i] = (f_plus - f_minus) / (2.0 * eps)
    np.testing.assert_allclose(J_analytic, J_num, rtol=1e-5, atol=1e-5)


def test_bistable_memory_circuit_attractors():
    alpha = 2.0
    circuit = BistableMemoryCircuit(alpha=alpha, decay2=1.0)

    # Fixed points have zero velocity
    fp_a, fp_b = circuit.stable_fixed_points
    assert np.linalg.norm(circuit.vector_field(fp_a)) == pytest.approx(0.0)
    assert np.linalg.norm(circuit.vector_field(fp_b)) == pytest.approx(0.0)

    saddle = circuit.saddle_fixed_point
    assert np.linalg.norm(circuit.vector_field(saddle)) == pytest.approx(0.0)

    # Jacobians at stable fixed points have negative real eigenvalues
    J_a = circuit.jacobian_at(fp_a)
    eig_a = np.linalg.eigvals(J_a)
    assert np.all(eig_a < 0)  # Stable node

    # Trajectory starting near A stays in basin A without input
    traj_a = circuit.integrate_rk4(np.array([0.8, 0.1]), duration_s=2.0, dt_s=0.005)
    assert traj_a.states[-1, 0] == pytest.approx(1.0, abs=0.02)


def test_driven_recurrent_circuit():
    circuit = DrivenRecurrentCircuit(latent_dim=4, input_dim=2, tau=0.05, seed=123)

    z0 = np.array([0.2, -0.1, 0.5, 0.0])
    u = np.array([1.0, -0.5])

    # Check finite difference Jacobian
    J_analytic = circuit.jacobian_at(z0)
    eps = 1e-6
    J_num = np.zeros((4, 4))
    for i in range(4):
        dz = np.zeros(4)
        dz[i] = eps
        f_plus = circuit.vector_field(z0 + dz, u)
        f_minus = circuit.vector_field(z0 - dz, u)
        J_num[:, i] = (f_plus - f_minus) / (2.0 * eps)
    np.testing.assert_allclose(J_analytic, J_num, rtol=1e-5, atol=1e-5)
