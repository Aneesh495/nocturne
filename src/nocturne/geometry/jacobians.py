"""Local linearization, Jacobian eigenspectra, and dynamical stability classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List
import numpy as np
import torch


@dataclass(frozen=True)
class JacobianDiagnostics:
    """Spectral analysis and topological stability classification of a local fixed point."""

    jacobian: np.ndarray             # [D, D] Jacobian matrix J = df/dz
    eigenvalues: np.ndarray          # [D] complex eigenvalues
    max_real_eigenvalue: float       # max(Re(lambda))
    is_stable: bool                  # True if max(Re(lambda)) < 0
    topological_type: str            # 'stable_node', 'stable_focus', 'saddle', 'unstable_node', 'unstable_focus'


class JacobianAnalyzer:
    """Computes exact vector field Jacobians and classifies fixed point topologies."""

    @staticmethod
    def analyze_fixed_point(
        vector_field: Callable[[torch.Tensor], torch.Tensor],
        fixed_point_location: np.ndarray,
        eps: float = 1e-4,
    ) -> JacobianDiagnostics:
        """Compute Jacobian and eigenspectrum at fixed_point_location.

        Args:
            vector_field: Vector field mapping z -> dz/dt.
            fixed_point_location: Coordinate vector z* of shape [D].
            eps: Epsilon threshold for distinguishing pure real vs imaginary parts.

        Returns:
            JacobianDiagnostics detailing eigenspectrum and stability classification.
        """
        z_torch = torch.as_tensor(fixed_point_location, dtype=torch.float32)

        # Compute exact Jacobian [D, D] via autograd functional
        def single_eval(x):
            return vector_field(x.unsqueeze(0)).squeeze(0)

        J_torch = torch.autograd.functional.jacobian(single_eval, z_torch)
        J = J_torch.detach().cpu().numpy()

        eigenvalues = np.linalg.eigvals(J)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)

        max_real = float(np.max(real_parts))
        min_real = float(np.min(real_parts))
        max_imag = float(np.max(np.abs(imag_parts)))

        has_oscillation = max_imag > eps
        is_stable = max_real < 0.0

        if is_stable:
            top_type = "stable_focus" if has_oscillation else "stable_node"
        elif min_real < 0.0 and max_real > 0.0:
            top_type = "saddle"
        else:
            top_type = "unstable_focus" if has_oscillation else "unstable_node"

        return JacobianDiagnostics(
            jacobian=J,
            eigenvalues=eigenvalues,
            max_real_eigenvalue=max_real,
            is_stable=is_stable,
            topological_type=top_type,
        )
