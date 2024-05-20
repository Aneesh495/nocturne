"""Independent exact Kalman filter and Rauch-Tung-Striebel (RTS) smoother reference."""

from __future__ import annotations

from typing import Tuple
import numpy as np


class KalmanSmootherReference:
    """Exact Kalman filter and RTS backward smoother for linear Gaussian state space models.

    Equations:
        z_{t+1} = A * z_t + w_t,   w_t ~ N(0, Q)
        y_t     = C * z_t + v_t,   v_t ~ N(0, R)
        z_0     ~ N(mu_0, P_0)
    """

    def __init__(
        self,
        A: np.ndarray,
        Q: np.ndarray,
        C: np.ndarray,
        R: np.ndarray,
        mu_0: np.ndarray,
        P_0: np.ndarray,
    ):
        self.A = np.asarray(A, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.mu_0 = np.asarray(mu_0, dtype=np.float64)
        self.P_0 = np.asarray(P_0, dtype=np.float64)

        self.D = self.A.shape[0]
        self.N = self.C.shape[0]

    def filter_and_smooth(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute filtered and smoothed state expectations and covariances.

        Args:
            observations: 2D array of observations [T, N].

        Returns:
            Tuple of (smoothed_means [T, D], smoothed_covariances [T, D, D]).
        """
        Y = np.asarray(observations, dtype=np.float64)
        T = Y.shape[0]
        D = self.D
        A, Q, C, R = self.A, self.Q, self.C, self.R

        # Forward filtering
        mu_pred = np.zeros((T, D), dtype=np.float64)
        P_pred = np.zeros((T, D, D), dtype=np.float64)
        mu_filt = np.zeros((T, D), dtype=np.float64)
        P_filt = np.zeros((T, D, D), dtype=np.float64)

        for t in range(T):
            if t == 0:
                m_prior = self.mu_0
                P_prior = self.P_0
            else:
                m_prior = A @ mu_filt[t - 1]
                P_prior = A @ P_filt[t - 1] @ A.T + Q

            mu_pred[t] = m_prior
            P_pred[t] = P_prior

            # Innovation and Kalman gain
            y_pred = C @ m_prior
            innov = Y[t] - y_pred
            S = C @ P_prior @ C.T + R
            K = P_prior @ C.T @ np.linalg.inv(S)

            mu_filt[t] = m_prior + K @ innov
            P_filt[t] = (np.eye(D) - K @ C) @ P_prior

        # Backward RTS smoothing
        mu_smooth = np.zeros((T, D), dtype=np.float64)
        P_smooth = np.zeros((T, D, D), dtype=np.float64)

        mu_smooth[T - 1] = mu_filt[T - 1]
        P_smooth[T - 1] = P_filt[T - 1]

        for t in range(T - 2, -1, -1):
            P_pred_inv = np.linalg.inv(P_pred[t + 1])
            J = P_filt[t] @ A.T @ P_pred_inv

            mu_smooth[t] = mu_filt[t] + J @ (mu_smooth[t + 1] - mu_pred[t + 1])
            P_smooth[t] = P_filt[t] + J @ (P_smooth[t + 1] - P_pred[t + 1]) @ J.T

        return mu_smooth, P_smooth
