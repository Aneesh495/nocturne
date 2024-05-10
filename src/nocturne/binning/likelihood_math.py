"""Reference Poisson likelihood calculations, exact log-factorial lookup tables, and deviance math."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import scipy.special


class LogFactorialLUT:
    """Precomputed exact log-factorial lookup table with high-order asymptotic fallback.

    Table stores exact values of ln(k!) for k in [0, MAX_LUT_K].
    For k > MAX_LUT_K, Ramanujan's high-precision asymptotic approximation is evaluated.
    """

    MAX_LUT_K: int = 1024

    def __init__(self):
        self._lut = np.zeros(self.MAX_LUT_K + 1, dtype=np.float64)
        acc = 0.0
        self._lut[0] = 0.0
        for k in range(1, self.MAX_LUT_K + 1):
            acc += math.log(k)
            self._lut[k] = acc

    def log_factorial(self, k: int | np.ndarray) -> float | np.ndarray:
        """Compute ln(k!) for non-negative integer or array of integers."""
        if isinstance(k, (int, np.integer)):
            if k < 0:
                raise ValueError(f"k must be non-negative, got {k}")
            if k <= self.MAX_LUT_K:
                return self._lut[k]
            return self._ramanujan_stirling(float(k))

        arr = np.asarray(k)
        if not np.issubdtype(arr.dtype, np.integer):
            if not np.all(np.equal(np.mod(arr, 1), 0)):
                raise ValueError("k array must contain non-negative integers")
            arr = arr.astype(np.int64)

        if np.any(arr < 0):
            raise ValueError("All elements of k must be non-negative")

        out = np.zeros(arr.shape, dtype=np.float64)
        lut_mask = arr <= self.MAX_LUT_K
        out[lut_mask] = self._lut[arr[lut_mask]]

        large_mask = ~lut_mask
        if np.any(large_mask):
            large_k = arr[large_mask].astype(np.float64)
            out[large_mask] = self._ramanujan_stirling_vec(large_k)

        return out

    @staticmethod
    def _ramanujan_stirling(k: float) -> float:
        """Ramanujan's refined Stirling formula: ln(k!) approx k*ln(k) - k + (1/6)*ln(8k^3 + 4k^2 + k + 1/30) + (1/2)*ln(pi)."""
        return (
            k * math.log(k)
            - k
            + (1.0 / 6.0) * math.log(8.0 * (k**3) + 4.0 * (k**2) + k + (1.0 / 30.0))
            + 0.5 * math.log(math.pi)
        )

    @staticmethod
    def _ramanujan_stirling_vec(k: np.ndarray) -> np.ndarray:
        """Vectorized Ramanujan Stirling formula."""
        return (
            k * np.log(k)
            - k
            + (1.0 / 6.0) * np.log(8.0 * (k**3) + 4.0 * (k**2) + k + (1.0 / 30.0))
            + 0.5 * math.log(math.pi)
        )


_DEFAULT_LUT = LogFactorialLUT()


def get_log_factorial(k: int | np.ndarray) -> float | np.ndarray:
    """Convenience function evaluating exact ln(k!)."""
    return _DEFAULT_LUT.log_factorial(k)


def poisson_log_likelihood(
    counts: np.ndarray,
    rates_hz: np.ndarray,
    bin_width_s: float,
    mask: Optional[np.ndarray] = None,
    include_log_factorial: bool = True,
    eps: float = 1e-12,
) -> float:
    """Compute point-process Poisson log-likelihood.

    Formula:
        mu = max(rates_hz * bin_width_s, eps)
        ll = sum_{valid} [ y * ln(mu) - mu - (ln(y!) if include_log_factorial else 0) ]

    Args:
        counts: Observed integer spike counts (shape [..., N]).
        rates_hz: Predicted firing rates in spikes/second (Hz).
        bin_width_s: Temporal bin duration in seconds.
        mask: Optional boolean mask where True indicates valid evaluated elements.
        include_log_factorial: Whether to include the exact ln(y!) normalizer.
        eps: Lower bound for rate expectation to prevent log(0).

    Returns:
        Total log-likelihood summed over valid evaluated elements.
    """
    if bin_width_s <= 0.0:
        raise ValueError(f"bin_width_s must be positive, got {bin_width_s}")

    y = np.asarray(counts, dtype=np.float64)
    r = np.asarray(rates_hz, dtype=np.float64)

    if y.shape != r.shape:
        raise ValueError(f"counts shape {y.shape} does not match rates shape {r.shape}")

    # Expected count per bin: mu = r * Delta_t
    mu = np.maximum(r * bin_width_s, eps)

    # Elementwise Poisson log-likelihood: y * ln(mu) - mu
    # For y == 0: 0 * ln(mu) = 0
    elem_ll = y * np.log(mu) - mu

    if include_log_factorial:
        ln_fact = get_log_factorial(counts)
        elem_ll = elem_ll - ln_fact

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != y.shape:
            raise ValueError(f"mask shape {m.shape} does not match counts shape {y.shape}")
        elem_ll = elem_ll[m]

    return float(np.sum(elem_ll))


def poisson_deviance(
    counts: np.ndarray,
    rates_hz: np.ndarray,
    bin_width_s: float,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """Compute Poisson deviance: 2 * sum [ y * ln(y / mu) - (y - mu) ].

    Deviance measures goodness-of-fit against the saturated model (where mu = y).
    For y == 0, the term y * ln(y / mu) evaluates to 0.
    """
    if bin_width_s <= 0.0:
        raise ValueError(f"bin_width_s must be positive, got {bin_width_s}")

    y = np.asarray(counts, dtype=np.float64)
    r = np.asarray(rates_hz, dtype=np.float64)
    mu = np.maximum(r * bin_width_s, eps)

    elem_dev = np.zeros_like(y)

    # Where y > 0: 2 * (y * ln(y / mu) - (y - mu))
    pos_mask = y > 0
    elem_dev[pos_mask] = 2.0 * (y[pos_mask] * np.log(y[pos_mask] / mu[pos_mask]) - (y[pos_mask] - mu[pos_mask]))

    # Where y == 0: 2 * (0 - (0 - mu)) = 2 * mu
    zero_mask = y == 0
    elem_dev[zero_mask] = 2.0 * mu[zero_mask]

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        elem_dev = elem_dev[m]

    return float(np.sum(elem_dev))


def poisson_bits_per_spike(
    counts: np.ndarray,
    model_rates_hz: np.ndarray,
    reference_rates_hz: np.ndarray,
    bin_width_s: float,
    mask: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Compute held-out information gain in bits per spike relative to a reference rate prior.

    Formula:
        BPS = (LL_model - LL_reference) / (ln(2) * total_spikes)

    Note: The ln(y!) terms in LL_model and LL_reference cancel out exactly.
    If total evaluated spikes is 0, returns None to represent an undefined metric
    rather than returning infinity or NaN.
    """
    y = np.asarray(counts, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        total_spikes = float(np.sum(y[m]))
    else:
        total_spikes = float(np.sum(y))

    if total_spikes <= 0.0:
        # Zero-spike denominator contract: return None
        return None

    # Compute un-normalized Poisson log likelihoods (without log factorial, since they cancel)
    ll_model = poisson_log_likelihood(
        counts=counts,
        rates_hz=model_rates_hz,
        bin_width_s=bin_width_s,
        mask=mask,
        include_log_factorial=False,
    )
    ll_ref = poisson_log_likelihood(
        counts=counts,
        rates_hz=reference_rates_hz,
        bin_width_s=bin_width_s,
        mask=mask,
        include_log_factorial=False,
    )

    diff_nats = ll_model - ll_ref
    bps = diff_nats / (math.log(2.0) * total_spikes)
    return float(bps)
