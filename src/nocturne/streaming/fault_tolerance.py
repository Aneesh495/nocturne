"""Fault tolerance, malformed packet handling, gap recovery, and mid-stream reset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from nocturne.streaming.streamer import OnlineCausalStreamer


@dataclass(frozen=True)
class FaultToleranceReport:
    """Audit report for streaming resilience under faults, corruptions, and interruptions."""

    gap_cases_tested: int
    gap_cases_recovered: int
    malformed_cases_tested: int
    malformed_cases_handled: int
    interruption_cases_tested: int
    interruption_cases_resumed: int
    is_resilient: bool


class StreamFaultHandler:
    """Validates, sanitizes, and recovers online streaming sessions under network/hardware faults."""

    def __init__(self, streamer: OnlineCausalStreamer):
        self.streamer = streamer
        self.expected_obs_dim = streamer.obs_dim
        self.last_step_index: int = 0

    def sanitize_and_push(
        self,
        chunk: torch.Tensor,
        step_index: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Validate, fill detected gaps, sanitize invalid values, and push to streamer.

        Args:
            chunk: Input candidate tensor.
            step_index: Optional sequential packet index for gap detection.

        Returns:
            Tuple of (predicted_rates_or_None, diagnostic_info).
        """
        diagnostics = {
            "gap_detected": False,
            "gap_length": 0,
            "sanitized_nans": False,
            "sanitized_negatives": False,
            "rejected": False,
        }

        # Step 1: Detect temporal packet gap
        delta_T = chunk.shape[1] if chunk.ndim == 3 else (chunk.shape[0] if chunk.ndim == 2 else 0)
        if step_index is not None and self.last_step_index > 0:
            expected_next = self.last_step_index + 1
            if step_index > expected_next:
                gap = step_index - expected_next
                diagnostics["gap_detected"] = True
                diagnostics["gap_length"] = gap
                # Fill gap with zero-count filler
                filler = torch.zeros((1, gap, self.expected_obs_dim), dtype=torch.float32, device=self.streamer.device)
                self.streamer.push_chunk(filler)

        # Step 2: Dimension check
        if chunk.ndim not in (2, 3):
            diagnostics["rejected"] = True
            return None, diagnostics

        N = chunk.shape[-1]
        if N != self.expected_obs_dim:
            diagnostics["rejected"] = True
            return None, diagnostics

        if delta_T == 0:
            diagnostics["rejected"] = True
            return None, diagnostics

        # Step 3: Sanitize Non-finite & Negative values
        sanitized = chunk.clone()
        if not torch.isfinite(sanitized).all():
            diagnostics["sanitized_nans"] = True
            sanitized = torch.nan_to_num(sanitized, nan=0.0, posinf=50.0, neginf=0.0)

        if (sanitized < 0.0).any():
            diagnostics["sanitized_negatives"] = True
            sanitized = torch.clamp(sanitized, min=0.0)

        # Step 4: Push to streamer
        res = self.streamer.push_chunk(sanitized)
        if step_index is not None:
            self.last_step_index = step_index + delta_T - 1

        return res.predicted_rates_hz, diagnostics

    @classmethod
    def audit_fault_resilience(
        cls,
        streamer: OnlineCausalStreamer,
        num_gap_cases: int = 50,
        num_malformed_cases: int = 50,
        num_interruption_cases: int = 50,
    ) -> FaultToleranceReport:
        """Run standard stress suite evaluating 150 total fault cases (Acceptance Gate BA11)."""
        handler = cls(streamer)

        # 1. Test Gap recovery (50 cases)
        gap_recovered = 0
        for i in range(num_gap_cases):
            streamer.reset()
            # Push valid chunk at step 0
            handler.last_step_index = 0
            c1 = torch.ones((1, 5, streamer.obs_dim))
            handler.sanitize_and_push(c1, step_index=1)
            # Push next chunk with gap (e.g. step_index jumps by 4)
            c2 = torch.ones((1, 5, streamer.obs_dim))
            out, diag = handler.sanitize_and_push(c2, step_index=10)
            if diag["gap_detected"] and out is not None:
                gap_recovered += 1

        # 2. Test Malformed handling (50 cases)
        malformed_handled = 0
        for i in range(num_malformed_cases):
            streamer.reset()
            case_type = i % 4
            if case_type == 0:
                # NaN injection
                bad_chunk = torch.ones((1, 5, streamer.obs_dim))
                bad_chunk[0, 2, 0] = float("nan")
                out, diag = handler.sanitize_and_push(bad_chunk)
                if diag["sanitized_nans"] and out is not None and torch.isfinite(out).all():
                    malformed_handled += 1
            elif case_type == 1:
                # Negative counts
                bad_chunk = torch.ones((1, 5, streamer.obs_dim)) * -10.0
                out, diag = handler.sanitize_and_push(bad_chunk)
                if diag["sanitized_negatives"] and out is not None and (out >= 0.0).all():
                    malformed_handled += 1
            elif case_type == 2:
                # Channel dimension mismatch
                bad_chunk = torch.ones((1, 5, streamer.obs_dim + 4))
                out, diag = handler.sanitize_and_push(bad_chunk)
                if diag["rejected"] and out is None:
                    malformed_handled += 1
            else:
                # Empty chunk
                bad_chunk = torch.zeros((1, 0, streamer.obs_dim))
                out, diag = handler.sanitize_and_push(bad_chunk)
                if diag["rejected"] and out is None:
                    malformed_handled += 1

        # 3. Test Interruption and Recovery (50 cases)
        resumed_count = 0
        for i in range(num_interruption_cases):
            streamer.reset()
            chunk_a = torch.randint(0, 4, (1, 10, streamer.obs_dim), dtype=torch.float32)
            streamer.push_chunk(chunk_a)

            # Checkpoint
            state = streamer.checkpoint()

            # Create fresh streamer and restore
            new_streamer = OnlineCausalStreamer(
                model=streamer.model,
                obs_dim=streamer.obs_dim,
                max_window_bins=streamer.max_window_bins,
                device=streamer.device,
            )
            new_streamer.restore(state)

            # Push subsequent chunk to both and verify outputs match
            chunk_b = torch.randint(0, 4, (1, 5, streamer.obs_dim), dtype=torch.float32)
            out_orig = streamer.push_chunk(chunk_b)
            out_resumed = new_streamer.push_chunk(chunk_b)

            if torch.allclose(out_orig.predicted_rates_hz, out_resumed.predicted_rates_hz, atol=1e-5):
                resumed_count += 1

        is_resilient = (
            gap_recovered == num_gap_cases
            and malformed_handled == num_malformed_cases
            and resumed_count == num_interruption_cases
        )

        return FaultToleranceReport(
            gap_cases_tested=num_gap_cases,
            gap_cases_recovered=gap_recovered,
            malformed_cases_tested=num_malformed_cases,
            malformed_cases_handled=malformed_handled,
            interruption_cases_tested=num_interruption_cases,
            interruption_cases_resumed=resumed_count,
            is_resilient=is_resilient,
        )
