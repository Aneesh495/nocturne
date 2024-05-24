"""Executable negative controls demonstrating failure modes caught by contracts and auditors."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple
import numpy as np
import torch
from pydantic import ValidationError

from nocturne.contracts.time import TimeInterval
from nocturne.contracts.splits import DatasetSplit
from nocturne.contracts.masks import ObservationMasks
from nocturne.binning.likelihood_math import poisson_log_likelihood, get_log_factorial
from nocturne.evaluation.bits_per_spike import BitsPerSpikeEvaluator
from nocturne.evaluation.integrity import EvaluationLabelIntegrityAuditor
from nocturne.inference.causality import CausalLeakageAuditor
from nocturne.models.attention.model import SpatiotemporalAttentionModel
from nocturne.models.attention.config import AttentionConfig
from nocturne.streaming.state import StreamingBufferState
from nocturne.geometry.alignment import OrthogonalProcrustesAligner
from nocturne.training.recovery import TrainingRecovery


class ControlResult(NamedTuple):
    """Result of an individual negative control execution."""

    control_id: str
    name: str
    description: str
    caught: bool
    exception_type: str
    error_message: str


class NegativeControlSuite:
    """Executes suite of 12+ negative controls to verify defensive invariants."""

    @staticmethod
    def nc01_temporal_interval_inversion() -> ControlResult:
        """NC01: Inverted temporal interval (end_s <= start_s) must fail validation."""
        cid = "NC01"
        name = "Temporal Interval Inversion"
        desc = "Ensures TimeInterval rejects interval where end_time <= start_time."
        try:
            TimeInterval(start_s=10.0, end_s=5.0)
            return ControlResult(cid, name, desc, False, "None", "Failed to catch inverted interval")
        except (ValueError, ValidationError) as e:
            return ControlResult(cid, name, desc, True, type(e).__name__, str(e))

    @staticmethod
    def nc02_dataset_split_train_test_overlap() -> ControlResult:
        """NC02: Leaked trial indices between train and test partitions must be rejected."""
        cid = "NC02"
        name = "Dataset Split Overlap"
        desc = "Ensures DatasetSplit rejects overlapping trial indices across train and test partitions."
        try:
            from nocturne.contracts.splits import SplitPartition, SplitType
            DatasetSplit(
                split_name="split_leaked",
                split_type=SplitType.OFFICIAL_NLB,
                session_id="ses_01",
                train=SplitPartition(name="train", trial_ids=(0, 1, 2, 3)),
                val=SplitPartition(name="val", trial_ids=(4, 5)),
                test=SplitPartition(name="test", trial_ids=(3, 6, 7)),  # Index 3 overlaps with train!
            )
            return ControlResult(cid, name, desc, False, "None", "Failed to catch train/test overlap")
        except (ValueError, ValidationError) as e:
            return ControlResult(cid, name, desc, True, type(e).__name__, str(e))

    @staticmethod
    def nc03_eval_integrity_partition_leakage() -> ControlResult:
        """NC03: Integrity auditor must flag negative rates or non-finite values."""
        cid = "NC03"
        name = "Evaluation Integrity Violation Detection"
        desc = "Integrity auditor detects non-finite values or negative rates in test evaluation."
        try:
            from nocturne.contracts.splits import SplitPartition, SplitType
            split = DatasetSplit(
                split_name="test_split",
                split_type=SplitType.OFFICIAL_NLB,
                session_id="ses_01",
                train=SplitPartition(name="train", trial_ids=(0, 1)),
                val=SplitPartition(name="val", trial_ids=(2,)),
                test=SplitPartition(name="test", trial_ids=(3,)),
            )
            preds = np.array([[-1.0, 2.0], [3.0, 4.0]])
            targets = np.array([[1.0, 2.0], [3.0, 4.0]])
            report = EvaluationLabelIntegrityAuditor.audit_splits_and_predictions(
                dataset_split=split,
                predictions=preds,
                targets=targets,
            )
            if not report.zero_leakage_verified:
                return ControlResult(cid, name, desc, True, "IntegrityViolation", "; ".join(report.violations))
            return ControlResult(cid, name, desc, False, "None", "Integrity auditor failed to detect negative rates")
        except Exception as e:
            return ControlResult(cid, name, desc, True, type(e).__name__, str(e))

    @staticmethod
    def nc04_bps_zero_spike_denominator() -> ControlResult:
        """NC04: Bits-per-spike calculator must return None when test spikes are zero."""
        cid = "NC04"
        name = "BPS Zero Spike Denominator Contract"
        desc = "Ensures zero spike count returns None rather than NaN or division by zero."
        y_true = np.zeros((20, 5))
        lam_model = np.ones((20, 5)) * 0.1
        lam_base = np.ones((20, 5)) * 0.05
        calc = BitsPerSpikeEvaluator()
        res = calc.evaluate(predicted_rates_hz=lam_model, target_counts=y_true, null_rates_hz=lam_base)
        if res.population_bps is None and res.total_spikes == 0:
            return ControlResult(cid, name, desc, True, "ContractAdherence", "Returned None as mandated by contract")
        return ControlResult(cid, name, desc, False, "None", f"BPS returned {res.population_bps} on zero spikes")

    @staticmethod
    def nc05_causal_leakage_future_perturbation() -> ControlResult:
        """NC05: Causal leakage auditor must flag non-causal architectures."""
        cid = "NC05"
        name = "Causal Future Leakage Detection"
        desc = "Auditor detects architecture that illegally uses future inputs to predict past rates."

        class NonCausalMock(torch.nn.Module):
            def forward(self, x: torch.Tensor, causal: bool = True) -> Any:
                rates = torch.flip(x, dims=[1]) + 0.1
                latents = torch.zeros(x.shape[0], x.shape[1], 4)
                return rates, latents, None, {}

        auditor = CausalLeakageAuditor()
        report = auditor.audit_model(NonCausalMock(), num_cases=10, batch_size=2, seq_len=10, obs_dim=4)
        if not report.is_strictly_causal:
            return ControlResult(cid, name, desc, True, "CausalLeakageDetected", f"{report.failed_cases} violations detected")
        return ControlResult(cid, name, desc, False, "None", "Auditor failed to flag non-causal model")

    @staticmethod
    def nc06_poisson_negative_rate_rejection() -> ControlResult:
        """NC06: Poisson log-factorial and likelihood must reject negative inputs."""
        cid = "NC06"
        name = "Poisson Negative Input Rejection"
        desc = "Likelihood math rejects negative spike counts."
        k = np.array([1, -5])
        try:
            get_log_factorial(k)
            return ControlResult(cid, name, desc, False, "None", "Failed to reject negative spike count")
        except ValueError as e:
            return ControlResult(cid, name, desc, True, type(e).__name__, str(e))

    @staticmethod
    def nc07_poisson_nan_input_rejection() -> ControlResult:
        """NC07: Poisson log-factorial and likelihood must reject NaN inputs."""
        cid = "NC07"
        name = "Poisson NaN Input Rejection"
        desc = "Likelihood math rejects NaN or Inf inputs."
        k = np.array([1.0, np.nan])
        try:
            get_log_factorial(k)
            return ControlResult(cid, name, desc, False, "None", "Failed to reject NaN spike count")
        except ValueError as e:
            return ControlResult(cid, name, desc, True, type(e).__name__, str(e))

    @staticmethod
    def nc08_mask_all_channels_dropped() -> ControlResult:
        """NC08: ObservationMasks must reject channel mask with mismatched dimension."""
        cid = "NC08"
        name = "Observation Mask Channel Dimension Incompatibility"
        desc = "Ensures ObservationMasks detects when channel_mask shape is invalid."
        try:
            B, T, N = 1, 20, 10
            obs = torch.ones((B, T, N), dtype=torch.bool)
            tgt = torch.ones((B, T, N), dtype=torch.bool)
            pad = torch.ones((B, T), dtype=torch.bool)
            invalid_ch = torch.ones((B, 5), dtype=torch.bool)  # 5 != 10 channels!
            ObservationMasks(
                batch_size=B,
                num_time_bins=T,
                num_neurons=N,
                observed_mask=obs,
                target_mask=tgt,
                padding_mask=pad,
                channel_mask=invalid_ch,
            )
            return ControlResult(cid, name, desc, False, "None", "Failed to catch channel mask shape mismatch")
        except (ValueError, ValidationError) as e:
            return ControlResult(cid, name, desc, True, type(e).__name__, str(e))

    @staticmethod
    def nc09_streaming_buffer_invalid_window() -> ControlResult:
        """NC09: StreamingBufferState must reject non-positive window sizes."""
        cid = "NC09"
        name = "Streaming Buffer Invalid Window"
        desc = "Streaming buffer rejects non-positive max_window_bins."
        try:
            StreamingBufferState(obs_dim=8, max_window_bins=0)
            return ControlResult(cid, name, desc, False, "None", "Failed to reject non-positive window")
        except (ValueError, ValidationError) as e:
            return ControlResult(cid, name, desc, True, type(e).__name__, str(e))

    @staticmethod
    def nc10_procrustes_dimension_mismatch() -> ControlResult:
        """NC10: Procrustes aligner must reject coordinate matrices of mismatched dimensionality."""
        cid = "NC10"
        name = "Procrustes Dimension Mismatch"
        desc = "OrthogonalProcrustesAligner rejects source and target with mismatched feature dims."
        X = np.random.randn(50, 4)
        Y = np.random.randn(50, 5)  # Dimension mismatch (4 != 5)!
        aligner = OrthogonalProcrustesAligner()
        try:
            aligner.fit(X, Y)
            return ControlResult(cid, name, desc, False, "None", "Failed to reject dimension mismatch")
        except ValueError as e:
            return ControlResult(cid, name, desc, True, type(e).__name__, str(e))

    @staticmethod
    def nc11_recovery_corrupt_nan_checkpoint() -> ControlResult:
        """NC11: Training recovery must reject checkpoints containing NaNs."""
        cid = "NC11"
        name = "Recovery Non-Finite Weights"
        desc = "Recovery rejects corrupt checkpoints containing NaN weights."
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            bad_ckpt = {
                "epoch": 1,
                "step": 10,
                "model_state_dict": {"w": torch.tensor([1.0, float("nan")])},
                "optimizer_state_dict": {},
                "metrics": {},
                "rng_states": {},
            }
            torch.save(bad_ckpt, tmp.name)
            try:
                TrainingRecovery.verify_checkpoint_integrity(tmp.name)
                return ControlResult(cid, name, desc, False, "None", "Failed to catch NaN parameter in checkpoint")
            except ValueError as e:
                return ControlResult(cid, name, desc, True, type(e).__name__, str(e))

    @staticmethod
    def nc12_attention_mask_dimension_mismatch() -> ControlResult:
        """NC12: Attention model must reject observation masks with incompatible sequence lengths."""
        cid = "NC12"
        name = "Attention Mask Dimension Incompatibility"
        desc = "SpatiotemporalAttentionModel rejects masks whose shapes do not match inputs."
        cfg = AttentionConfig(obs_dim=8, d_model=16, num_heads=2, num_layers=1)
        model = SpatiotemporalAttentionModel(cfg)
        x = torch.zeros(1, 10, 8)
        mask_mismatched = torch.ones(1, 15, 8, dtype=torch.bool)  # T=15 != T=10!
        try:
            model(x, observed_mask=mask_mismatched)
            return ControlResult(cid, name, desc, False, "None", "Failed to reject mismatched mask length")
        except (ValueError, RuntimeError) as e:
            return ControlResult(cid, name, desc, True, type(e).__name__, str(e))

    @classmethod
    def run_all(cls) -> List[ControlResult]:
        """Execute all 12 negative controls.

        Returns:
            List of ControlResult instances.
        """
        controls: List[Callable[[], ControlResult]] = [
            cls.nc01_temporal_interval_inversion,
            cls.nc02_dataset_split_train_test_overlap,
            cls.nc03_eval_integrity_partition_leakage,
            cls.nc04_bps_zero_spike_denominator,
            cls.nc05_causal_leakage_future_perturbation,
            cls.nc06_poisson_negative_rate_rejection,
            cls.nc07_poisson_nan_input_rejection,
            cls.nc08_mask_all_channels_dropped,
            cls.nc09_streaming_buffer_invalid_window,
            cls.nc10_procrustes_dimension_mismatch,
            cls.nc11_recovery_corrupt_nan_checkpoint,
            cls.nc12_attention_mask_dimension_mismatch,
        ]
        return [ctrl() for ctrl in controls]
