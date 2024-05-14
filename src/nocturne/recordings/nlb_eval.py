"""Local NLB test evaluation adapter and official metric scoring bridge."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from nocturne.binning.likelihood_math import poisson_bits_per_spike, poisson_log_likelihood
from nocturne.contracts.neurons import NeuronRole, PopulationMetadata
from nocturne.recordings.base import ProcessedRecording


class NLBEvaluationTarget(BaseModel):
    """Target evaluation tensors for held-out local test units in an NLB benchmark dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    dataset_name: str = Field(..., description="Canonical NLB dataset name (e.g. mc_maze)")
    test_trial_ids: Tuple[int, ...] = Field(..., description="Official test trial IDs")
    held_out_channel_indices: Tuple[int, ...] = Field(..., description="Indices of held-out evaluated neurons")
    test_counts: np.ndarray = Field(
        ..., description="Observed test spike counts [num_test_trials, num_bins, num_held_out]"
    )
    baseline_rates_hz: np.ndarray = Field(
        ..., description="Training-set empirical mean rates for held-out units [num_held_out]"
    )
    bin_width_s: float = Field(..., gt=0.0, description="Bin width in seconds")

    def model_post_init(self, __context: object) -> None:
        N_trials = len(self.test_trial_ids)
        N_units = len(self.held_out_channel_indices)
        if self.test_counts.shape[0] != N_trials:
            raise ValueError(f"test_counts trial dim ({self.test_counts.shape[0]}) != N_trials ({N_trials})")
        if self.test_counts.shape[2] != N_units:
            raise ValueError(f"test_counts unit dim ({self.test_counts.shape[2]}) != N_units ({N_units})")
        if self.baseline_rates_hz.shape != (N_units,):
            raise ValueError(f"baseline_rates_hz shape ({self.baseline_rates_hz.shape}) != ({N_units},)")

    @property
    def total_test_spikes(self) -> int:
        """Official evaluation denominator: total spike count on held-out units."""
        return int(np.sum(self.test_counts))

    @property
    def num_evaluated_bins(self) -> int:
        return self.test_counts.shape[1]

    @property
    def num_held_out_units(self) -> int:
        return len(self.held_out_channel_indices)

    def evaluate_predictions(
        self,
        predicted_rates_hz: np.ndarray,
        use_official_nlb_tools: bool = True,
    ) -> Dict[str, float]:
        """Score predicted firing rates against test ground truth.

        Args:
            predicted_rates_hz: Array of shape [num_test_trials, num_bins, num_held_out] in Hz.
            use_official_nlb_tools: Whether to cross-validate against pinned nlb_tools if installed.

        Returns:
            Dictionary containing bits_per_spike, log_likelihood, and total_spikes.
        """
        if predicted_rates_hz.shape != self.test_counts.shape:
            raise ValueError(
                f"predicted_rates_hz shape {predicted_rates_hz.shape} != target shape {self.test_counts.shape}"
            )

        # Baseline reference rates expanded to [num_test_trials, num_bins, num_held_out]
        ref_expanded = np.broadcast_to(
            self.baseline_rates_hz[None, None, :],
            self.test_counts.shape,
        )

        bps_independent = poisson_bits_per_spike(
            counts=self.test_counts,
            model_rates_hz=predicted_rates_hz,
            reference_rates_hz=ref_expanded,
            bin_width_s=self.bin_width_s,
        )

        ll_model = poisson_log_likelihood(
            counts=self.test_counts,
            rates_hz=predicted_rates_hz,
            bin_width_s=self.bin_width_s,
            include_log_factorial=True,
        )

        results: Dict[str, float] = {
            "bits_per_spike": float(bps_independent) if bps_independent is not None else 0.0,
            "log_likelihood": float(ll_model),
            "total_test_spikes": float(self.total_test_spikes),
            "num_evaluated_trials": float(len(self.test_trial_ids)),
        }

        # Optional comparison with nlb_tools.evaluation
        if use_official_nlb_tools:
            try:
                import nlb_tools.evaluation as nlb_eval

                # Prepare inputs for nlb_tools: y and y_hat
                # nlb_tools bits_per_spike expects y and y_hat arrays
                official_bps = nlb_eval.bits_per_spike(
                    predicted_rates_hz * self.bin_width_s,
                    self.test_counts,
                )
                results["official_nlb_tools_bps"] = float(official_bps)
            except Exception:
                # nlb_tools might use custom format or be unavailable
                pass

        return results


class NLBEvaluationAdapter:
    """Extracts official local evaluation targets from processed recordings."""

    @classmethod
    def extract_evaluation_target(
        cls,
        recording: ProcessedRecording,
        split_name: str = "official",
    ) -> NLBEvaluationTarget:
        """Build an NLBEvaluationTarget from a ProcessedRecording.

        Extracts held-out channel indices, computes training-set baseline rates,
        and isolates test trial ground truth counts.
        """
        split = recording.get_split(split_name)
        pop = recording.metadata.population
        held_out_indices = pop.held_out_indices

        if len(held_out_indices) == 0:
            raise ValueError(f"Recording {recording.session_id} has no channels designated as HELD_OUT")

        test_trial_ids = split.test.trial_ids
        train_trial_ids = split.train.trial_ids

        if len(test_trial_ids) == 0:
            raise ValueError(f"Split '{split_name}' has zero test trials")

        if recording.trials is None:
            raise ValueError(f"Recording {recording.session_id} must have trials for trial-structured evaluation")

        # Compute empirical baseline rates for held-out units using training trials only
        train_counts_list: List[np.ndarray] = []
        train_duration_total_s = 0.0
        grid = recording.binned_counts.grid
        dt = grid.bin_config.bin_width_s

        for t_id in train_trial_ids:
            trial = recording.trials.get_by_id(t_id)
            # Find bins overlapping trial interval
            start_bin = grid.get_bin_index(trial.interval.start) or 0
            stop_bin = min(grid.get_bin_index(trial.interval.stop) or grid.num_bins, grid.num_bins)
            if stop_bin > start_bin:
                counts_slice = recording.binned_counts.counts[start_bin:stop_bin, held_out_indices]
                train_counts_list.append(counts_slice)
                train_duration_total_s += (stop_bin - start_bin) * dt

        if len(train_counts_list) > 0 and train_duration_total_s > 0:
            total_train_counts = np.sum(np.vstack(train_counts_list), axis=0)
            baseline_rates = total_train_counts / train_duration_total_s
        else:
            baseline_rates = np.full(len(held_out_indices), 10.0)

        # Baseline rate lower bound to prevent 0 Hz reference
        baseline_rates = np.maximum(baseline_rates, 0.01)

        # Extract test counts: [num_test_trials, T_eval, num_held_out]
        test_counts_trials: List[np.ndarray] = []
        target_len: Optional[int] = None

        for t_id in test_trial_ids:
            trial = recording.trials.get_by_id(t_id)
            start_bin = grid.get_bin_index(trial.interval.start) or 0
            stop_bin = min(grid.get_bin_index(trial.interval.stop) or grid.num_bins, grid.num_bins)
            trial_slice = recording.binned_counts.counts[start_bin:stop_bin, held_out_indices]

            if target_len is None:
                target_len = trial_slice.shape[0]
            # Standardize length to target_len for batch matrix
            if trial_slice.shape[0] < target_len:
                pad = np.zeros((target_len - trial_slice.shape[0], len(held_out_indices)), dtype=np.int64)
                trial_slice = np.vstack([trial_slice, pad])
            elif trial_slice.shape[0] > target_len:
                trial_slice = trial_slice[:target_len, :]

            test_counts_trials.append(trial_slice)

        test_matrix = np.stack(test_counts_trials, axis=0)

        return NLBEvaluationTarget(
            dataset_name=recording.metadata.provenance.dataset_name,
            test_trial_ids=test_trial_ids,
            held_out_channel_indices=tuple(held_out_indices),
            test_counts=test_matrix,
            baseline_rates_hz=baseline_rates,
            bin_width_s=dt,
        )
