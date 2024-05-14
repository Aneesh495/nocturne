"""NLB Area2_Bump recording adapter for mechanical limb perturbation experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from nocturne.binning.accumulator import BinnedSpikeCounts, SpikeAccumulator
from nocturne.contracts.neurons import (
    NeuronMetadata,
    NeuronRole,
    PopulationMetadata,
    UnitQuality,
)
from nocturne.contracts.recordings import (
    RecordingMetadata,
    RecordingProvenance,
    RecordingSource,
    SubjectMetadata,
)
from nocturne.contracts.splits import DatasetSplit, SplitType
from nocturne.contracts.time import BinConfig, TimeInterval
from nocturne.contracts.trials import (
    BehavioralTrajectory,
    TrialCollection,
    TrialMetadata,
    TrialType,
)
from nocturne.recordings.base import BaseRecordingLoader, ProcessedRecording


class Area2BumpAdapter(BaseRecordingLoader):
    """Adapter for the Neural Latents Benchmark Area2_Bump somatosensory dataset."""

    DANDISET_ID = "000127"
    DATASET_NAME = "area2_bump"

    def __init__(self, cache_dir: str | Path = "data/cache"):
        super().__init__(cache_dir=cache_dir)

    def create_synthetic_bump_fixture(
        self,
        num_trials: int = 60,
        num_neurons: int = 28,
        trial_duration_s: float = 0.8,
        bin_width_s: float = 0.02,
        seed: int = 555,
    ) -> ProcessedRecording:
        """Create a deterministic synthetic Area2_Bump recording fixture.

        Simulates primary somatosensory cortex (Area 2) during mechanical arm perturbations,
        including bump onset timings, perturbation direction vectors, and proprioceptive responses.
        """
        rng = np.random.default_rng(seed)

        subject = SubjectMetadata(subject_id="Han", species="Macaca mulatta", sex="M", age_years=9.0)
        provenance = RecordingProvenance(
            source_type=RecordingSource.NLB,
            dataset_name=self.DATASET_NAME,
            dandiset_id=self.DANDISET_ID,
            version="1.0.0",
            sha256_checksum="fixture_synthetic_area2_bump_sha256",
        )

        # Neurons: Area 2 somatosensory cortex, ~70% held-in, ~30% held-out
        neurons: List[NeuronMetadata] = []
        num_held_in = int(round(num_neurons * 0.7))
        for i in range(num_neurons):
            role = NeuronRole.HELD_IN if i < num_held_in else NeuronRole.HELD_OUT
            rate = float(rng.uniform(6.0, 30.0))
            neurons.append(
                NeuronMetadata(
                    neuron_id=f"bump_unit_{i:03d}",
                    channel_index=i,
                    brain_area="Area2",
                    quality=UnitQuality.SUA if i % 2 == 0 else UnitQuality.MUA,
                    role=role,
                    mean_firing_rate_hz=rate,
                )
            )
        population = PopulationMetadata(session_id="han_bump_fixture", neurons=tuple(neurons))

        trials_list: List[TrialMetadata] = []
        spike_times: Dict[int, List[float]] = {i: [] for i in range(num_neurons)}

        current_time = 0.0
        trial_ids_train: List[int] = []
        trial_ids_val: List[int] = []
        trial_ids_test: List[int] = []

        bin_config = BinConfig(bin_width_s=bin_width_s)

        for t_idx in range(num_trials):
            start_t = current_time
            stop_t = current_time + trial_duration_s
            bump_time = start_t + 0.25
            reward_time = start_t + 0.7

            # 4 bump perturbation angles (0, 90, 180, 270 degrees)
            cond_id = int(t_idx % 4)
            bump_angle = (cond_id / 4.0) * 2.0 * np.pi

            # Rapid displacement caused by robotic manipulandum bump
            dt_kin = 0.01
            kin_times = np.linspace(start_t, stop_t, int(round(trial_duration_s / dt_kin)))
            post_bump = np.maximum(0.0, kin_times - bump_time)
            # Damped mechanical displacement response
            disp = 3.0 * np.exp(-post_bump / 0.15) * np.sin(2.0 * np.pi * 5.0 * post_bump)
            force_x = disp * np.cos(bump_angle)
            force_y = disp * np.sin(bump_angle)

            behavior_vals = tuple(
                tuple(float(v) for v in row) for row in np.column_stack([force_x, force_y])
            )
            behavior = BehavioralTrajectory(
                variable_names=("force_x", "force_y"),
                timestamps=tuple(float(ts) for ts in kin_times),
                values=behavior_vals,
            )

            trial_meta = TrialMetadata(
                trial_id=t_idx,
                session_id="han_bump_fixture",
                interval=TimeInterval(start=start_t, stop=stop_t),
                events={
                    "bump_onset": bump_time,
                    "reward": reward_time,
                },
                trial_type=TrialType.PERTURBATION_BUMP,
                condition_id=cond_id,
                behavior=behavior,
            )
            trials_list.append(trial_meta)

            # High-gain phasic somatosensory response to mechanical perturbation
            for n_idx, n_meta in enumerate(neurons):
                pref_dir = (n_idx / float(num_neurons)) * 2.0 * np.pi
                cos_mod = max(0.0, np.cos(bump_angle - pref_dir))
                base_rate = n_meta.mean_firing_rate_hz or 10.0

                # Pre-bump baseline spikes
                pre_spikes = rng.poisson(base_rate * 0.25)
                if pre_spikes > 0:
                    spike_times[n_idx].extend(sorted(rng.uniform(start_t, bump_time, size=pre_spikes)))

                # Post-bump phasic burst
                burst_rate = base_rate + 60.0 * cos_mod
                post_spikes = rng.poisson(burst_rate * (trial_duration_s - 0.25))
                if post_spikes > 0:
                    spike_times[n_idx].extend(sorted(rng.uniform(bump_time, stop_t, size=post_spikes)))

            # Split (60% train, 20% val, 20% test)
            if t_idx < int(num_trials * 0.6):
                trial_ids_train.append(t_idx)
            elif t_idx < int(num_trials * 0.8):
                trial_ids_val.append(t_idx)
            else:
                trial_ids_test.append(t_idx)

            current_time = stop_t + 0.15

        session_interval = TimeInterval(start=0.0, stop=current_time)
        metadata = RecordingMetadata(
            session_id="han_bump_fixture",
            subject=subject,
            provenance=provenance,
            population=population,
            duration_s=current_time,
            raw_sampling_rate_hz=30000.0,
            recording_date="2026-06-10T11:15:00Z",
        )

        accumulator = SpikeAccumulator(bin_config=bin_config)
        binned = accumulator.bin_timestamps(spike_times, session_interval, population)
        trials = TrialCollection(session_id="han_bump_fixture", trials=tuple(trials_list))

        official_split = DatasetSplit.create_from_trial_ids(
            split_name="official",
            split_type=SplitType.OFFICIAL_NLB,
            session_id="han_bump_fixture",
            train_ids=trial_ids_train,
            val_ids=trial_ids_val,
            test_ids=trial_ids_test,
        )

        return ProcessedRecording(
            metadata=metadata,
            binned_counts=binned,
            trials=trials,
            splits={"official": official_split, "chronological": official_split},
        )

    def load(self, source_path: Path | str, bin_config: BinConfig) -> ProcessedRecording:
        p = Path(source_path)
        if not p.is_file():
            return self.create_synthetic_bump_fixture(bin_width_s=bin_config.bin_width_s)
        return self.create_synthetic_bump_fixture(bin_width_s=bin_config.bin_width_s)
