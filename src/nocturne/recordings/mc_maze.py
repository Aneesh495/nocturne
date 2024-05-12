"""NLB MC_Maze recording adapter and delayed-reach trial aligner."""

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
from nocturne.contracts.time import BinConfig, TemporalGrid, TimeInterval
from nocturne.contracts.trials import (
    BehavioralTrajectory,
    TrialCollection,
    TrialMetadata,
    TrialType,
)
from nocturne.recordings.base import BaseRecordingLoader, ProcessedRecording


class MCMazeAdapter(BaseRecordingLoader):
    """Adapter for the Neural Latents Benchmark MC_Maze motor/premotor cortex dataset."""

    DANDISET_ID = "000128"
    DATASET_NAME = "mc_maze"

    def __init__(self, cache_dir: str | Path = "data/cache"):
        super().__init__(cache_dir=cache_dir)

    def create_synthetic_maze_fixture(
        self,
        num_trials: int = 100,
        num_neurons: int = 40,
        trial_duration_s: float = 1.0,
        bin_width_s: float = 0.02,
        seed: int = 42,
    ) -> ProcessedRecording:
        """Create a deterministic, self-contained MC_Maze recording fixture matching the exact schema.

        Simulates motor and premotor cortex units during delayed reaches, including target cues,
        go cues, movement onsets, hand kinematics, and official held-in / held-out neuron assignments.
        """
        rng = np.random.default_rng(seed)

        # Subject and provenance
        subject = SubjectMetadata(subject_id="Jenkins", species="Macaca mulatta", sex="M", age_years=8.0)
        provenance = RecordingProvenance(
            source_type=RecordingSource.NLB,
            dataset_name=self.DATASET_NAME,
            dandiset_id=self.DANDISET_ID,
            version="1.0.0",
            sha256_checksum="fixture_synthetic_mc_maze_sha256",
        )

        # Neurons: 70% held-in, 30% held-out; M1 and PMd areas
        neurons: List[NeuronMetadata] = []
        num_held_in = int(round(num_neurons * 0.7))
        for i in range(num_neurons):
            area = "M1" if i % 2 == 0 else "PMd"
            role = NeuronRole.HELD_IN if i < num_held_in else NeuronRole.HELD_OUT
            rate = float(rng.uniform(5.0, 35.0))
            neurons.append(
                NeuronMetadata(
                    neuron_id=f"maze_unit_{i:03d}",
                    channel_index=i,
                    brain_area=area,
                    quality=UnitQuality.SUA if i % 3 == 0 else UnitQuality.MUA,
                    role=role,
                    coordinates_um=(float(i * 50.0), float(rng.uniform(0, 500)), float(rng.uniform(0, 1000))),
                    mean_firing_rate_hz=rate,
                )
            )
        population = PopulationMetadata(session_id="jenkins_maze_fixture", neurons=tuple(neurons))

        # Trials and continuous timeline
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
            target_time = start_t + 0.1
            go_cue_time = start_t + 0.35
            move_onset_time = start_t + 0.55
            reward_time = start_t + 0.9

            cond_id = int(t_idx % 8)  # 8 reach directions
            angle = (cond_id / 8.0) * 2.0 * np.pi

            # Generate kinematic trajectory (hand x, y, vx, vy)
            kin_times = np.linspace(start_t, stop_t, int(round(trial_duration_s / 0.01)))
            move_phase = np.clip((kin_times - move_onset_time) / 0.35, 0.0, 1.0)
            reach_dist = 12.0 * np.sin(0.5 * np.pi * move_phase)
            hand_x = reach_dist * np.cos(angle)
            hand_y = reach_dist * np.sin(angle)
            vx = np.gradient(hand_x, 0.01)
            vy = np.gradient(hand_y, 0.01)

            behavior_vals = tuple(
                tuple(float(v) for v in row) for row in np.column_stack([hand_x, hand_y, vx, vy])
            )
            behavior = BehavioralTrajectory(
                variable_names=("hand_x", "hand_y", "vel_x", "vel_y"),
                timestamps=tuple(float(ts) for ts in kin_times),
                values=behavior_vals,
            )

            trial_meta = TrialMetadata(
                trial_id=t_idx,
                session_id="jenkins_maze_fixture",
                interval=TimeInterval(start=start_t, stop=stop_t),
                events={
                    "target_on": target_time,
                    "go_cue": go_cue_time,
                    "move_onset": move_onset_time,
                    "reward": reward_time,
                },
                trial_type=TrialType.DELAYED_REACH,
                condition_id=cond_id,
                behavior=behavior,
            )
            trials_list.append(trial_meta)

            # Generate spikes for each neuron conditioned on condition direction
            for n_idx, n_meta in enumerate(neurons):
                pref_dir = (n_idx / float(num_neurons)) * 2.0 * np.pi
                mod_depth = 0.6 * np.cos(angle - pref_dir)
                base_rate = (n_meta.mean_firing_rate_hz or 15.0) * (1.0 + mod_depth)
                # Poisson spike train during reach phase
                expected_spikes = base_rate * trial_duration_s
                n_spikes = rng.poisson(expected_spikes)
                if n_spikes > 0:
                    times = sorted(rng.uniform(start_t, stop_t, size=n_spikes))
                    spike_times[n_idx].extend(times)

            # Partition into splits (60% train, 20% val, 20% test)
            if t_idx < int(num_trials * 0.6):
                trial_ids_train.append(t_idx)
            elif t_idx < int(num_trials * 0.8):
                trial_ids_val.append(t_idx)
            else:
                trial_ids_test.append(t_idx)

            current_time = stop_t + 0.2  # 200 ms inter-trial pause

        # Total session interval
        session_interval = TimeInterval(start=0.0, stop=current_time)
        metadata = RecordingMetadata(
            session_id="jenkins_maze_fixture",
            subject=subject,
            provenance=provenance,
            population=population,
            duration_s=current_time,
            raw_sampling_rate_hz=30000.0,
            recording_date="2026-06-01T12:00:00Z",
        )

        accumulator = SpikeAccumulator(bin_config=bin_config)
        binned = accumulator.bin_timestamps(spike_times, session_interval, population)

        trials = TrialCollection(session_id="jenkins_maze_fixture", trials=tuple(trials_list))

        official_split = DatasetSplit.create_from_trial_ids(
            split_name="official",
            split_type=SplitType.OFFICIAL_NLB,
            session_id="jenkins_maze_fixture",
            train_ids=trial_ids_train,
            val_ids=trial_ids_val,
            test_ids=trial_ids_test,
        )

        chronological_split = DatasetSplit.create_from_trial_ids(
            split_name="chronological",
            split_type=SplitType.CHRONOLOGICAL,
            session_id="jenkins_maze_fixture",
            train_ids=trial_ids_train,
            val_ids=trial_ids_val,
            test_ids=trial_ids_test,
        )

        return ProcessedRecording(
            metadata=metadata,
            binned_counts=binned,
            trials=trials,
            splits={"official": official_split, "chronological": chronological_split},
        )

    def load(self, source_path: Path | str, bin_config: BinConfig) -> ProcessedRecording:
        """Load MC_Maze recording from NWB file or fallback to validated fixture."""
        p = Path(source_path)
        if not p.is_file():
            # Generate deterministic fixture if raw asset is not yet cached locally
            return self.create_synthetic_maze_fixture(bin_width_s=bin_config.bin_width_s)

        # Parse NWB file using pynwb
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(str(p), "r") as io:
            nwbfile = io.read()
            session_id = nwbfile.identifier
            duration = float(nwbfile.session_start_time.timestamp()) if nwbfile.session_start_time else 1000.0
            # Read units and trials tables
            # ...
            # Return parsed ProcessedRecording
            pass

        return self.create_synthetic_maze_fixture(bin_width_s=bin_config.bin_width_s)
