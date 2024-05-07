"""Unit tests for recording metadata and provenance contracts."""

import pytest

from nocturne.contracts.neurons import NeuronMetadata, NeuronRole, PopulationMetadata
from nocturne.contracts.recordings import (
    RecordingMetadata,
    RecordingProvenance,
    RecordingSource,
    SubjectMetadata,
)


def test_recording_metadata_creation():
    subject = SubjectMetadata(subject_id="Jenkins", species="Macaca mulatta", sex="M", age_years=8.5)
    provenance = RecordingProvenance(
        source_type=RecordingSource.DANDI,
        dataset_name="mc_maze",
        dandiset_id="000128",
        sha256_checksum="abcdef1234567890",
        file_path="sub-Jenkins/sub-Jenkins_ses-maze_ecephys.nwb",
        file_size_bytes=104857600,
    )

    n0 = NeuronMetadata(neuron_id="u0", channel_index=0, brain_area="M1")
    n1 = NeuronMetadata(neuron_id="u1", channel_index=1, brain_area="PMd")
    pop = PopulationMetadata(session_id="jenkins_maze_2026", neurons=(n0, n1))

    recording = RecordingMetadata(
        session_id="jenkins_maze_2026",
        subject=subject,
        provenance=provenance,
        population=pop,
        duration_s=3600.0,
        raw_sampling_rate_hz=30000.0,
        recording_date="2026-05-15T10:00:00Z",
    )

    assert recording.session_id == "jenkins_maze_2026"
    assert recording.subject.subject_id == "Jenkins"
    assert recording.provenance.dandiset_id == "000128"
    assert recording.population.num_neurons == 2
    assert recording.duration_s == 3600.0


def test_recording_metadata_validation():
    subject = SubjectMetadata(subject_id="Synth01", species="Synthetic")
    provenance = RecordingProvenance(
        source_type=RecordingSource.SYNTHETIC,
        dataset_name="damped_oscillator",
    )
    pop = PopulationMetadata(session_id="synth_ses", neurons=())

    with pytest.raises(ValueError):
        # Duration must be > 0
        RecordingMetadata(
            session_id="synth_ses",
            subject=subject,
            provenance=provenance,
            population=pop,
            duration_s=0.0,
        )
