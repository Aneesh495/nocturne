"""Recording, session, subject, and provenance metadata contracts."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from nocturne.contracts.neurons import PopulationMetadata


class RecordingSource(str, Enum):
    """Source provenance archive for neural recordings."""

    DANDI = "dandi"
    NLB = "neural_latents_benchmark"
    SYNTHETIC = "synthetic_circuit"
    CUSTOM = "custom"


class SubjectMetadata(BaseModel):
    """Biological or synthetic subject information."""

    model_config = ConfigDict(frozen=True)

    subject_id: str = Field(..., description="Unique subject identifier (e.g. animal name or synthetic model ID)")
    species: str = Field(default="Macaca mulatta", description="Species taxonomic name")
    sex: Optional[str] = Field(default=None, description="Biological sex if recorded")
    age_years: Optional[float] = Field(default=None, ge=0.0, description="Age in years at time of recording")


class RecordingProvenance(BaseModel):
    """Provenance tracking for data reproducibility and cryptographic verification."""

    model_config = ConfigDict(frozen=True)

    source_type: RecordingSource = Field(..., description="Source repository type")
    dataset_name: str = Field(..., description="Canonical dataset name (e.g. mc_maze, mc_rtt, area2_bump)")
    dandiset_id: Optional[str] = Field(default=None, description="DANDI archive identifier if applicable")
    version: str = Field(default="1.0.0", description="Dataset or format version string")
    sha256_checksum: Optional[str] = Field(default=None, description="SHA-256 hash of the source data asset")
    file_path: Optional[str] = Field(default=None, description="Relative or local path to source file")
    file_size_bytes: Optional[int] = Field(default=None, ge=0, description="Size of source file in bytes")


class RecordingMetadata(BaseModel):
    """Top-level metadata describing a single electrophysiology recording session."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(..., description="Unique recording session identifier")
    subject: SubjectMetadata = Field(..., description="Subject metadata")
    provenance: RecordingProvenance = Field(..., description="Source provenance record")
    population: PopulationMetadata = Field(..., description="Recorded neural population metadata")
    duration_s: float = Field(..., gt=0.0, description="Total continuous recording duration in seconds")
    raw_sampling_rate_hz: float = Field(default=30000.0, gt=0.0, description="Raw acquisition sampling rate in Hz")
    recording_date: Optional[str] = Field(default=None, description="ISO-8601 recording date string")
