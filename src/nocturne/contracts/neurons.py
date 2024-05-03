"""Neuron and channel metadata contracts defining unit identity, brain areas, and observation roles."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Sequence, Tuple
from pydantic import BaseModel, ConfigDict, Field, field_validator


class UnitQuality(str, Enum):
    """Electrophysiological isolation quality of a spike-sorted unit."""

    SUA = "single_unit"
    MUA = "multi_unit"
    UNSORTED = "unsorted"


class NeuronRole(str, Enum):
    """Observation role assigned to a neuron in representation learning benchmarks."""

    PRIMARY_OBSERVED = "primary_observed"
    HELD_IN = "held_in"
    HELD_OUT = "held_out"


class NeuronMetadata(BaseModel):
    """Metadata and physiological attributes of an individual recorded neural unit."""

    model_config = ConfigDict(frozen=True)

    neuron_id: str = Field(..., description="Unique unit identifier within session")
    channel_index: int = Field(..., ge=0, description="0-indexed population array channel index")
    brain_area: str = Field(..., description="Anatomical brain region (e.g. M1, PMd, Area2)")
    quality: UnitQuality = Field(default=UnitQuality.SUA, description="Spike sorting isolation quality")
    role: NeuronRole = Field(default=NeuronRole.PRIMARY_OBSERVED, description="Benchmark observation role")
    coordinates_um: Tuple[float, float, float] | None = Field(
        default=None, description="Probe coordinates (x, y, z) in micrometers"
    )
    mean_firing_rate_hz: float | None = Field(
        default=None, ge=0.0, description="Baseline firing rate across the entire recording"
    )
    snr: float | None = Field(default=None, ge=0.0, description="Signal-to-noise ratio of mean waveform")


class PopulationMetadata(BaseModel):
    """Complete collection of neural units for a recording session."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(..., description="Session identifier")
    neurons: Tuple[NeuronMetadata, ...] = Field(..., description="Tuple of recorded neurons")

    def model_post_init(self, __context: object) -> None:
        ids = set()
        channel_indices = set()
        for n in self.neurons:
            if n.neuron_id in ids:
                raise ValueError(f"Duplicate neuron_id found: {n.neuron_id}")
            ids.add(n.neuron_id)
            if n.channel_index in channel_indices:
                raise ValueError(f"Duplicate channel_index found: {n.channel_index}")
            channel_indices.add(n.channel_index)

    @property
    def num_neurons(self) -> int:
        """Total number of neurons in the population."""
        return len(self.neurons)

    @property
    def channel_indices(self) -> List[int]:
        """List of all channel indices."""
        return [n.channel_index for n in self.neurons]

    @property
    def neuron_ids(self) -> List[str]:
        """List of all neuron unique IDs."""
        return [n.neuron_id for n in self.neurons]

    def get_by_id(self, neuron_id: str) -> NeuronMetadata:
        """Look up neuron metadata by its unique ID."""
        for n in self.neurons:
            if n.neuron_id == neuron_id:
                return n
        raise KeyError(f"Neuron with ID '{neuron_id}' not found in population {self.session_id}")

    def get_indices_by_role(self, role: NeuronRole) -> List[int]:
        """Return sorted channel indices matching a specific observation role."""
        return sorted([n.channel_index for n in self.neurons if n.role == role])

    @property
    def held_in_indices(self) -> List[int]:
        """Channel indices designated as held-in (observed during co-smoothing inference)."""
        return sorted([n.channel_index for n in self.neurons if n.role in (NeuronRole.HELD_IN, NeuronRole.PRIMARY_OBSERVED)])

    @property
    def held_out_indices(self) -> List[int]:
        """Channel indices designated as held-out (unobserved targets during co-smoothing)."""
        return sorted([n.channel_index for n in self.neurons if n.role == NeuronRole.HELD_OUT])

    def get_indices_by_area(self, brain_area: str) -> List[int]:
        """Return sorted channel indices located in a specific brain area."""
        return sorted([n.channel_index for n in self.neurons if n.brain_area == brain_area])

    def filter_by_indices(self, indices: Sequence[int]) -> PopulationMetadata:
        """Create a new PopulationMetadata containing only neurons at the specified channel indices."""
        target_set = set(indices)
        sub_neurons = tuple(n for n in self.neurons if n.channel_index in target_set)
        return PopulationMetadata(
            session_id=self.session_id,
            neurons=sub_neurons,
        )
