"""Structured artifact schema and scientific artifact registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ArtifactMetadata(BaseModel):
    """Provenance and execution metadata for a saved experimental artifact."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str = Field(..., description="Unique artifact identifier")
    artifact_type: str = Field(..., description="Type ('figure', 'model_weights', 'eval_bundle', 'metrics')")
    dataset_name: str = Field(..., description="Target recording or circuit name")
    split_name: str = Field(..., description="Dataset split partition evaluated")
    model_name: str = Field(..., description="Model architecture evaluated")
    seed: int = Field(default=42, description="RNG seed")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Summary evaluation metrics")
    file_path: str = Field(..., description="Relative or absolute path to primary artifact file")
    sha256_hash: Optional[str] = Field(default=None, description="SHA-256 integrity hash")


class ArtifactRegistry:
    """Manages creation, indexing, and validation of experimental scientific artifacts."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root_dir / "artifact_index.json"

    def register(self, metadata: ArtifactMetadata) -> None:
        """Add artifact metadata to persistent index.

        Args:
            metadata: ArtifactMetadata instance.
        """
        records = self.list_artifacts()
        # Remove any existing record with identical artifact_id
        records = [r for r in records if r["artifact_id"] != metadata.artifact_id]
        records.append(metadata.model_dump())

        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def list_artifacts(self) -> List[Dict[str, Any]]:
        """Retrieve all registered artifacts."""
        if not self.index_file.exists():
            return []
        with open(self.index_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_artifact(self, artifact_id: str) -> Optional[ArtifactMetadata]:
        """Look up artifact by ID."""
        for record in self.list_artifacts():
            if record["artifact_id"] == artifact_id:
                return ArtifactMetadata(**record)
        return None
