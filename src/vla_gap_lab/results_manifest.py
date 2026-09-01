"""Validation for the tracked Gate-0 result summary."""

from pathlib import Path

import yaml

TRACKS = {"latent_action", "memory_revision", "state_transport"}


def load_and_validate_results(path: str | Path, repository_root: str | Path) -> dict:
    path = Path(path)
    root = Path(repository_root)
    report = yaml.safe_load(path.read_text())
    if report.get("schema_version") != 1:
        raise ValueError("unsupported result manifest schema")
    tracks = report.get("tracks", {})
    if set(tracks) != TRACKS:
        raise ValueError(f"manifest must contain exactly {sorted(TRACKS)}")
    for name, result in tracks.items():
        if not isinstance(result.get("gate_passed"), bool):
            raise TypeError(f"{name}.gate_passed must be boolean")
        if result.get("method_training_started") and not result["gate_passed"]:
            raise ValueError(f"{name} started method training despite a failed gate")
        document = root / result["document"]
        if not document.is_file():
            raise ValueError(f"{name} evidence document does not exist: {document}")
    return report
