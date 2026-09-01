"""Validation for the tracked Gate-0 result summary."""

from pathlib import Path

import yaml

TRACKS = {"latent_action", "memory_revision", "state_transport"}

THRESHOLD_LINKS = {
    "latent_action": {
        "representation_threshold": "representation_retention_min",
        "control_threshold": "control_retention_max",
    },
    "memory_revision": {"required_causal_success_gain_pp": "reset_refresh_gain_pp_min"},
    "state_transport": {"required_cross_phase_accuracy": "cross_phase_accuracy_min"},
}


def load_and_validate_results(path: str | Path, repository_root: str | Path) -> dict:
    path = Path(path)
    root = Path(repository_root)
    report = yaml.safe_load(path.read_text())
    if report.get("schema_version") != 2:
        raise ValueError("unsupported result manifest schema")
    tracks = report.get("tracks", {})
    if set(tracks) != TRACKS:
        raise ValueError(f"manifest must contain exactly {sorted(TRACKS)}")
    for name, result in tracks.items():
        if not isinstance(result.get("gate_passed"), bool):
            raise TypeError(f"{name}.gate_passed must be boolean")
        if not isinstance(result.get("protocol_complete"), bool):
            raise TypeError(f"{name}.protocol_complete must be boolean")
        if result["gate_passed"] and not result["protocol_complete"]:
            raise ValueError(f"{name} cannot pass an incomplete protocol")
        if not result["protocol_complete"] and not result.get("protocol_deviations"):
            raise ValueError(f"{name} must document incomplete-protocol deviations")
        if result.get("method_training_started") and not result["gate_passed"]:
            raise ValueError(f"{name} started method training despite a failed gate")
        document = root / result["document"]
        if not document.is_file():
            raise ValueError(f"{name} evidence document does not exist: {document}")
        config_path = root / "configs" / name / "gate0.yaml"
        config = yaml.safe_load(config_path.read_text())
        if config.get("track") != name:
            raise ValueError(f"{name} config has mismatched track name")
        for evidence_key, config_key in THRESHOLD_LINKS[name].items():
            evidence_value = result.get("evidence", {}).get(evidence_key)
            config_value = config.get("gate", {}).get(config_key)
            if evidence_value != config_value:
                raise ValueError(
                    f"{name} threshold drift: evidence.{evidence_key}={evidence_value}, "
                    f"config.gate.{config_key}={config_value}"
                )
    return report
