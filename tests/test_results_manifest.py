from pathlib import Path

import pytest
import yaml

from vla_gap_lab.results_manifest import load_and_validate_results

ROOT = Path(__file__).resolve().parents[1]


def test_tracked_results_manifest_is_valid():
    report = load_and_validate_results(ROOT / "results/gate0_summary.yaml", ROOT)
    assert all(not result["gate_passed"] for result in report["tracks"].values())
    assert all(not result["protocol_complete"] for result in report["tracks"].values())


def test_manifest_rejects_training_after_failed_gate(tmp_path):
    report = yaml.safe_load((ROOT / "results/gate0_summary.yaml").read_text())
    report["tracks"]["latent_action"]["method_training_started"] = True
    path = tmp_path / "results.yaml"
    path.write_text(yaml.safe_dump(report))
    with pytest.raises(ValueError, match="failed gate"):
        load_and_validate_results(path, ROOT)


def test_manifest_rejects_incomplete_protocol_claims(tmp_path):
    report = yaml.safe_load((ROOT / "results/gate0_summary.yaml").read_text())
    report["tracks"]["latent_action"]["gate_passed"] = True
    path = tmp_path / "results.yaml"
    path.write_text(yaml.safe_dump(report))
    with pytest.raises(ValueError, match="incomplete protocol"):
        load_and_validate_results(path, ROOT)

    report["tracks"]["latent_action"]["gate_passed"] = False
    report["tracks"]["latent_action"]["protocol_deviations"] = []
    path.write_text(yaml.safe_dump(report))
    with pytest.raises(ValueError, match="document incomplete"):
        load_and_validate_results(path, ROOT)


def test_manifest_rejects_threshold_drift(tmp_path):
    report = yaml.safe_load((ROOT / "results/gate0_summary.yaml").read_text())
    report["tracks"]["latent_action"]["evidence"]["representation_threshold"] = 0.75
    path = tmp_path / "results.yaml"
    path.write_text(yaml.safe_dump(report))
    with pytest.raises(ValueError, match="threshold drift"):
        load_and_validate_results(path, ROOT)
