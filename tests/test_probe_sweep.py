from __future__ import annotations

import pytest

from vla_gap_lab.probe_sweep import summarize_probe_reports


def report(cache: str, alpha: float, seed: int, rows: list[tuple[int, float, float]]) -> dict:
    return {
        "cache": cache,
        "probe": {"alpha": alpha, "split_seed": seed},
        "layers": [
            {"layer": layer, "clean": {"r2": clean}, "shifted": {"r2": shifted}}
            for layer, clean, shifted in rows
        ],
    }


def test_sweep_uses_each_runs_best_valid_layer() -> None:
    reports = [
        report("same.npz", 1, 7, [(4, 0.5, 0.45), (8, 0.8, 0.4)]),
        report("same.npz", 10, 7, [(4, -0.1, 0.9), (8, 0.5, 0.3)]),
    ]
    result = summarize_probe_reports(reports, representation_threshold=0.8)
    assert result["maximum_retention"] == pytest.approx(0.9)
    assert result["representation_gate_passed"] is True
    assert result["by_alpha"]["1.0"]["max"] == pytest.approx(0.9)


def test_sweep_keeps_invalid_run_and_rejects_mixed_caches() -> None:
    valid = report("a.npz", 1, 1, [(4, 0.5, 0.1)])
    invalid = report("a.npz", 1, 2, [(4, -0.5, 0.1)])
    result = summarize_probe_reports([valid, invalid])
    assert result["valid_runs"] == 1
    assert result["invalid_runs"] == 1
    with pytest.raises(ValueError, match="mix caches"):
        summarize_probe_reports([valid, report("b.npz", 1, 1, [(4, 0.5, 0.1)])])
