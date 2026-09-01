"""Aggregate layerwise probe reports without silently dropping failed runs."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize_probe_reports(
    reports: list[dict[str, Any]], representation_threshold: float = 0.8
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one report is required")
    caches = {str(report.get("cache")) for report in reports}
    if len(caches) != 1:
        raise ValueError(f"reports mix caches: {sorted(caches)}")

    runs = []
    by_alpha: dict[float, list[float]] = defaultdict(list)
    invalid_runs = 0
    for report in reports:
        candidates = []
        for row in report["layers"]:
            clean = float(row["clean"]["r2"])
            shifted = float(row["shifted"]["r2"])
            if clean > 0 and math.isfinite(clean) and math.isfinite(shifted):
                candidates.append((shifted / clean, int(row["layer"]), clean, shifted))
        probe = report["probe"]
        alpha, seed = float(probe["alpha"]), int(probe["split_seed"])
        if not candidates:
            invalid_runs += 1
            runs.append({"alpha": alpha, "seed": seed, "valid": False})
            continue
        retention, layer, clean, shifted = max(candidates)
        by_alpha[alpha].append(retention)
        runs.append(
            {
                "alpha": alpha,
                "seed": seed,
                "valid": True,
                "best_layer": layer,
                "clean_r2": clean,
                "shifted_r2": shifted,
                "max_retention": retention,
            }
        )

    valid = [row["max_retention"] for row in runs if row["valid"]]
    if not valid:
        raise ValueError("no layer has finite R2 with positive clean R2")
    alpha_summary = {
        str(alpha): {"runs": len(values), "min": min(values), "max": max(values)}
        for alpha, values in sorted(by_alpha.items())
    }
    maximum = max(valid)
    return {
        "schema_version": 1,
        "cache": next(iter(caches)),
        "reports": len(reports),
        "valid_runs": len(valid),
        "invalid_runs": invalid_runs,
        "representation_threshold": representation_threshold,
        "maximum_retention": maximum,
        "representation_gate_passed": maximum >= representation_threshold,
        "by_alpha": alpha_summary,
        "runs": sorted(runs, key=lambda row: (row["alpha"], row["seed"])),
    }


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    import json

    return [json.loads(path.read_text()) for path in paths]
