#!/usr/bin/env python3
"""Analyze paired IPSI causal runs and apply the preregistered Gate-1 thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gap_lab.paired_binary import paired_binary_summary


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text())
    if "episodes" not in report:
        raise ValueError(f"{path} is missing episodes")
    return report


def paired_success(reference: dict, candidate: dict, *, samples: int, seed: int) -> dict:
    ref_seeds = [row["seed"] for row in reference["episodes"]]
    cand_seeds = [row["seed"] for row in candidate["episodes"]]
    if ref_seeds != cand_seeds:
        raise ValueError("paired reports must contain the same episode seeds in the same order")
    return paired_binary_summary(
        [row["success"] for row in reference["episodes"]],
        [row["success"] for row in candidate["episodes"]],
        bootstrap_samples=samples,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--slot-only", type=Path, required=True)
    parser.add_argument("--ipsi", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        name: load_report(path)
        for name, path in {
            "normal": args.normal,
            "random": args.random,
            "slot_only": args.slot_only,
            "ipsi": args.ipsi,
        }.items()
    }
    task = reports["normal"].get("task")
    if any(report.get("task") != task for report in reports.values()):
        raise ValueError("all reports must evaluate the same task")
    ipsi_vs_normal = paired_success(
        reports["normal"], reports["ipsi"], samples=args.bootstrap_samples, seed=args.seed
    )
    ipsi_vs_random = paired_success(
        reports["random"], reports["ipsi"], samples=args.bootstrap_samples, seed=args.seed + 1
    )
    ipsi_vs_slot = paired_success(
        reports["slot_only"], reports["ipsi"], samples=args.bootstrap_samples, seed=args.seed + 2
    )
    ipsi_editor = reports["ipsi"].get("editor_summary") or {}
    slot_gain = float(ipsi_editor.get("slot_accuracy_after", 0.0)) - float(
        ipsi_editor.get("slot_accuracy_before", 0.0)
    )
    identity_change = float(ipsi_editor.get("identity_accuracy_after", 0.0)) - float(
        ipsi_editor.get("identity_accuracy_before", 0.0)
    )
    behavior_pass = (
        ipsi_vs_normal["paired_rate_difference"] >= 0.10
        and ipsi_vs_normal["paired_bootstrap_95_ci"][0] > 0.0
    )
    random_pass = ipsi_vs_random["paired_rate_difference"] >= 0.08
    representation_pass = slot_gain >= 0.15 and identity_change >= -0.05
    mechanism_pass = ipsi_vs_slot["paired_rate_difference"] > 0.0 or (
        (reports["slot_only"].get("editor_summary") or {}).get("identity_accuracy_after", 1.0)
        - (reports["slot_only"].get("editor_summary") or {}).get("identity_accuracy_before", 1.0)
        < -0.05
    )
    result = {
        "schema_version": 1,
        "task": task,
        "ipsi_vs_normal": ipsi_vs_normal,
        "ipsi_vs_random": ipsi_vs_random,
        "ipsi_vs_slot_only": ipsi_vs_slot,
        "slot_accuracy_gain": slot_gain,
        "identity_accuracy_change": identity_change,
        "thresholds": {
            "ipsi_minus_normal_min": 0.10,
            "ipsi_vs_normal_ci_lower_must_exceed": 0.0,
            "ipsi_minus_random_min": 0.08,
            "slot_accuracy_gain_min": 0.15,
            "identity_accuracy_change_min": -0.05,
        },
        "checks": {
            "behavior": behavior_pass,
            "random_control": random_pass,
            "representation": representation_pass,
            "identity_preservation_mechanism": mechanism_pass,
        },
        "passed": behavior_pass and random_pass and representation_pass and mechanism_pass,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
