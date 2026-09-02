#!/usr/bin/env python3
"""Summarize paired K=2/K=8 horizon-pilot evaluator artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gap_lab.paired_binary import paired_binary_summary


def load(path: Path) -> dict:
    report = json.loads(path.read_text())
    episodes = report.get("episodes", [])
    if not episodes:
        raise ValueError(f"{path} has no episodes")
    seeds = [episode["seed"] for episode in episodes]
    if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        raise ValueError(f"{path} episode seeds are not contiguous")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k2", type=Path, required=True)
    parser.add_argument("--k8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    k2, k8 = load(args.k2), load(args.k8)
    if k2.get("task") != k8.get("task"):
        raise ValueError("K=2 and K=8 reports must use the same task")
    k2_episodes, k8_episodes = k2["episodes"], k8["episodes"]
    if [episode["seed"] for episode in k2_episodes] != [episode["seed"] for episode in k8_episodes]:
        raise ValueError("K=2 and K=8 reports must use identical paired seeds")
    summary = paired_binary_summary(
        [episode["success"] for episode in k2_episodes],
        [episode["success"] for episode in k8_episodes],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    summary.update(
        {
            "task": k2["task"],
            "k2_checkpoint": k2.get("checkpoint"),
            "k8_checkpoint": k8.get("checkpoint"),
            "k2_success_rate": k2.get("success_rate"),
            "k8_success_rate": k8.get("success_rate"),
            "stop_rule_20pp_fired": summary["paired_rate_difference"] >= 0.20,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
