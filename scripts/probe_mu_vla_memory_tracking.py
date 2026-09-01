#!/usr/bin/env python3
"""Probe whether mu-VLA memory retains identity and revises target location."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_gap_lab.memory_probes import grouped_ridge_classification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(args.trajectory, allow_pickle=False)
    memory = data["memory"].astype(np.float32)
    memory = memory.reshape(len(memory), -1)
    episode = data["episode"]
    first_by_episode = {item: memory[np.flatnonzero(episode == item)[0]] for item in np.unique(episode)}
    delta = np.stack([row - first_by_episode[item] for row, item in zip(memory, episode)])

    segments = [(phase_name, data["phase"] == phase_name) for phase_name in ("cue", "shuffle", "manipulation")]
    for completed in sorted(np.unique(data["completed_swaps"][data["phase"] == "shuffle"])):
        segments.append(
            (
                f"shuffle_after_{int(completed)}_swaps",
                (data["phase"] == "shuffle") & (data["completed_swaps"] == completed),
            )
        )

    results = []
    for feature_name, features in (("raw", memory), ("delta_from_reset", delta)):
        for segment_name, mask in segments:
            for label_name in ("target_mug", "target_slot"):
                try:
                    metrics = grouped_ridge_classification(
                        features[mask],
                        data[label_name][mask],
                        episode[mask],
                        alpha=args.alpha,
                        folds=args.folds,
                    )
                except ValueError as error:
                    metrics = {"error": str(error)}
                results.append(
                    {
                        "features": feature_name,
                        "segment": segment_name,
                        "label": label_name,
                        **metrics,
                    }
                )
    report = {
        "schema_version": 1,
        "trajectory": str(args.trajectory),
        "alpha": args.alpha,
        "folds": args.folds,
        "chance_balanced_accuracy": 1 / 3,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
