#!/usr/bin/env python3
"""Diagnose whether recurrent memory stores state but fails to encode dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA

from vla_gap_lab.dynamics_io import load_episode_npz
from vla_gap_lab.dynamics_probes import select_ridge_regression, split_episode_ids


def parse_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("grid values must be positive")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--dev-episodes", type=int, default=8)
    parser.add_argument("--test-episodes", type=int, default=8)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--pca-batch-size", type=int, default=256)
    parser.add_argument(
        "--alpha-grid", type=parse_grid, default=parse_grid("0.1,1,10,100")
    )
    parser.add_argument("--min-step", type=int, default=2)
    parser.add_argument("--include-post-contact", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.episodes_dir.glob("episode_*.npz"))
    if not files:
        raise ValueError("no episode_*.npz files found")
    episodes = []
    for episode_id, path in enumerate(files):
        arrays, metadata = load_episode_npz(path)
        length = len(arrays["step"])
        arrays = {key: value for key, value in arrays.items() if len(value) == length}
        episodes.append((episode_id, path, arrays, metadata))

    train_ids, dev_ids, test_ids = split_episode_ids(
        np.arange(len(episodes)),
        train=args.train_episodes,
        dev=args.dev_episodes,
        test=args.test_episodes,
        seed=args.seed,
    )
    features = {name: [] for name in ("memory_before", "memory_after", "memory_delta")}
    targets = {name: [] for name in ("position_xy", "velocity_xy", "initial_velocity_xy")}
    groups = []
    for episode_id, _, arrays, _ in episodes:
        mask = arrays["step"] >= args.min_step
        if not args.include_post_contact:
            mask &= arrays["reached_status"] < 0.5
        before = arrays["memory_before"][mask].astype(np.float32).reshape(mask.sum(), -1)
        after = arrays["memory_after"][mask].astype(np.float32).reshape(mask.sum(), -1)
        features["memory_before"].append(before)
        features["memory_after"].append(after)
        features["memory_delta"].append(after - before)
        targets["position_xy"].append(arrays["ball_position_xy"][mask])
        targets["velocity_xy"].append(arrays["ball_velocity_xy"][mask])
        targets["initial_velocity_xy"].append(arrays["initial_velocity_xy"][mask])
        groups.append(np.full(mask.sum(), episode_id, dtype=np.int32))

    features = {
        key: np.concatenate(value, axis=0) for key, value in features.items()
    }
    targets = {key: np.concatenate(value, axis=0) for key, value in targets.items()}
    group = np.concatenate(groups)
    split_masks = {
        "train": np.isin(group, train_ids),
        "dev": np.isin(group, dev_ids),
        "test": np.isin(group, test_ids),
    }
    report = {
        "schema_version": 1,
        "episodes_dir": str(args.episodes_dir),
        "train_episode_ids": train_ids,
        "dev_episode_ids": dev_ids,
        "test_episode_ids": test_ids,
        "min_step": args.min_step,
        "pre_contact_only": not args.include_post_contact,
        "features": {},
    }

    for feature_name, values in features.items():
        n_components = min(
            args.pca_dim,
            int(split_masks["train"].sum()) - 1,
            values.shape[1],
        )
        if n_components < 2:
            raise ValueError("not enough train rows for PCA")
        pca = IncrementalPCA(
            n_components=n_components,
            batch_size=max(n_components, args.pca_batch_size),
        )
        pca.fit(values[split_masks["train"]])
        latent = {
            name: pca.transform(values[mask]).astype(np.float32)
            for name, mask in split_masks.items()
        }
        feature_report = {
            "pca_dim": n_components,
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "targets": {},
        }
        for target_name, target_values in targets.items():
            _, metrics = select_ridge_regression(
                latent["train"],
                target_values[split_masks["train"]],
                latent["dev"],
                target_values[split_masks["dev"]],
                latent["test"],
                target_values[split_masks["test"]],
                alphas=args.alpha_grid,
            )
            feature_report["targets"][target_name] = metrics
        report["features"][feature_name] = feature_report

    position_r2 = report["features"]["memory_after"]["targets"]["position_xy"][
        "test_r2_mean"
    ]
    velocity_r2 = report["features"]["memory_after"]["targets"]["velocity_xy"][
        "test_r2_mean"
    ]
    delta_velocity_r2 = report["features"]["memory_delta"]["targets"][
        "velocity_xy"
    ]["test_r2_mean"]
    if position_r2 >= 0.50 and velocity_r2 <= 0.20:
        diagnosis = "storage_dynamics_gap"
    elif velocity_r2 >= 0.50:
        diagnosis = "dynamics_represented_test_control_utilization_next"
    elif delta_velocity_r2 - velocity_r2 >= 0.15:
        diagnosis = "dynamics_concentrated_in_memory_update"
    else:
        diagnosis = "no_clear_dynamics_signal"
    report["gate"] = {
        "position_r2_after": position_r2,
        "velocity_r2_after": velocity_r2,
        "velocity_r2_delta": delta_velocity_r2,
        "diagnosis": diagnosis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
