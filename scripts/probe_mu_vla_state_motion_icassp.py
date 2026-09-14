#!/usr/bin/env python3
"""Probe whether recurrent compression preserves motion available in visual history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA

from vla_gap_lab.dynamics_io import load_episode_npz
from vla_gap_lab.dynamics_probes import split_episode_ids
from vla_gap_lab.state_motion import (
    fit_mlp_probe,
    fit_ridge_probe,
    make_row_mask,
    select_memory_tokens,
    summarize_split_metrics,
    temporal_visual_features,
)

# representation, contact mode, probe, lag
FROZEN_CELLS = (
    ("memory", "pre_contact", "ridge", 0),
    ("memory", "pre_contact", "mlp", 0),
    ("visual_current", "pre_contact", "ridge", 0),
    ("visual_current", "pre_contact", "mlp", 0),
    ("visual_pair", "pre_contact", "ridge", 1),
    ("visual_pair", "pre_contact", "mlp", 1),
    # Reviewer-facing ablations.
    ("memory_stride8", "pre_contact", "ridge", 0),
    ("memory", "all_steps", "ridge", 0),
    ("visual_pair", "pre_contact", "ridge", 2),
)


def _cell_name(representation: str, contact_mode: str, probe: str, lag: int) -> str:
    suffix = f"_lag{lag}" if representation == "visual_pair" else ""
    return f"{representation}{suffix}__{contact_mode}__{probe}"


def _episode_paths(episodes_dir: Path, expected_episodes: int) -> list[Path]:
    files = sorted(episodes_dir.glob("episode_*.npz"))
    if len(files) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} episode files, found {len(files)}")
    for episode_id, path in enumerate(files):
        arrays, metadata = load_episode_npz(path)
        if metadata.get("study") != "icassp_motion_compression":
            raise ValueError(f"{path}: wrong study metadata")
        if metadata.get("episode") != episode_id:
            raise ValueError(f"{path}: episode metadata mismatch")
        memory = arrays.get("memory_after")
        visual = arrays.get("visual_tokens")
        if memory is None or memory.ndim != 3 or memory.shape[1] != 64:
            raise ValueError(f"{path}: expected memory_after [steps,64,hidden]")
        if visual is None or visual.ndim != 3 or visual.shape[1] != 8:
            raise ValueError(f"{path}: expected visual_tokens [steps,8,hidden]")
        if len(memory) != len(visual) or len(memory) != len(arrays["step"]):
            raise ValueError(f"{path}: row-count mismatch")
    return files


def _fit_token_pca(
    files: list[Path],
    train_ids: list[int],
    *,
    array_key: str,
    min_step: int,
    n_components: int,
) -> IncrementalPCA:
    pca = IncrementalPCA(n_components=n_components, batch_size=4096)
    for episode_id in train_ids:
        arrays, _ = load_episode_npz(files[episode_id])
        mask = np.asarray(arrays["step"]) >= int(min_step)
        values = arrays[array_key][mask].astype(np.float32)
        tokens = values.reshape(-1, values.shape[-1])
        if tokens.shape[0] < n_components:
            raise ValueError(f"episode {episode_id} has too few {array_key} rows for token PCA")
        pca.partial_fit(tokens)
    return pca


def _build_memory_dataset(
    files: list[Path],
    token_pca: IncrementalPCA,
    *,
    stride8: bool,
    min_step: int,
    pre_contact_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, position, velocity, groups = [], [], [], []
    for episode_id, path in enumerate(files):
        arrays, _ = load_episode_npz(path)
        mask = make_row_mask(
            arrays["step"], arrays["reached_status"],
            min_step=min_step, pre_contact_only=pre_contact_only,
        )
        memory = arrays["memory_after"][mask].astype(np.float32)
        flat_tokens = memory.reshape(-1, memory.shape[-1])
        compressed = token_pca.transform(flat_tokens).astype(np.float32)
        compressed = compressed.reshape(memory.shape[0], 64, -1)
        features.append(select_memory_tokens(compressed, "stride8" if stride8 else "full64"))
        position.append(arrays["ball_position_xy"][mask].astype(np.float32))
        velocity.append(arrays["ball_velocity_xy"][mask].astype(np.float32))
        groups.append(np.full(mask.sum(), episode_id, dtype=np.int32))
    return (
        np.concatenate(features), np.concatenate(position),
        np.concatenate(velocity), np.concatenate(groups),
    )


def _build_visual_dataset(
    files: list[Path],
    token_pca: IncrementalPCA,
    *,
    mode: str,
    lag: int,
    min_step: int,
    pre_contact_only: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features, position, velocity, groups = [], [], [], []
    for episode_id, path in enumerate(files):
        arrays, _ = load_episode_npz(path)
        visual = arrays["visual_tokens"].astype(np.float32)
        flat_tokens = visual.reshape(-1, visual.shape[-1])
        compressed = token_pca.transform(flat_tokens).astype(np.float32)
        compressed = compressed.reshape(visual.shape[0], 8, -1)
        x, current_indices = temporal_visual_features(compressed, mode=mode, lag=lag)
        base_mask = make_row_mask(
            arrays["step"], arrays["reached_status"],
            min_step=min_step, pre_contact_only=pre_contact_only,
        )
        valid = base_mask[current_indices]
        current_indices = current_indices[valid]
        features.append(x[valid])
        position.append(arrays["ball_position_xy"][current_indices].astype(np.float32))
        velocity.append(arrays["ball_velocity_xy"][current_indices].astype(np.float32))
        groups.append(np.full(valid.sum(), episode_id, dtype=np.int32))
    return (
        np.concatenate(features), np.concatenate(position),
        np.concatenate(velocity), np.concatenate(groups),
    )


def _fit_cell(
    x: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    groups: np.ndarray,
    train_ids: list[int],
    dev_ids: list[int],
    test_ids: list[int],
    *,
    sample_pca_dim: int,
    probe: str,
    split_seed: int,
) -> dict:
    masks = {
        "train": np.isin(groups, train_ids),
        "dev": np.isin(groups, dev_ids),
        "test": np.isin(groups, test_ids),
    }
    n_components = min(sample_pca_dim, int(masks["train"].sum()) - 1, x.shape[1])
    if n_components < 2:
        raise ValueError("not enough train rows for sample PCA")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=split_seed)
    pca.fit(x[masks["train"]])
    latent = {name: pca.transform(x[mask]).astype(np.float32) for name, mask in masks.items()}

    fitter = fit_ridge_probe if probe == "ridge" else fit_mlp_probe
    kwargs = {} if probe == "ridge" else {"seed": split_seed}
    _, position_metrics = fitter(
        latent["train"], position[masks["train"]],
        latent["dev"], position[masks["dev"]],
        latent["test"], position[masks["test"]], **kwargs,
    )
    _, velocity_metrics = fitter(
        latent["train"], velocity[masks["train"]],
        latent["dev"], velocity[masks["dev"]],
        latent["test"], velocity[masks["test"]], **kwargs,
    )
    return {
        "sample_pca_dim": n_components,
        "sample_pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "position": position_metrics,
        "velocity": velocity_metrics,
        "position_r2_mean": float(position_metrics["test_r2_mean"]),
        "velocity_y_r2": float(velocity_metrics["test_r2_per_dim"][1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--train-episodes", type=int, default=36)
    parser.add_argument("--dev-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=12)
    parser.add_argument("--split-seeds", default="0,1,2,3,4")
    parser.add_argument("--min-step", type=int, default=2)
    parser.add_argument("--token-pca-dim", type=int, default=32)
    parser.add_argument("--sample-pca-dim", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split_seeds = [int(item) for item in args.split_seeds.split(",") if item.strip()]
    if split_seeds != [0, 1, 2, 3, 4]:
        raise ValueError("the frozen ICASSP protocol requires split seeds 0,1,2,3,4")
    if (args.train_episodes, args.dev_episodes, args.test_episodes) != (36, 12, 12):
        raise ValueError("the frozen ICASSP split is 36/12/12 episodes")

    files = _episode_paths(args.episodes_dir, args.episodes)
    success_rates = []
    for path in files:
        _, metadata = load_episode_npz(path)
        success_rates.append(bool(metadata.get("success_once", False)))

    cells = []
    for split_seed in split_seeds:
        train_ids, dev_ids, test_ids = split_episode_ids(
            np.arange(args.episodes), train=36, dev=12, test=12, seed=split_seed
        )
        memory_token_pca = _fit_token_pca(
            files, train_ids, array_key="memory_after",
            min_step=args.min_step, n_components=args.token_pca_dim,
        )
        visual_token_pca = _fit_token_pca(
            files, train_ids, array_key="visual_tokens",
            min_step=args.min_step, n_components=args.token_pca_dim,
        )
        dataset_cache: dict[tuple, tuple] = {}
        for representation, contact_mode, probe, lag in FROZEN_CELLS:
            key = (representation, contact_mode, lag)
            if key not in dataset_cache:
                if representation in {"memory", "memory_stride8"}:
                    dataset_cache[key] = _build_memory_dataset(
                        files, memory_token_pca,
                        stride8=representation == "memory_stride8",
                        min_step=args.min_step,
                        pre_contact_only=contact_mode == "pre_contact",
                    )
                elif representation in {"visual_current", "visual_pair"}:
                    dataset_cache[key] = _build_visual_dataset(
                        files, visual_token_pca,
                        mode="current" if representation == "visual_current" else "pair",
                        lag=max(1, lag),
                        min_step=args.min_step,
                        pre_contact_only=contact_mode == "pre_contact",
                    )
                else:
                    raise ValueError(f"unknown representation: {representation}")
            x, position, velocity, groups = dataset_cache[key]
            result = _fit_cell(
                x, position, velocity, groups, train_ids, dev_ids, test_ids,
                sample_pca_dim=args.sample_pca_dim, probe=probe, split_seed=split_seed,
            )
            cells.append({
                "split_seed": split_seed,
                "train_episode_ids": train_ids,
                "dev_episode_ids": dev_ids,
                "test_episode_ids": test_ids,
                "representation": representation,
                "contact_mode": contact_mode,
                "probe": probe,
                "lag": lag,
                "token_pca_dim": args.token_pca_dim,
                **result,
            })

    aggregates = {}
    for representation, contact_mode, probe, lag in FROZEN_CELLS:
        rows = [
            cell for cell in cells
            if cell["representation"] == representation
            and cell["contact_mode"] == contact_mode
            and cell["probe"] == probe
            and cell["lag"] == lag
        ]
        aggregates[_cell_name(representation, contact_mode, probe, lag)] = summarize_split_metrics(rows)

    report = {
        "schema_version": 2,
        "study": "icassp_motion_compression",
        "question": "does recurrent compression preserve motion available in short visual history?",
        "task": args.task,
        "episodes_dir": str(args.episodes_dir),
        "episodes": args.episodes,
        "success_rate": float(np.mean(success_rates)),
        "split_seeds": split_seeds,
        "split": {"train": 36, "dev": 12, "test": 12},
        "min_step": args.min_step,
        "token_pca_dim": args.token_pca_dim,
        "sample_pca_dim": args.sample_pca_dim,
        "cells": cells,
        "aggregates": aggregates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"task": args.task, "success_rate": report["success_rate"], "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    main()
