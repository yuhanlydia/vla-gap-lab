#!/usr/bin/env python3
"""Evaluate held-out linear state transport between paired embodiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-8)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-8)
    return left @ right.T


def _retrieval_stats(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    similarity = _cosine_rows(left, right)
    diagonal = np.diag(similarity)
    off_diagonal = similarity.copy()
    np.fill_diagonal(off_diagonal, -np.inf)
    return np.asarray(
        [
            np.sum(np.argmax(similarity, axis=1) == np.arange(len(left))),
            len(left),
            np.sum(diagonal),
            np.sum(diagonal - np.max(off_diagonal, axis=1)),
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source", default="aloha-agilex")
    parser.add_argument("--target", default="franka-panda")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--null-permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = np.load(args.cache, allow_pickle=False)
    seeds = np.intersect1d(
        data["seed"][data["embodiment"] == args.source],
        data["seed"][data["embodiment"] == args.target],
    )
    folds = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(seeds))
    rng = np.random.default_rng(args.seed)
    results = []
    metadata = json.loads(str(data["metadata"]))
    for stack_name, layer_ids in (
        ("vlm", metadata["vlm_layers"]),
        ("control", metadata["action_layers"]),
    ):
        stack = data[stack_name]
        for layer_col, layer_id in enumerate(layer_ids):

            def select(
                embodiment: str,
                stack: np.ndarray = stack,
                layer_col: int = layer_col,
            ) -> np.ndarray:
                return np.stack(
                    [
                        stack[(data["embodiment"] == embodiment) & (data["seed"] == seed)][
                            0, layer_col
                        ]
                        for seed in seeds
                    ]
                )

            source_all, target_all = select(args.source), select(args.target)
            totals = np.zeros(4, dtype=np.float64)
            raw_totals = np.zeros(4, dtype=np.float64)
            null_hits = 0.0
            for train_indices, test_indices in folds:
                mapping = Ridge(alpha=args.alpha).fit(
                    source_all[train_indices], target_all[train_indices]
                )
                totals += _retrieval_stats(
                    mapping.predict(source_all[test_indices]), target_all[test_indices]
                )
                raw_totals += _retrieval_stats(source_all[test_indices], target_all[test_indices])
                for _ in range(args.null_permutations):
                    shuffled = rng.permutation(train_indices)
                    null_mapping = Ridge(alpha=args.alpha).fit(
                        source_all[train_indices], target_all[shuffled]
                    )
                    null_hits += _retrieval_stats(
                        null_mapping.predict(source_all[test_indices]), target_all[test_indices]
                    )[0]
            results.append(
                {
                    "stack": stack_name,
                    "layer": layer_id,
                    "transport_retrieval": float(totals[0] / totals[1]),
                    "raw_retrieval": float(raw_totals[0] / raw_totals[1]),
                    "permuted_train_retrieval": float(
                        null_hits / (args.null_permutations * totals[1])
                    ),
                    "transport_paired_cosine": float(totals[2] / totals[1]),
                    "transport_margin": float(totals[3] / totals[1]),
                }
            )
    report = {
        "source": args.source,
        "target": args.target,
        "seeds": seeds.tolist(),
        "folds": args.folds,
        "null_permutations": args.null_permutations,
        "chance_by_fold": float(args.folds / len(seeds)),
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
