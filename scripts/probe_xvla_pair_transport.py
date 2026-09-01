#!/usr/bin/env python3
"""Evaluate held-out linear state transport between paired embodiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from vla_gap_lab.multiple_testing import bonferroni, holm
from vla_gap_lab.state_transport import dual_ridge_predict


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
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
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
                selected = np.stack(
                    [
                        stack[(data["embodiment"] == embodiment) & (data["seed"] == seed)][
                            0, layer_col
                        ]
                        for seed in seeds
                    ]
                )
                return selected.reshape(len(selected), -1)

            source_all, target_all = select(args.source), select(args.target)
            totals = np.zeros(4, dtype=np.float64)
            raw_totals = np.zeros(4, dtype=np.float64)
            null_hits = np.zeros(args.null_permutations, dtype=np.float64)
            for train_indices, test_indices in folds:
                totals += _retrieval_stats(
                    dual_ridge_predict(
                        source_all[train_indices],
                        target_all[train_indices],
                        source_all[test_indices],
                        alpha=args.alpha,
                        standardize=args.standardize,
                    ),
                    target_all[test_indices],
                )
                raw_totals += _retrieval_stats(source_all[test_indices], target_all[test_indices])
                for permutation_index in range(args.null_permutations):
                    shuffled = rng.permutation(train_indices)
                    null_hits[permutation_index] += _retrieval_stats(
                        dual_ridge_predict(
                            source_all[train_indices],
                            target_all[shuffled],
                            source_all[test_indices],
                            alpha=args.alpha,
                            standardize=args.standardize,
                        ),
                        target_all[test_indices],
                    )[0]
            results.append(
                {
                    "stack": stack_name,
                    "layer": layer_id,
                    "transport_retrieval": float(totals[0] / totals[1]),
                    "raw_retrieval": float(raw_totals[0] / raw_totals[1]),
                    "permuted_train_retrieval": float(
                        null_hits.sum() / (args.null_permutations * totals[1])
                    ),
                    "permutation_retrieval_95_percentile": float(
                        np.quantile(null_hits / totals[1], 0.95)
                    ),
                    "permutation_p_value": float(
                        (1 + np.sum(null_hits >= totals[0])) / (args.null_permutations + 1)
                    ),
                    "transport_paired_cosine": float(totals[2] / totals[1]),
                    "transport_margin": float(totals[3] / totals[1]),
                }
            )
    raw_p_values = [row["permutation_p_value"] for row in results]
    for row, bonferroni_p, holm_p in zip(
        results, bonferroni(raw_p_values), holm(raw_p_values)
    ):
        row["bonferroni_p_value_across_layers"] = bonferroni_p
        row["holm_p_value_across_layers"] = holm_p
    report = {
        "schema_version": 2,
        "cache": str(args.cache),
        "source": args.source,
        "target": args.target,
        "alpha": args.alpha,
        "seed": args.seed,
        "seeds": seeds.tolist(),
        "folds": args.folds,
        "null_permutations": args.null_permutations,
        "standardize_source": args.standardize,
        "chance_by_fold": float(args.folds / len(seeds)),
        "multiple_testing_family": "all VLM and control layers in this report",
        "multiple_testing_family_size": len(results),
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
