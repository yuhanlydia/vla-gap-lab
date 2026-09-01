"""Cluster-aware comparisons for paired continuous diagnostic records."""

from __future__ import annotations

import numpy as np


def paired_cluster_mean_difference(
    reference: np.ndarray,
    shifted: np.ndarray,
    clusters: np.ndarray,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int | list[float]]:
    reference = np.asarray(reference, dtype=float)
    shifted = np.asarray(shifted, dtype=float)
    clusters = np.asarray(clusters)
    if reference.ndim != 1 or shifted.shape != reference.shape or clusters.shape != reference.shape:
        raise ValueError("reference, shifted, and clusters must be aligned vectors")
    unique = np.unique(clusters)
    if not len(unique):
        raise ValueError("at least one cluster is required")
    differences = shifted - reference
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(bootstrap_samples):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == cluster) for cluster in sampled])
        boot.append(float(differences[indices].mean()))
    ci = np.quantile(boot, [0.025, 0.975])
    return {
        "records": len(reference),
        "clusters": len(unique),
        "reference_mean": float(reference.mean()),
        "shifted_mean": float(shifted.mean()),
        "paired_mean_difference": float(differences.mean()),
        "cluster_bootstrap_95_ci": [float(ci[0]), float(ci[1])],
    }
