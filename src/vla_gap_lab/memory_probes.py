"""Leakage-safe probes for persistent and revised memory content."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def grouped_ridge_classification(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float = 100.0,
    folds: int = 4,
    bootstrap_samples: int = 2_000,
    seed: int = 0,
) -> dict[str, float | int | list[float]]:
    """Cross-validate a ridge classifier while holding out whole episodes."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(groups):
        raise ValueError("features, labels, and groups must have aligned rows")
    if n_splits < 2:
        raise ValueError("at least two groups are required")

    predictions = np.empty_like(labels)
    covered = np.zeros(len(labels), dtype=bool)
    for train, test in GroupKFold(n_splits=n_splits).split(features, labels, groups):
        if len(np.unique(labels[train])) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            RidgeClassifier(alpha=alpha, class_weight="balanced"),
        )
        model.fit(features[train], labels[train])
        predictions[test] = model.predict(features[test])
        covered[test] = True
    if not covered.any():
        raise ValueError("no fold contained at least two training classes")
    labels_covered = labels[covered]
    predictions_covered = predictions[covered]
    groups_covered = groups[covered]
    group_values = np.unique(groups_covered)
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(bootstrap_samples):
        sampled_groups = rng.choice(group_values, size=len(group_values), replace=True)
        sampled_rows = np.concatenate(
            [np.flatnonzero(groups_covered == group) for group in sampled_groups]
        )
        bootstrap.append(
            recall_score(
                labels_covered[sampled_rows],
                predictions_covered[sampled_rows],
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        )
    ci = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {
        "samples": int(covered.sum()),
        "episodes": len(unique_groups),
        "accuracy": float(np.mean(predictions_covered == labels_covered)),
        "balanced_accuracy": float(
            recall_score(
                labels_covered,
                predictions_covered,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy_cluster_bootstrap_95_ci": [float(ci[0]), float(ci[1])],
    }
