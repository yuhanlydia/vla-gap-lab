"""Layerwise linear action probes with leakage-safe matched splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from .cache import HiddenCache
from .metrics import RegressionMetrics, regression_metrics


@dataclass
class LayerProbeResult:
    layer: int
    clean: RegressionMetrics
    shifted: RegressionMetrics
    model: Ridge


def matched_split(sample_id: np.ndarray, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split by unique simulator state, never by rendered view."""
    unique = np.unique(sample_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, round(len(unique) * validation_fraction))
    validation_ids = set(unique[:n_val].tolist())
    validation = np.array([item in validation_ids for item in sample_id])
    return ~validation, validation


def fit_layerwise_ridge(
    cache: HiddenCache,
    alpha: float = 1.0,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> list[LayerProbeResult]:
    cache.validate()
    train, validation = matched_split(cache.sample_id, validation_fraction, seed)
    targets = cache.actions.reshape(len(cache.actions), -1)
    results = []
    for index, layer in enumerate(cache.layers):
        model = Ridge(alpha=alpha)
        model.fit(cache.clean[train, index], targets[train])
        clean_prediction = model.predict(cache.clean[validation, index])
        shifted_prediction = model.predict(cache.shifted[validation, index])
        results.append(
            LayerProbeResult(
                layer=int(layer),
                clean=regression_metrics(targets[validation], clean_prediction),
                shifted=regression_metrics(targets[validation], shifted_prediction),
                model=model,
            )
        )
    return results

