"""Leakage-safe regression helpers for recurrent dynamics diagnostics."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_episode_ids(
    episode_ids: np.ndarray | list[int],
    *,
    train: int,
    dev: int,
    test: int,
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Return deterministic disjoint episode-level train/dev/test splits."""
    ids = np.asarray(episode_ids, dtype=np.int64)
    unique = np.unique(ids)
    if train < 1 or dev < 1 or test < 1 or train + dev + test > len(unique):
        raise ValueError("train/dev/test counts must be positive and fit available episodes")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    return (
        [int(value) for value in shuffled[:train]],
        [int(value) for value in shuffled[train : train + dev]],
        [int(value) for value in shuffled[train + dev : train + dev + test]],
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[float]]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    per_dim = np.atleast_1d(
        r2_score(true, pred, multioutput="raw_values")
    ).astype(float)
    return {
        "r2_mean": float(np.mean(per_dim)),
        "r2_per_dim": [float(value) for value in per_dim],
        "mae": float(mean_absolute_error(true, pred)),
    }


def select_ridge_regression(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
):
    """Select ridge strength on dev data and report untouched test performance."""
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("alphas must be positive")
    best = None
    for alpha in alphas:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha, solver="lsqr"),
        )
        model.fit(train_x, train_y)
        dev_pred = model.predict(dev_x)
        dev_r2 = float(r2_score(dev_y, dev_pred, multioutput="uniform_average"))
        candidate = (dev_r2, -float(alpha), model, float(alpha))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    model = best[2]
    report = {"alpha": best[3], "dev_r2_mean": best[0]}
    report.update(
        {
            f"test_{key}": value
            for key, value in _metrics(test_y, model.predict(test_x)).items()
        }
    )
    return model, report
