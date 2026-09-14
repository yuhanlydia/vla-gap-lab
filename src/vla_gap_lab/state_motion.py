"""Helpers for the ICASSP state-motion accessibility study."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def select_memory_tokens(memory: np.ndarray, mode: str) -> np.ndarray:
    """Flatten full or legacy stride-8 recurrent memory tokens per row."""
    values = np.asarray(memory)
    if values.ndim != 3:
        raise ValueError("memory must have shape [rows,tokens,dim]")
    if values.shape[1] != 64:
        raise ValueError(f"expected 64 memory tokens, got {values.shape[1]}")
    if mode == "full64":
        selected = values
    elif mode == "stride8":
        selected = values[:, ::8, :]
    else:
        raise ValueError("mode must be 'full64' or 'stride8'")
    return selected.astype(np.float32, copy=False).reshape(selected.shape[0], -1)


def make_row_mask(
    step: np.ndarray,
    reached_status: np.ndarray,
    *,
    min_step: int = 2,
    pre_contact_only: bool = True,
) -> np.ndarray:
    """Build the frozen row mask for primary and contact-ablation probes."""
    steps = np.asarray(step)
    reached = np.asarray(reached_status)
    if steps.shape[0] != reached.shape[0]:
        raise ValueError("step and reached_status must have the same length")
    mask = steps >= int(min_step)
    if pre_contact_only:
        mask &= reached < 0.5
    return mask.astype(bool, copy=False)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[float]]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    per_dim = np.atleast_1d(r2_score(true, pred, multioutput="raw_values")).astype(float)
    return {
        "test_r2_mean": float(np.mean(per_dim)),
        "test_r2_per_dim": [float(value) for value in per_dim],
        "test_mae": float(mean_absolute_error(true, pred)),
    }


def fit_ridge_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
):
    """Select ridge strength on dev only, then evaluate once on test."""
    best = None
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError("ridge alphas must be positive")
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        model.fit(train_x, train_y)
        dev_r2 = float(r2_score(dev_y, model.predict(dev_x), multioutput="uniform_average"))
        candidate = (dev_r2, -float(alpha), model, float(alpha))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    report = {"alpha": best[3], "dev_r2_mean": best[0]}
    report.update(_metrics(test_y, best[2].predict(test_x)))
    return best[2], report


def fit_mlp_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    dev_x: np.ndarray,
    dev_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int,
):
    """Fit the single frozen nonlinear probe used to challenge the linear finding."""
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=int(seed),
        ),
    )
    model.fit(train_x, train_y)
    dev_r2 = float(r2_score(dev_y, model.predict(dev_x), multioutput="uniform_average"))
    report = {"dev_r2_mean": dev_r2}
    report.update(_metrics(test_y, model.predict(test_x)))
    return model, report


def summarize_split_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate split-level paper metrics by median and interquartile range."""
    if not rows:
        raise ValueError("rows must be non-empty")
    pos = np.asarray([row["position_r2_mean"] for row in rows], dtype=np.float64)
    vel = np.asarray([row["velocity_y_r2"] for row in rows], dtype=np.float64)
    gap = pos - vel

    def stats(name: str, values: np.ndarray) -> dict[str, float]:
        return {
            f"{name}_median": float(np.median(values)),
            f"{name}_q25": float(np.quantile(values, 0.25)),
            f"{name}_q75": float(np.quantile(values, 0.75)),
        }

    return {
        **stats("position_r2_mean", pos),
        **stats("velocity_y_r2", vel),
        **stats("state_motion_gap", gap),
    }
