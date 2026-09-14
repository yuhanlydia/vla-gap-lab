"""Helpers for the ICASSP motion-preservation study."""

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


def pool_projector_tokens(projected, *, num_views: int = 2, grid: int = 2):
    """Coarsely pool same-backbone visual projector patches without extra learning."""
    import math
    import torch

    if not torch.is_tensor(projected) or projected.ndim != 3 or projected.shape[0] != 1:
        raise ValueError("projected features must have shape [1,patches,dim]")
    if num_views < 1 or grid < 1:
        raise ValueError("num_views and grid must be positive")
    total = int(projected.shape[1])
    if total % num_views:
        raise ValueError("patch count must be divisible by num_views")
    per_view = total // num_views
    side = int(math.isqrt(per_view))
    if side * side != per_view or side % grid:
        raise ValueError("patches per view must form a square grid divisible by pooling grid")
    hidden = int(projected.shape[2])
    block = side // grid
    values = projected[0].reshape(num_views, side, side, hidden)
    values = values.reshape(num_views, grid, block, grid, block, hidden).mean(dim=(2, 4))
    return values.reshape(num_views * grid * grid, hidden)


def temporal_visual_features(
    tokens: np.ndarray,
    *,
    mode: str,
    lag: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten current visual tokens or concatenate a previous/current pair.

    Returns features and the timestep indices that each row predicts. The pair
    representation is deliberately simple: the same frozen visual tokens from
    steps t-lag and t are concatenated, so any motion gain comes from temporal
    evidence rather than a learned temporal module.
    """
    values = np.asarray(tokens, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("tokens must have shape [steps,tokens,dim]")
    if lag < 1:
        raise ValueError("lag must be positive")
    if mode == "current":
        return (
            values.reshape(values.shape[0], -1),
            np.arange(values.shape[0], dtype=np.int64),
        )
    if mode == "pair":
        if values.shape[0] <= lag:
            raise ValueError("not enough steps for requested lag")
        previous = values[:-lag].reshape(values.shape[0] - lag, -1)
        current = values[lag:].reshape(values.shape[0] - lag, -1)
        return (
            np.concatenate([previous, current], axis=1),
            np.arange(lag, values.shape[0], dtype=np.int64),
        )
    raise ValueError("mode must be 'current' or 'pair'")


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
    """Fit the frozen nonlinear probe used to challenge linear-probe artifacts."""
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


def motion_compression_decision(
    *,
    medium_current: dict[str, float],
    medium_pair: dict[str, float],
    medium_memory: dict[str, float],
    fast_pair: dict[str, float],
    fast_memory: dict[str, float],
) -> dict[str, object]:
    """Frozen claim gate for the revised ICASSP motion-compression story."""
    conditions = {
        "medium_pair_velocity_ge_0_60": medium_pair["velocity_y_r2_median"] >= 0.60,
        "medium_pair_gain_over_current_ge_0_20": (
            medium_pair["velocity_y_r2_median"] - medium_current["velocity_y_r2_median"]
            >= 0.20
        ),
        "medium_pair_gain_over_memory_ge_0_15": (
            medium_pair["velocity_y_r2_median"] - medium_memory["velocity_y_r2_median"]
            >= 0.15
        ),
        "medium_memory_position_ge_0_65": medium_memory["position_r2_mean_median"] >= 0.65,
        "fast_pair_gt_memory": (
            fast_pair["velocity_y_r2_median"] > fast_memory["velocity_y_r2_median"]
        ),
        "fast_memory_position_ge_0_60": fast_memory["position_r2_mean_median"] >= 0.60,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    return {
        "go_icassp": not failed,
        "conditions": conditions,
        "failed_conditions": failed,
        "interpretation": (
            "motion_observable_but_weaker_after_recurrent_compression"
            if not failed
            else "claim_not_supported_stop_without_rescue"
        ),
    }


# Legacy gate retained for backward compatibility with the first ICASSP draft.
def paper_decision(
    medium_ridge: dict[str, float],
    medium_mlp: dict[str, float],
    fast_ridge: dict[str, float],
    fast_mlp: dict[str, float],
) -> dict[str, object]:
    conditions = {
        "medium_ridge_position_ge_0_65": medium_ridge["position_r2_mean_median"] >= 0.65,
        "medium_mlp_position_ge_0_65": medium_mlp["position_r2_mean_median"] >= 0.65,
        "medium_ridge_gap_ge_0_25": medium_ridge["state_motion_gap_median"] >= 0.25,
        "medium_mlp_gap_ge_0_25": medium_mlp["state_motion_gap_median"] >= 0.25,
        "fast_ridge_state_gt_motion": fast_ridge["state_motion_gap_median"] > 0.0,
        "fast_mlp_state_gt_motion": fast_mlp["state_motion_gap_median"] > 0.0,
        "mlp_velocity_y_below_0_60": medium_mlp["velocity_y_r2_median"] < 0.60,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    return {"go_icassp": not failed, "conditions": conditions, "failed_conditions": failed}
