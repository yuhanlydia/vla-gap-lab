"""Metrics and preregistered Gate-0 decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    r2: float
    cosine: float


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> RegressionMetrics:
    target = target.reshape(len(target), -1)
    prediction = prediction.reshape(len(prediction), -1)
    numerator = np.sum(target * prediction, axis=1)
    denominator = np.linalg.norm(target, axis=1) * np.linalg.norm(prediction, axis=1)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return RegressionMetrics(
        mae=float(mean_absolute_error(target, prediction)),
        r2=float(r2_score(target, prediction, multioutput="variance_weighted")),
        cosine=float(cosine.mean()),
    )


def safe_retention(shifted: float, clean: float) -> float:
    return float(shifted / clean) if abs(clean) > 1e-12 else float("nan")


def latent_action_gate(
    clean_r2: np.ndarray,
    shifted_r2: np.ndarray,
    control_retention: float,
    representation_threshold: float = 0.8,
    control_threshold: float = 0.7,
) -> dict:
    retention = np.divide(
        shifted_r2,
        clean_r2,
        out=np.full_like(shifted_r2, np.nan, dtype=float),
        where=np.abs(clean_r2) > 1e-12,
    )
    eligible = np.flatnonzero(retention > representation_threshold)
    best = int(np.nanargmax(retention)) if np.isfinite(retention).any() else None
    passed = bool(len(eligible) and control_retention < control_threshold)
    return {
        "passed": passed,
        "representation_retention": retention.tolist(),
        "eligible_layer_indices": eligible.tolist(),
        "best_layer_index": best,
        "control_retention": float(control_retention),
        "thresholds": {
            "representation": representation_threshold,
            "control": control_threshold,
        },
    }


def metrics_dict(metrics: RegressionMetrics) -> dict:
    return asdict(metrics)

