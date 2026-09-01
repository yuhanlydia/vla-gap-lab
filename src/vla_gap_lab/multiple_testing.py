"""Dependency-free family-wise p-value corrections."""

from __future__ import annotations

import numpy as np


def bonferroni(p_values: list[float]) -> list[float]:
    count = len(p_values)
    return [min(1.0, float(value) * count) for value in p_values]


def holm(p_values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1.0, adjusted_sorted)
    return adjusted.tolist()
