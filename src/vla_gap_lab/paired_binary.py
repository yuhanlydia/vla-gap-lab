"""Statistics for paired binary closed-loop outcomes."""

from __future__ import annotations

import math

import numpy as np


def paired_binary_summary(
    reference: list[bool] | np.ndarray,
    shifted: list[bool] | np.ndarray,
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 0,
) -> dict[str, float | int | list[float]]:
    """Summarize paired successes with an exact McNemar test and bootstrap CI."""
    ref = np.asarray(reference, dtype=np.int8)
    shift = np.asarray(shifted, dtype=np.int8)
    if ref.ndim != 1 or shift.ndim != 1 or ref.shape != shift.shape or ref.size == 0:
        raise ValueError("reference and shifted must be non-empty, equally sized vectors")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    ref_rate = float(ref.mean())
    shift_rate = float(shift.mean())
    ref_only = int(np.sum((ref == 1) & (shift == 0)))
    shift_only = int(np.sum((ref == 0) & (shift == 1)))
    discordant = ref_only + shift_only
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(ref_only, shift_only) + 1))
        mcnemar_p = min(1.0, 2.0 * tail / (2**discordant))
    else:
        mcnemar_p = 1.0

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, ref.size, size=(bootstrap_samples, ref.size))
    differences = (shift[indices] - ref[indices]).mean(axis=1)
    ci = np.quantile(differences, [0.025, 0.975]).tolist()
    retention = shift_rate / ref_rate if ref_rate else float("nan")
    return {
        "episodes": int(ref.size),
        "reference_success_rate": ref_rate,
        "shifted_success_rate": shift_rate,
        "control_retention": retention,
        "paired_rate_difference": shift_rate - ref_rate,
        "paired_bootstrap_95_ci": [float(ci[0]), float(ci[1])],
        "reference_only_successes": ref_only,
        "shifted_only_successes": shift_only,
        "mcnemar_exact_two_sided_p": mcnemar_p,
    }
