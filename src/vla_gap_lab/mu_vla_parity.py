"""Reference-check utilities for released mu-VLA train-task parity."""

from __future__ import annotations


def evaluate_parity(
    observed: dict[str, float],
    references: dict[str, float],
    *,
    tolerance_pp: float = 15.0,
) -> dict:
    """Require every observed train-task SR to lie near its released reference."""
    if tolerance_pp < 0:
        raise ValueError("tolerance_pp must be non-negative")
    missing = sorted(set(references) - set(observed))
    if missing:
        raise ValueError(f"missing observed tasks: {missing}")
    rows = {}
    passed = True
    for task, reference in references.items():
        value = float(observed[task])
        error = round(100.0 * abs(value - float(reference)), 10)
        ok = error <= tolerance_pp
        rows[task] = {
            "observed_sr": value,
            "reference_sr": float(reference),
            "absolute_error_pp": error,
            "passed": ok,
        }
        passed = passed and ok
    return {
        "passed": passed,
        "tolerance_pp": float(tolerance_pp),
        "tasks": rows,
    }
