"""Utilities for constructing action-preserving LIBERO-Plus image pairs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_PRESERVING_CATEGORIES = frozenset(
    {"Background Textures", "Camera View", "Lighting", "Sensor Noise"}
)


@dataclass(frozen=True)
class Perturbation:
    suite: str
    task_id: int
    name: str
    category: str
    difficulty_level: int


def load_perturbation(classification_file: str | Path, suite: str, task_id: int) -> Perturbation:
    """Resolve one official LIBERO-Plus task and reject unknown identifiers."""
    catalog = json.loads(Path(classification_file).read_text())
    try:
        item = next(row for row in catalog[suite] if int(row["id"]) == task_id)
    except (KeyError, StopIteration) as exc:
        raise ValueError(f"unknown perturbation: {suite} task {task_id}") from exc
    return Perturbation(
        suite=suite,
        task_id=task_id,
        **{k: item[k] for k in ("name", "category", "difficulty_level")},
    )


def require_action_preserving(perturbation: Perturbation) -> None:
    """Guard the paired-action diagnostic against dynamics-changing shifts."""
    if perturbation.category not in ACTION_PRESERVING_CATEGORIES:
        allowed = ", ".join(sorted(ACTION_PRESERVING_CATEGORIES))
        raise ValueError(f"{perturbation.category!r} is not action-preserving; allowed: {allowed}")


def instruction_slug(instruction: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", instruction.lower()).strip("_")


def demonstration_instruction(path: str | Path) -> str:
    """Recover the clean policy instruction from an official demo filename."""
    stem = Path(path).name.removesuffix("_demo.hdf5")
    return stem.replace("_", " ")


def find_demonstration(
    dataset_dir: str | Path, instruction: str, variant_name: str | None = None
) -> Path:
    """Map the BDDL language instruction back to the original LIBERO HDF5."""
    expected = Path(dataset_dir) / f"{instruction_slug(instruction)}_demo.hdf5"
    if expected.is_file():
        return expected
    # Perturbed BDDL files sometimes append visual-variant tokens to the parsed
    # language. Match the longest original demonstration stem that prefixes the
    # official variant name instead of encoding every upstream naming scheme.
    if variant_name is not None:
        matches = [
            path
            for path in Path(dataset_dir).glob("*_demo.hdf5")
            if variant_name.startswith(path.name.removesuffix("_demo.hdf5"))
        ]
        if matches:
            return max(matches, key=lambda path: len(path.name))
    raise FileNotFoundError(f"demonstration not found: {expected}")


def evenly_spaced_indices(length: int, count: int, margin: int = 0) -> np.ndarray:
    """Return deterministic unique frame indices, avoiding optional edge frames."""
    if count < 1 or length <= 2 * margin:
        raise ValueError("invalid length/count/margin")
    count = min(count, length - 2 * margin)
    return np.unique(np.linspace(margin, length - margin - 1, count, dtype=np.int64))
