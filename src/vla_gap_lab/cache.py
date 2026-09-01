"""Portable hidden-state cache used by all three tracks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HiddenCache:
    """Matched clean/shift states and action targets.

    Hidden arrays are shaped ``[N, L, D]`` and actions ``[N, H, A]``.
    ``sample_id`` guarantees that clean and shifted rows describe the same
    simulator state and therefore share the same expert action.
    """

    clean: np.ndarray
    shifted: np.ndarray
    actions: np.ndarray
    layers: np.ndarray
    sample_id: np.ndarray
    shift: np.ndarray

    def validate(self) -> None:
        if self.clean.shape != self.shifted.shape or self.clean.ndim != 3:
            raise ValueError("clean and shifted must both have shape [N, L, D]")
        n, num_layers, _ = self.clean.shape
        if self.actions.ndim != 3 or self.actions.shape[0] != n:
            raise ValueError("actions must have shape [N, H, A]")
        if self.layers.shape != (num_layers,):
            raise ValueError("layers must have one entry per cached layer")
        if self.sample_id.shape != (n,) or self.shift.shape != (n,):
            raise ValueError("sample_id and shift must have shape [N]")
        for name, value in (
            ("clean", self.clean),
            ("shifted", self.shifted),
            ("actions", self.actions),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            clean=self.clean,
            shifted=self.shifted,
            actions=self.actions,
            layers=self.layers,
            sample_id=self.sample_id,
            shift=self.shift,
        )

    @classmethod
    def load(cls, path: str | Path) -> HiddenCache:
        with np.load(path, allow_pickle=False) as data:
            cache = cls(**{key: data[key] for key in cls.__dataclass_fields__})
        cache.validate()
        return cache
