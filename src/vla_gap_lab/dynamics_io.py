"""Crash-safe per-episode NPZ storage for predictive-dynamics probes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def save_episode_npz_atomic(
    path: Path, arrays: dict[str, np.ndarray], metadata: dict
) -> None:
    """Atomically store one episode so long 16GB runs can resume safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = {len(np.asarray(value)) for value in arrays.values()}
    if len(lengths) > 1:
        raise ValueError("all episode arrays must have the same first dimension")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_episode_npz(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load one episode and its JSON metadata without pickle."""
    with np.load(path, allow_pickle=False) as source:
        metadata = json.loads(str(source["metadata"]))
        arrays = {
            key: source[key].copy() for key in source.files if key != "metadata"
        }
    return arrays, metadata
