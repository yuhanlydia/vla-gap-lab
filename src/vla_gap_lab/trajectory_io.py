"""Crash-safe storage and resume validation for long trajectory collection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

ARRAY_DTYPES = {
    "memory": np.float16,
    "episode": np.int32,
    "step": np.int32,
    "phase": None,
    "target_mug": np.int8,
    "target_slot": np.int8,
    "completed_swaps": np.int8,
}


def save_trajectory_atomic(path: Path, rows: dict[str, list], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.asarray(rows[key], dtype=dtype) if dtype is not None else np.asarray(rows[key])
        for key, dtype in ARRAY_DTYPES.items()
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle, **arrays, metadata=np.asarray(json.dumps(metadata, sort_keys=True))
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_trajectory_for_resume(path: Path, expected: dict) -> tuple[dict[str, list], dict]:
    source = np.load(path, allow_pickle=False)
    metadata = json.loads(str(source["metadata"]))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"resume metadata mismatch for {key}: {metadata.get(key)!r} != {value!r}")
    episodes = metadata.get("episodes", [])
    indices = [int(row["episode"]) for row in episodes]
    if indices != list(range(len(indices))):
        raise ValueError("resume episodes must be contiguous and zero-indexed")
    rows = {key: source[key].tolist() for key in ARRAY_DTYPES}
    return rows, metadata
