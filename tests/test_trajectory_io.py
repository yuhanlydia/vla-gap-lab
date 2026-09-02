from __future__ import annotations

import numpy as np
import pytest

from vla_gap_lab.trajectory_io import load_trajectory_for_resume, save_trajectory_atomic


def test_color_lamp_identity_fallback() -> None:
    # The collector supports both canonical ShellGame and color-lamp tasks;
    # keep this small contract test independent of the simulator.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "scripts" / "collect_mu_vla_memory_trajectory.py"
    spec = importlib.util.spec_from_file_location("collector", path)
    collector = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(collector)

    class Env:
        class Unwrapped:
            target_color = 2

        unwrapped = Unwrapped()

    assert collector.current_target_identity(Env()) == 2


def rows() -> dict[str, list]:
    return {
        "memory": [np.ones((2, 3))],
        "episode": [0],
        "step": [0],
        "phase": ["cue"],
        "target_mug": [1],
        "target_slot": [1],
        "completed_swaps": [0],
    }


def test_atomic_trajectory_roundtrip(tmp_path) -> None:
    path = tmp_path / "trajectory.npz"
    metadata = {"task": "task", "episodes": [{"episode": 0, "seed": 9}]}
    save_trajectory_atomic(path, rows(), metadata)
    loaded, loaded_metadata = load_trajectory_for_resume(path, {"task": "task"})
    assert loaded_metadata == metadata
    assert loaded["episode"] == [0]
    np.testing.assert_allclose(loaded["memory"], rows()["memory"])
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_rejects_mismatch_and_noncontiguous_episodes(tmp_path) -> None:
    path = tmp_path / "trajectory.npz"
    save_trajectory_atomic(path, rows(), {"task": "a", "episodes": []})
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_trajectory_for_resume(path, {"task": "b"})
    save_trajectory_atomic(path, rows(), {"task": "a", "episodes": [{"episode": 2}]})
    with pytest.raises(ValueError, match="contiguous"):
        load_trajectory_for_resume(path, {"task": "a"})
