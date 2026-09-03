import numpy as np

from vla_gap_lab.dynamics_io import load_episode_npz, save_episode_npz_atomic


def test_atomic_episode_npz_roundtrip(tmp_path):
    path = tmp_path / "episode.npz"
    arrays = {
        "step": np.arange(3, dtype=np.int32),
        "velocity": np.ones((3, 2), dtype=np.float32),
    }
    metadata = {"seed": 7, "task": "InterceptMedium-VLA-v0"}
    save_episode_npz_atomic(path, arrays, metadata)
    loaded, restored = load_episode_npz(path)
    np.testing.assert_array_equal(loaded["step"], arrays["step"])
    np.testing.assert_allclose(loaded["velocity"], arrays["velocity"])
    assert restored == metadata
