import numpy as np
import cv2
import h5py
import json

from vla_gap_lab.robotwin_data import pose_gripper_to_ee6d, sample_episode


def test_robotwin_proprio_conversion_identity_quaternions():
    pose = np.array([[1, 2, 3, 0, 0, 0, 1]], dtype=float)
    result = pose_gripper_to_ee6d(pose, np.array([[0.25]]), pose, np.array([[0.75]]))
    assert result.shape == (1, 20)
    np.testing.assert_allclose(result[0, :3], [1, 2, 3])
    np.testing.assert_allclose(result[0, 9], 0.5)
    np.testing.assert_allclose(result[0, 19], -0.5)


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((4, 5, 3), color, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_sample_episode_reads_xpolicylab_target_schema(tmp_path):
    data_dir = tmp_path / "franka" / "data"
    data_dir.mkdir(parents=True)
    path = data_dir / "episode_0000000.hdf5"
    length = 3
    with h5py.File(path, "w") as handle:
        handle.create_dataset("endpose/left_endpose", data=np.tile([1, 2, 3, 0, 0, 0, 1], (length, 1)))
        handle.create_dataset("endpose/right_endpose", data=np.tile([4, 5, 6, 0, 0, 0, 1], (length, 1)))
        handle.create_dataset("endpose/left_gripper", data=np.zeros(length))
        handle.create_dataset("endpose/right_gripper", data=np.ones(length))
        for camera in ("head_camera", "left_camera", "right_camera"):
            handle.create_dataset(
                f"observation/{camera}/rgb",
                data=np.asarray([_jpeg_bytes((10, 20, 30))] * length),
            )
    instruction_dir = path.parent.parent / "instruction"
    instruction_dir.mkdir()
    (instruction_dir / "episode_0000000.json").write_text(
        json.dumps({"seen": ["place the blocks"], "unseen": []})
    )

    result = sample_episode(path, num_frames=2)

    assert result["images"].shape == (2, 3, 4, 5, 3)
    assert result["proprio"].shape == (2, 20)
    assert result["instruction"] == "place the blocks"


def test_sample_episode_keeps_reading_legacy_aloha_schema(tmp_path):
    path = tmp_path / "legacy.hdf5"
    length = 3
    with h5py.File(path, "w") as handle:
        for side, xyz in (("left", [1, 2, 3]), ("right", [4, 5, 6])):
            handle.create_dataset(
                f"state/{side}_ee_poses",
                data=np.tile([*xyz, 0, 0, 0, 1], (length, 1)),
            )
            handle.create_dataset(f"state/{side}_ee_joint_states", data=np.zeros(length))
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            handle.create_dataset(
                f"vision/{camera}/colors",
                data=np.asarray([_jpeg_bytes((10, 20, 30))] * length),
            )
        handle.create_dataset("instruction", data=np.bytes_("legacy task"))

    result = sample_episode(path, num_frames=2)

    assert result["images"].shape == (2, 3, 4, 5, 3)
    assert result["proprio"].shape == (2, 20)
    assert result["instruction"] == "legacy task"
