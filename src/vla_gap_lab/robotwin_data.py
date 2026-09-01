"""Read official RoboTwin XPolicyLab HDF5 trajectories."""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def decode_jpeg(value: np.bytes_) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid JPEG frame")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def pose_gripper_to_ee6d(
    left_pose: np.ndarray,
    left_gripper: np.ndarray,
    right_pose: np.ndarray,
    right_gripper: np.ndarray,
) -> np.ndarray:
    """Match the official X-VLA RoboTwin client's 20D proprio conversion."""
    left_rot = Rotation.from_quat(left_pose[..., 3:]).as_matrix()[..., :, :2].reshape(-1, 6)
    right_rot = Rotation.from_quat(right_pose[..., 3:]).as_matrix()[..., :, :2].reshape(-1, 6)
    return np.concatenate(
        [
            left_pose[..., :3],
            left_rot,
            1 - 2 * left_gripper.reshape(-1, 1),
            right_pose[..., :3],
            right_rot,
            1 - 2 * right_gripper.reshape(-1, 1),
        ],
        axis=-1,
    ).astype(np.float32)


def sample_episode(path: str | Path, num_frames: int) -> dict[str, np.ndarray | str]:
    """Sample deterministic normalized-progress points from one episode."""
    with h5py.File(path, "r") as handle:
        length = len(handle["state/left_ee_poses"])
        indices = np.unique(np.linspace(0, length - 1, min(num_frames, length), dtype=np.int64))
        images = np.stack(
            [
                np.stack(
                    [decode_jpeg(handle[f"vision/{camera}/colors"][index]) for camera in CAMERAS]
                )
                for index in indices
            ]
        )
        proprio = pose_gripper_to_ee6d(
            handle["state/left_ee_poses"][indices],
            handle["state/left_ee_joint_states"][indices],
            handle["state/right_ee_poses"][indices],
            handle["state/right_ee_joint_states"][indices],
        )
        instruction = handle["instruction"][()].decode()
    progress = indices.astype(np.float32) / max(1, length - 1)
    return {
        "images": images,
        "proprio": proprio,
        "indices": indices,
        "progress": progress,
        "instruction": instruction,
    }
