"""Read official RoboTwin XPolicyLab HDF5 trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
TARGET_CAMERAS = ("head_camera", "left_camera", "right_camera")


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
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if "state/left_ee_poses" in handle:
            pose_paths = (
                "state/left_ee_poses",
                "state/left_ee_joint_states",
                "state/right_ee_poses",
                "state/right_ee_joint_states",
            )
            camera_paths = tuple(f"vision/{camera}/colors" for camera in CAMERAS)
        elif "endpose/left_endpose" in handle:
            pose_paths = (
                "endpose/left_endpose",
                "endpose/left_gripper",
                "endpose/right_endpose",
                "endpose/right_gripper",
            )
            camera_paths = tuple(f"observation/{camera}/rgb" for camera in TARGET_CAMERAS)
        else:
            raise ValueError(
                f"unsupported RoboTwin HDF5 schema in {path}: expected state/ or endpose/"
            )

        length = len(handle[pose_paths[0]])
        indices = np.unique(np.linspace(0, length - 1, min(num_frames, length), dtype=np.int64))
        images = np.stack(
            [
                np.stack(
                    [decode_jpeg(handle[camera_path][index]) for camera_path in camera_paths]
                )
                for index in indices
            ]
        )
        proprio = pose_gripper_to_ee6d(
            handle[pose_paths[0]][indices],
            handle[pose_paths[1]][indices],
            handle[pose_paths[2]][indices],
            handle[pose_paths[3]][indices],
        )
        if "instruction" in handle:
            instruction = handle["instruction"][()]
            if isinstance(instruction, bytes):
                instruction = instruction.decode()
        else:
            instruction_path = (
                path.parent.parent / "instruction" / f"{path.stem}.json"
            )
            if not instruction_path.is_file():
                raise ValueError(
                    f"no instruction dataset or sidecar found for {path}"
                )
            payload = json.loads(instruction_path.read_text())
            candidates = payload.get("seen", [])
            if isinstance(candidates, str):
                instruction = candidates
            elif candidates:
                instruction = candidates[0]
            else:
                instruction = payload.get("instruction")
            if not isinstance(instruction, str) or not instruction:
                raise ValueError(f"instruction sidecar has no usable text: {instruction_path}")
    progress = indices.astype(np.float32) / max(1, length - 1)
    return {
        "images": images,
        "proprio": proprio,
        "indices": indices,
        "progress": progress,
        "instruction": instruction,
    }
