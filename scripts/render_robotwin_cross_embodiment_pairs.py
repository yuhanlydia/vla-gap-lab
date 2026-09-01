#!/usr/bin/env python3
"""Render same-task, same-seed RoboTwin states across embodiments.

This is a reset/render diagnostic and deliberately sets ``need_plan=False``.
Apply ``patches/robotwin-reset-without-curobo.patch`` to the RoboTwin submodule
when Curobo cannot be compiled (for example, when nvcc is unavailable).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from vla_gap_lab.robotwin_data import pose_gripper_to_ee6d

CAMERA_MAP = ("head_camera", "left_camera", "right_camera")


def _load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def _render(task_name: str, embodiment: str, seed: int, root: Path) -> dict:
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module(f"envs.{task_name}")
    task_cls = getattr(module, task_name)
    configs = Path("env_cfg/task_config")
    config = _load_yaml(configs / "demo_clean.yml")
    embodiments = _load_yaml(configs / "_embodiment_config.yml")
    robot_path = Path(embodiments[embodiment]["file_path"])
    robot_config = _load_yaml(robot_path / "config.yml")
    config.update(
        task_name=task_name,
        left_robot_file=str(robot_path),
        right_robot_file=str(robot_path),
        left_embodiment_config=robot_config,
        right_embodiment_config=robot_config,
        dual_arm_embodied=True,
        embodiment_name=embodiment,
        task_config="demo_clean",
        need_plan=False,
        save_data=False,
    )
    task = task_cls()
    # Robot construction can consume an embodiment-dependent number of random
    # draws. Re-seed exactly at the task actor boundary so paired scenes share
    # object geometry and poses, while retaining the official task sampler.
    original_load_actors = task.load_actors

    def paired_load_actors() -> None:
        np.random.seed(seed)
        original_load_actors()

    task.load_actors = paired_load_actors
    try:
        task.setup_demo(now_ep_num=seed, seed=seed, **config)
        obs = task.get_obs()
        images = np.stack([obs["observation"][name]["rgb"] for name in CAMERA_MAP])
        endpose = obs["endpose"]
        proprio = pose_gripper_to_ee6d(
            np.asarray(endpose["left_endpose"], dtype=np.float32)[None],
            np.asarray([endpose["left_gripper"]], dtype=np.float32),
            np.asarray(endpose["right_endpose"], dtype=np.float32)[None],
            np.asarray([endpose["right_gripper"]], dtype=np.float32),
        )[0]
        actor_xyz = np.stack(
            [getattr(task, f"block{index}").get_pose().p for index in (1, 2, 3)]
        ).astype(np.float32)
        return {"images": images, "proprio": proprio, "actor_xyz": actor_xyz}
    finally:
        task.close_env(clear_cache=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", default="blocks_ranking_rgb")
    parser.add_argument("--embodiments", nargs="+", default=["aloha-agilex", "franka-panda"])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.robotwin_root.resolve()
    output = args.output.resolve()
    rows = []
    for seed in range(args.seeds):
        for embodiment in args.embodiments:
            rows.append((seed, embodiment, _render(args.task, embodiment, seed, root)))
    # Identical task seeds must produce the same non-robot object state.
    for seed in range(args.seeds):
        states = [row[2]["actor_xyz"] for row in rows if row[0] == seed]
        if not all(np.allclose(states[0], state, atol=1e-6) for state in states[1:]):
            raise RuntimeError(f"seed {seed} did not preserve object state across embodiments")
    metadata = {
        "schema_version": 1,
        "task": args.task,
        "embodiments": args.embodiments,
        "seeds": args.seeds,
        "state_label": "same-seed reset state; not a semantic trajectory phase",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        images=np.stack([row[2]["images"] for row in rows]),
        proprio=np.stack([row[2]["proprio"] for row in rows]),
        actor_xyz=np.stack([row[2]["actor_xyz"] for row in rows]),
        seed=np.asarray([row[0] for row in rows]),
        embodiment=np.asarray([row[1] for row in rows]),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
