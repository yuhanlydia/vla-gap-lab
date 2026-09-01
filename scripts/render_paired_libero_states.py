#!/usr/bin/env python3
"""Render LIBERO-Plus observations at states from original demonstrations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from vla_gap_lab.libero_pairs import (
    demonstration_instruction,
    evenly_spaced_indices,
    find_demonstration,
    load_perturbation,
    require_action_preserving,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--demo-index", type=int, default=0, help="first demonstration index")
    parser.add_argument("--num-demos", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--margin", type=int, default=5)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/libero"))
    parser.add_argument("--libero-root", type=Path, default=Path("external/LIBERO-plus"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    classification = args.libero_root / "libero/libero/benchmark/task_classification.json"
    perturbation = load_perturbation(classification, args.suite, args.task_id)
    require_action_preserving(perturbation)

    # Imports are intentionally delayed so metadata/unit tests do not require MuJoCo.
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = get_benchmark(args.suite)()
    task = suite.get_task(args.task_id)
    demo_path = find_demonstration(
        args.dataset_root / args.suite, task.language, variant_name=task.name
    )
    policy_instruction = demonstration_instruction(demo_path)
    if args.num_demos < 1:
        raise ValueError("num-demos must be positive")
    states_parts, action_parts, agent_parts, wrist_parts, pair_ids = [], [], [], [], []
    with h5py.File(demo_path, "r") as handle:
        for demo_index in range(args.demo_index, args.demo_index + args.num_demos):
            key = f"demo_{demo_index}"
            if key not in handle["data"]:
                raise ValueError(f"demonstration {key} is unavailable")
            demo = handle["data"][key]
            indices = evenly_spaced_indices(len(demo["actions"]), args.num_frames, args.margin)
            states_parts.append(demo["states"][indices])
            action_parts.append(demo["actions"][indices])
            agent_parts.append(demo["obs/agentview_rgb"][indices])
            wrist_parts.append(demo["obs/eye_in_hand_rgb"][indices])
            pair_ids.extend(f"{key}:{int(index)}" for index in indices)
    states = np.concatenate(states_parts)
    actions = np.concatenate(action_parts)
    clean_agent = np.concatenate(agent_parts)
    clean_wrist = np.concatenate(wrist_parts)

    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
        camera_heights=128,
        camera_widths=128,
    )
    env.reset()
    shifted_agent, shifted_wrist = [], []
    try:
        for state in states:
            obs = env.set_init_state(state)
            shifted_agent.append(obs["agentview_image"])
            shifted_wrist.append(obs["robot0_eye_in_hand_image"])
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1,
        "suite": args.suite,
        "task_id": args.task_id,
        "task_name": task.name,
        "instruction": policy_instruction,
        "variant_instruction": task.language,
        "category": perturbation.category,
        "difficulty_level": perturbation.difficulty_level,
        "demo_path": str(demo_path),
        "demo_indices": list(range(args.demo_index, args.demo_index + args.num_demos)),
        "pair_ids": pair_ids,
        "action_semantics": "identical simulator state; original expert action",
    }
    np.savez_compressed(
        args.output,
        clean_agent=clean_agent,
        clean_wrist=clean_wrist,
        shifted_agent=np.asarray(shifted_agent),
        shifted_wrist=np.asarray(shifted_wrist),
        states=states,
        actions=actions,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    pixel_mae = float(np.abs(clean_agent.astype(float) - shifted_agent).mean())
    print(json.dumps({**metadata, "output": str(args.output), "pixel_mae": pixel_mae}, indent=2))


if __name__ == "__main__":
    main()
