#!/usr/bin/env python3
"""Run one real MIKASA observation through the released mu-VLA policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="RememberColor3-VLA-v0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401
    import torch
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    from vla_gap_lab.mu_vla_adapter import MuVLAPolicy

    env = gym.make(
        args.task,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        reward_mode="normalized_dense",
        render_mode="all",
        sim_backend="gpu",
    )
    env = apply_mikasa_vla_wrappers(env, include_overlays=False)
    try:
        obs, _ = env.reset(seed=4242424242)
        policy = MuVLAPolicy(args.checkpoint, env.unwrapped.LANGUAGE_INSTRUCTION)
        torch.cuda.reset_peak_memory_stats()
        action = policy.forward(obs)
        report = {
            "task": args.task,
            "action_shape": list(action.shape),
            "finite": bool(torch.isfinite(action).all()),
            "memory_shape": list(policy.memory.shape),
            "memory_inertia": policy.inertia[-1],
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
