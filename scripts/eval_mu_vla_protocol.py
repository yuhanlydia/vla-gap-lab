#!/usr/bin/env python3
"""Protocol-matched single-task mu-VLA evaluation for 16/24 GB GPUs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scalar(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(np.asarray(value).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.start_seed < 0 or args.start_seed + args.episodes - 1 >= 2**32:
        raise ValueError(
            "episode seeds must be in NumPy RandomState range [0, 2**32 - 1]"
        )

    expected = {
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "start_seed": args.start_seed,
        "precision": args.precision,
        "preprocess": "official_224_center_crop_0.9",
    }
    episodes = []
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(f"resume metadata mismatch for {key}")
        episodes = previous["episodes"]
    if len(episodes) > args.episodes:
        raise ValueError("resume file contains too many episodes")
    if len(episodes) == args.episodes:
        print(
            json.dumps(
                {**expected, "completed_episodes": len(episodes), "resumed": True},
                indent=2,
            )
        )
        return

    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    from vla_gap_lab.artifact_io import write_json_atomic
    from vla_gap_lab.mu_vla_protocol import ProtocolMatchedMuVLAPolicy

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
    policy = ProtocolMatchedMuVLAPolicy(
        args.checkpoint,
        env.unwrapped.LANGUAGE_INSTRUCTION,
        load_in_4bit=args.precision == "4bit",
    )

    def checkpoint() -> dict:
        report = {
            "schema_version": 1,
            **expected,
            "episodes": episodes,
            "success_rate": float(
                np.mean([row["success"] for row in episodes])
            )
            if episodes
            else 0.0,
        }
        write_json_atomic(args.output, report)
        return report

    try:
        for episode_index in range(len(episodes), args.episodes):
            seed = args.start_seed + episode_index
            obs, _ = env.reset(seed=seed)
            policy.reset()
            success = False
            total_reward = 0.0
            for step in range(int(env.max_episode_steps)):
                action = policy.forward(obs).to(env.unwrapped.device)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += scalar(reward)
                success = success or bool(scalar(info.get("success", False)))
                if bool(scalar(terminated)) or bool(scalar(truncated)):
                    break
            episodes.append(
                {
                    "seed": seed,
                    "success": success,
                    "steps": step + 1,
                    "return": total_reward,
                    "memory_update_norm_mean": float(np.mean(policy.update_norm)),
                }
            )
            checkpoint()
    finally:
        env.close()
    report = checkpoint()
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "episodes"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
