#!/usr/bin/env python3
"""Closed-loop single-task mu-VLA evaluation with memory interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import mikasa_robo_suite.vla.memory_envs  # noqa: F401
import numpy as np
import torch
from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

from vla_gap_lab.mu_vla_adapter import MuVLAPolicy


def scalar(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(np.asarray(value).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=["normal", "freeze", "oracle_refresh"], default="normal")
    parser.add_argument("--revision-step", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    policy = MuVLAPolicy(
        args.checkpoint,
        env.unwrapped.LANGUAGE_INSTRUCTION,
        mode=args.mode,
        revision_step=args.revision_step,
    )
    episodes = []
    try:
        for episode_index in range(args.episodes):
            obs, _ = env.reset(seed=args.start_seed + episode_index)
            policy.reset()
            success, total_reward = False, 0.0
            for step in range(int(env.max_episode_steps)):
                action = policy.forward(obs).to(env.unwrapped.device)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += scalar(reward)
                success = success or bool(scalar(info.get("success", False)))
                if bool(scalar(terminated)) or bool(scalar(truncated)):
                    break
            episodes.append(
                {
                    "seed": args.start_seed + episode_index,
                    "success": success,
                    "steps": step + 1,
                    "return": total_reward,
                    "inertia_mean": float(np.mean(policy.inertia)),
                    "inertia_by_step": policy.inertia,
                }
            )
    finally:
        env.close()
    report = {
        "task": args.task,
        "mode": args.mode,
        "revision_step": args.revision_step,
        "episodes": episodes,
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
