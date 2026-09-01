#!/usr/bin/env python3
"""Collect mu-VLA memory states with ShellGame ground-truth tracking labels."""

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


def scalar_int(value) -> int:
    if torch.is_tensor(value):
        return int(value.detach().reshape(-1)[0].cpu())
    return int(np.asarray(value).reshape(-1)[0])


def current_target_slot(env, elapsed: int) -> tuple[int, int, str]:
    """Return target slot, completed swaps, and phase at the current observation."""
    base = env.unwrapped
    cue = scalar_int(base.cue_steps_per_env)
    shuffle = scalar_int(base.shuffle_steps_per_env)
    swaps = scalar_int(base.num_swaps_per_env)
    steps_per_swap = max(1, scalar_int(base.steps_per_swap_per_env))
    if elapsed < cue:
        completed, phase = 0, "cue"
    elif elapsed < cue + shuffle:
        completed = min(swaps, max(0, (elapsed - cue) // steps_per_swap))
        phase = "shuffle"
    else:
        completed, phase = swaps, "manipulation"
    target_mug = scalar_int(base.cup_with_ball_number)
    slot = scalar_int(base.slot_of_mug[0, completed, target_mug])
    return slot, completed, phase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="ShellGameShuffleTouch-VLA-v0")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument("--pooling", choices=["mean", "summary", "full"], default="summary")
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
    policy = MuVLAPolicy(args.checkpoint, env.unwrapped.LANGUAGE_INSTRUCTION)
    rows = {key: [] for key in ("memory", "episode", "step", "phase", "target_mug", "target_slot", "completed_swaps")}
    episode_summaries = []
    try:
        for episode_index in range(args.episodes):
            seed = args.start_seed + episode_index
            obs, _ = env.reset(seed=seed)
            policy.reset()
            success = False
            for step in range(int(env.max_episode_steps)):
                elapsed = scalar_int(env.unwrapped.elapsed_steps)
                slot, completed, phase = current_target_slot(env, elapsed)
                tokens = policy.memory.float()[0]
                if args.pooling == "mean":
                    memory_features = tokens.mean(dim=0)
                elif args.pooling == "summary":
                    memory_features = torch.stack(
                        [tokens.mean(dim=0), tokens.std(dim=0), tokens[0], tokens[-1]]
                    )
                else:
                    memory_features = tokens
                rows["memory"].append(memory_features.cpu().numpy())
                rows["episode"].append(episode_index)
                rows["step"].append(elapsed)
                rows["phase"].append(phase)
                rows["target_mug"].append(scalar_int(env.unwrapped.cup_with_ball_number))
                rows["target_slot"].append(slot)
                rows["completed_swaps"].append(completed)
                action = policy.forward(obs).to(env.unwrapped.device)
                obs, _, terminated, truncated, info = env.step(action)
                success = success or bool(scalar_int(info.get("success", False)))
                if bool(scalar_int(terminated)) or bool(scalar_int(truncated)):
                    break
            episode_summaries.append({"episode": episode_index, "seed": seed, "success": success})
    finally:
        env.close()

    metadata = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "start_seed": args.start_seed,
        "episodes": episode_summaries,
        "memory_pooling": args.pooling,
        "memory_timing": "before current observation update",
        "target_slot": "simulator slot of target mug after completed swaps at current observation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        memory=np.asarray(rows["memory"], dtype=np.float16),
        episode=np.asarray(rows["episode"], dtype=np.int32),
        step=np.asarray(rows["step"], dtype=np.int32),
        phase=np.asarray(rows["phase"]),
        target_mug=np.asarray(rows["target_mug"], dtype=np.int8),
        target_slot=np.asarray(rows["target_slot"], dtype=np.int8),
        completed_swaps=np.asarray(rows["completed_swaps"], dtype=np.int8),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "samples": len(rows["step"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
