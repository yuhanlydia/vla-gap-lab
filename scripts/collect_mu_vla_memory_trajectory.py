#!/usr/bin/env python3
"""Collect mu-VLA memory states with ShellGame ground-truth tracking labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_gap_lab.trajectory_io import load_trajectory_for_resume, save_trajectory_atomic


def scalar_int(value) -> int:
    if hasattr(value, "detach"):
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
    # ShellGameTouch exposes the hidden object as ``cup_with_ball_number``.
    # The color-lamp tracking task instead hides three colored balls and
    # reveals the target color only during manipulation; its equivalent
    # persistent identity is ``target_color``.  Keep the label semantics
    # explicit so a privileged slot intervention can preserve identity while
    # changing only the current slot.
    if hasattr(base, "cup_with_ball_number"):
        target_mug = scalar_int(base.cup_with_ball_number)
    elif hasattr(base, "target_color"):
        target_mug = scalar_int(base.target_color)
    else:
        raise AttributeError(
            "environment must expose cup_with_ball_number or target_color for tracking labels"
        )
    slot = scalar_int(base.slot_of_mug[0, completed, target_mug])
    return slot, completed, phase


def current_target_identity(env) -> int:
    """Return the persistent hidden identity used by the task."""
    base = env.unwrapped
    if hasattr(base, "cup_with_ball_number"):
        return scalar_int(base.cup_with_ball_number)
    if hasattr(base, "target_color"):
        return scalar_int(base.target_color)
    raise AttributeError(
        "environment must expose cup_with_ball_number or target_color for tracking labels"
    )


def target_identity_semantics(env) -> str:
    """Describe whether the task's target identity is hidden during the cue."""
    base = env.unwrapped
    if hasattr(base, "cup_with_ball_number"):
        return "hidden_target_mug"
    if hasattr(base, "target_color"):
        # Color-lamp tasks reveal this label at manipulation; it is not a
        # hidden identity that can be causally preserved through shuffling.
        return "lamp_target_color_revealed_at_manipulation"
    raise AttributeError(
        "environment must expose cup_with_ball_number or target_color for tracking labels"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="ShellGameShuffleTouch-VLA-v0")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument(
        "--pooling", choices=["mean", "summary", "strided", "full"], default="summary"
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")

    rows = {
        key: []
        for key in (
            "memory",
            "episode",
            "step",
            "phase",
            "target_mug",
            "target_slot",
            "completed_swaps",
        )
    }
    expected = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "start_seed": args.start_seed,
        "memory_pooling": args.pooling,
    }
    episode_summaries = []
    if args.resume and args.output.exists():
        rows, prior_metadata = load_trajectory_for_resume(args.output, expected)
        episode_summaries = prior_metadata["episodes"]
    start_episode = len(episode_summaries)
    if start_episode > args.episodes:
        raise ValueError(f"resume file has {start_episode} episodes, above target {args.episodes}")
    if start_episode == args.episodes:
        print(
            json.dumps(
                {
                    **expected,
                    "completed_episodes": len(episode_summaries),
                    "samples": len(rows["step"]),
                    "resumed": True,
                },
                indent=2,
            )
        )
        return

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
    policy = MuVLAPolicy(args.checkpoint, env.unwrapped.LANGUAGE_INSTRUCTION)

    def checkpoint() -> None:
        metadata = {
            "schema_version": 2,
            **expected,
            "episodes": episode_summaries,
            "memory_timing": "before current observation update",
            "target_slot": "simulator slot of target mug after completed swaps at current observation",
            "target_identity_semantics": target_identity_semantics(env),
        }
        save_trajectory_atomic(args.output, rows, metadata)

    try:
        for episode_index in range(start_episode, args.episodes):
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
                elif args.pooling == "strided":
                    memory_features = tokens[::8]
                else:
                    memory_features = tokens
                rows["memory"].append(memory_features.cpu().numpy())
                rows["episode"].append(episode_index)
                rows["step"].append(elapsed)
                rows["phase"].append(phase)
                rows["target_mug"].append(current_target_identity(env))
                rows["target_slot"].append(slot)
                rows["completed_swaps"].append(completed)
                action = policy.forward(obs).to(env.unwrapped.device)
                obs, _, terminated, truncated, info = env.step(action)
                success = success or bool(scalar_int(info.get("success", False)))
                if bool(scalar_int(terminated)) or bool(scalar_int(truncated)):
                    break
            episode_summaries.append({"episode": episode_index, "seed": seed, "success": success})
            if len(episode_summaries) % args.checkpoint_every == 0:
                checkpoint()
    finally:
        env.close()
    checkpoint()
    metadata = {"schema_version": 2, **expected, "episodes": episode_summaries}
    print(json.dumps({**metadata, "samples": len(rows["step"]), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
