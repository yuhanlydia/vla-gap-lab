#!/usr/bin/env python3
"""Closed-loop single-task mu-VLA evaluation with memory interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_gap_lab.artifact_io import write_json_atomic


def scalar(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(np.asarray(value).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=["normal", "freeze", "reset_refresh"], default="normal")
    parser.add_argument("--revision-step", type=int, default=None)
    parser.add_argument("--revision-event", choices=["cue_end", "shuffle_end"], default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "revision_step": args.revision_step,
        "revision_event": args.revision_event,
        "start_seed": args.start_seed,
    }
    episodes = []
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(
                    f"resume metadata mismatch for {key}: {previous.get(key)!r} != {value!r}"
                )
        episodes = previous["episodes"]
        seeds = [episode["seed"] for episode in episodes]
        if seeds != [args.start_seed + index for index in range(len(episodes))]:
            raise ValueError("resume episode seeds must be contiguous")
    if len(episodes) > args.episodes:
        raise ValueError(f"resume file has {len(episodes)} episodes, above target {args.episodes}")
    if len(episodes) == args.episodes:
        print(json.dumps({**expected, "completed_episodes": len(episodes), "resumed": True}, indent=2))
        return

    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401
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
    policy = MuVLAPolicy(
        args.checkpoint,
        env.unwrapped.LANGUAGE_INSTRUCTION,
        mode=args.mode,
        revision_step=args.revision_step,
    )
    def checkpoint() -> dict:
        report = {
            "schema_version": 2,
            **expected,
            "episodes": episodes,
            "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        }
        write_json_atomic(args.output, report)
        return report

    try:
        for episode_index in range(len(episodes), args.episodes):
            obs, _ = env.reset(seed=args.start_seed + episode_index)
            policy.reset()
            cue_steps = int(scalar(getattr(env.unwrapped, "cue_steps_per_env", [0])))
            shuffle_steps = int(scalar(getattr(env.unwrapped, "shuffle_steps_per_env", [0])))
            swaps = int(scalar(getattr(env.unwrapped, "num_swaps_per_env", [0])))
            if args.revision_event == "cue_end":
                policy.revision_step = cue_steps
            elif args.revision_event == "shuffle_end":
                policy.revision_step = cue_steps + shuffle_steps
            else:
                policy.revision_step = args.revision_step
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
                    "cue_steps": cue_steps,
                    "shuffle_steps": shuffle_steps,
                    "num_swaps": swaps,
                    "effective_revision_step": policy.revision_step,
                    "inertia_mean": float(np.mean(policy.inertia)),
                    "inertia_by_step": policy.inertia,
                    "candidate_inertia_by_step": policy.candidate_inertia,
                    "candidate_update_norm_by_step": policy.update_norm,
                }
            )
            checkpoint()
    finally:
        env.close()
    report = checkpoint()
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
