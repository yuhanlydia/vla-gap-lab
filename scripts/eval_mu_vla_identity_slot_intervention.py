#!/usr/bin/env python3
"""Closed-loop mu-VLA evaluation for identity-preserving slot interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_gap_lab.artifact_io import write_json_atomic

EDITOR_MODES = {"ipsi", "slot_only", "random_orthogonal"}


def scalar(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(np.asarray(value).reshape(-1)[0])


def scalar_int(value) -> int:
    return int(scalar(value))


def hidden_target_state(env, elapsed: int) -> tuple[int, int, int]:
    """Return hidden target identity, current slot, and completed swaps."""
    base = env.unwrapped
    if not hasattr(base, "cup_with_ball_number"):
        raise ValueError(
            "identity-preserving interventions require hidden cup_with_ball_number"
        )
    cue = scalar_int(base.cue_steps_per_env)
    shuffle = scalar_int(base.shuffle_steps_per_env)
    swaps = scalar_int(base.num_swaps_per_env)
    steps_per_swap = max(1, scalar_int(base.steps_per_swap_per_env))
    if elapsed < cue:
        completed = 0
    elif elapsed < cue + shuffle:
        completed = min(swaps, max(0, (elapsed - cue) // steps_per_swap))
    else:
        completed = swaps
    identity = scalar_int(base.cup_with_ball_number)
    slot = scalar_int(base.slot_of_mug[0, completed, identity])
    return identity, slot, completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--mode",
        choices=["normal", "ipsi", "slot_only", "random_orthogonal"],
        default="normal",
    )
    parser.add_argument("--editor", type=Path, default=None)
    parser.add_argument("--edit-seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=4242424242)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.start_seed < 0 or args.start_seed + args.episodes - 1 >= 2**32:
        raise ValueError("episode seeds must be in NumPy RandomState range [0, 2**32 - 1]")
    if args.mode in EDITOR_MODES and args.editor is None:
        raise ValueError(f"--editor is required for mode {args.mode}")

    expected = {
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "start_seed": args.start_seed,
    }
    if args.mode in EDITOR_MODES:
        expected.update({"editor": str(args.editor), "edit_seed": args.edit_seed})
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
        print(
            json.dumps(
                {**expected, "completed_episodes": len(episodes), "resumed": True}, indent=2
            )
        )
        return

    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401
    import torch
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    from vla_gap_lab.identity_slot_editor import IdentitySlotEditor
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
        mode="normal",
    )
    editor = IdentitySlotEditor.from_npz(str(args.editor)) if args.mode in EDITOR_MODES else None

    def checkpoint() -> dict:
        all_edits = [edit for episode in episodes for edit in episode.get("edits", [])]
        editor_summary = None
        if all_edits:
            editor_summary = {
                "events": len(all_edits),
                "applied_edits": sum(not edit.get("skipped", False) for edit in all_edits),
                "slot_accuracy_before": float(
                    np.mean([edit["predicted_slot"] == edit["target_slot"] for edit in all_edits])
                ),
                "slot_accuracy_after": float(
                    np.mean(
                        [edit["predicted_slot_after"] == edit["target_slot"] for edit in all_edits]
                    )
                ),
                "identity_accuracy_before": float(
                    np.mean(
                        [
                            edit["predicted_identity"] == edit["target_identity"]
                            for edit in all_edits
                        ]
                    )
                ),
                "identity_accuracy_after": float(
                    np.mean(
                        [
                            edit["predicted_identity_after"] == edit["target_identity"]
                            for edit in all_edits
                        ]
                    )
                ),
                "mean_identity_logit_shift_l2": float(
                    np.mean([edit["identity_logit_shift_l2"] for edit in all_edits])
                ),
            }
        report = {
            "schema_version": 1,
            **expected,
            "episodes": episodes,
            "success_rate": float(np.mean([episode["success"] for episode in episodes])),
            "editor_summary": editor_summary,
        }
        write_json_atomic(args.output, report)
        return report

    try:
        for episode_index in range(len(episodes), args.episodes):
            obs, _ = env.reset(seed=args.start_seed + episode_index)
            policy.reset()
            success, total_reward = False, 0.0
            last_completed_swap = 0
            episode_edits = []
            for step in range(int(env.max_episode_steps)):
                if editor is not None:
                    elapsed = scalar_int(env.unwrapped.elapsed_steps)
                    target_identity, target_slot, completed = hidden_target_state(env, elapsed)
                    if completed > last_completed_swap:
                        generator = torch.Generator(device=policy.memory.device).manual_seed(
                            args.edit_seed + 10_000 * episode_index + completed
                        )
                        policy.memory, edit = editor.edit(
                            policy.memory,
                            target_slot=target_slot,
                            mode=args.mode,
                            generator=generator,
                        )
                        edit.update(
                            {
                                "step": elapsed,
                                "completed_swap": completed,
                                "target_identity": target_identity,
                            }
                        )
                        episode_edits.append(edit)
                        last_completed_swap = completed
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
                    "edits": episode_edits,
                }
            )
            checkpoint()
    finally:
        env.close()
    report = checkpoint()
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
