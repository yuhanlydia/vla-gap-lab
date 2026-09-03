#!/usr/bin/env python3
"""Collect protocol-matched mu-VLA recurrent states on predictive Intercept tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scalar(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(np.asarray(value).reshape(-1)[0])


def vec2(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(-1, array.shape[-1])[0, :2]


def pool_memory(memory, pooling: str) -> np.ndarray:
    tokens = memory.float()[0]
    if pooling != "strided":
        raise ValueError("pooling must be strided")
    return tokens[::8].cpu().numpy().astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="InterceptMedium-VLA-v0")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--start-seed", type=int, default=4242624242)
    parser.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument("--pooling", choices=["strided"], default="strided")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.start_seed < 0 or args.start_seed + args.episodes - 1 >= 2**32:
        raise ValueError(
            "episode seeds must be in NumPy RandomState range [0, 2**32 - 1]"
        )

    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    from vla_gap_lab.dynamics_io import load_episode_npz, save_episode_npz_atomic
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
    base = env.unwrapped
    required = ("ball", "goal_region", "oracle_info", "agent", "reached_status")
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        raise ValueError(f"task does not expose predictive-dynamics labels: {missing}")
    policy = ProtocolMatchedMuVLAPolicy(
        args.checkpoint,
        base.LANGUAGE_INSTRUCTION,
        load_in_4bit=args.precision == "4bit",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for episode_index in range(args.episodes):
            seed = args.start_seed + episode_index
            path = args.output_dir / f"episode_{episode_index:04d}_seed_{seed}.npz"
            expected = {
                "schema_version": 1,
                "episode": episode_index,
                "seed": seed,
                "task": args.task,
                "checkpoint": str(args.checkpoint),
                "precision": args.precision,
                "pooling": args.pooling,
                "preprocess": "official_224_center_crop_0.9",
            }
            if path.exists() and args.resume:
                _, metadata = load_episode_npz(path)
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        raise ValueError(
                            f"{path}: resume metadata mismatch for {key}"
                        )
                continue

            obs, _ = env.reset(seed=seed)
            policy.reset()
            rows = {
                key: []
                for key in (
                    "memory_before",
                    "memory_after",
                    "step",
                    "ball_position_xy",
                    "ball_velocity_xy",
                    "initial_velocity_xy",
                    "goal_position_xy",
                    "tcp_position_xy",
                    "reached_status",
                    "action",
                    "reward",
                    "success",
                )
            }
            success_once = False
            for step in range(int(env.max_episode_steps)):
                rows["memory_before"].append(pool_memory(policy.memory, args.pooling))
                rows["step"].append(step)
                rows["ball_position_xy"].append(vec2(base.ball.pose.p))
                rows["ball_velocity_xy"].append(vec2(base.ball.linear_velocity))
                rows["initial_velocity_xy"].append(vec2(base.oracle_info))
                rows["goal_position_xy"].append(vec2(base.goal_region.pose.p))
                rows["tcp_position_xy"].append(vec2(base.agent.tcp.pose.p))
                rows["reached_status"].append(scalar(base.reached_status))

                action = policy.forward(obs)
                rows["memory_after"].append(pool_memory(policy.memory, args.pooling))
                rows["action"].append(action.cpu().numpy()[0])
                obs, reward, terminated, truncated, info = env.step(
                    action.to(base.device)
                )
                success = bool(scalar(info.get("success", False)))
                success_once = success_once or success
                rows["reward"].append(scalar(reward))
                rows["success"].append(success)
                if bool(scalar(terminated)) or bool(scalar(truncated)):
                    break

            arrays = {
                "memory_before": np.asarray(rows["memory_before"], dtype=np.float16),
                "memory_after": np.asarray(rows["memory_after"], dtype=np.float16),
                "step": np.asarray(rows["step"], dtype=np.int32),
                "ball_position_xy": np.asarray(
                    rows["ball_position_xy"], dtype=np.float32
                ),
                "ball_velocity_xy": np.asarray(
                    rows["ball_velocity_xy"], dtype=np.float32
                ),
                "initial_velocity_xy": np.asarray(
                    rows["initial_velocity_xy"], dtype=np.float32
                ),
                "goal_position_xy": np.asarray(
                    rows["goal_position_xy"], dtype=np.float32
                ),
                "tcp_position_xy": np.asarray(
                    rows["tcp_position_xy"], dtype=np.float32
                ),
                "reached_status": np.asarray(rows["reached_status"], dtype=np.float32),
                "action": np.asarray(rows["action"], dtype=np.float32),
                "reward": np.asarray(rows["reward"], dtype=np.float32),
                "success": np.asarray(rows["success"], dtype=np.int8),
            }
            save_episode_npz_atomic(
                path,
                arrays,
                {
                    **expected,
                    "steps": len(rows["step"]),
                    "success_once": success_once,
                },
            )
            print(
                json.dumps(
                    {
                        "episode": episode_index,
                        "seed": seed,
                        "steps": len(rows["step"]),
                        "success_once": success_once,
                        "output": str(path),
                    }
                )
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
