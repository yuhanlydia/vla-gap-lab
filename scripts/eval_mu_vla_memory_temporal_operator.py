#!/usr/bin/env python3
"""Paired causal test of velocity information injected into recurrent memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vla_gap_lab.artifact_io import write_json_atomic
from vla_gap_lab.mu_vla_protocol import step_mikasa_env
from vla_gap_lab.temporal_operator import apply_memory_velocity_operator

# Make the sibling script importable when this file is launched by absolute path
# (the normal evaluator invocation used from external/MIKASA-Robo).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.eval_mu_vla_temporal_operator import (
    _bootstrap_lower,
    _load_probe_fit,
    _scalar_bool,
    resolve_episodes_dir,
)

CONDITIONS = ("normal", "memory_velocity", "oracle_velocity", "orthogonal_sham")


def _velocity_direction(pca, velocity_model) -> np.ndarray:
    scaler = velocity_model.named_steps["standardscaler"]
    ridge = velocity_model.named_steps["ridge"]
    coefficient = ridge.coef_[1].astype(np.float32) / scaler.scale_.astype(np.float32)
    direction = pca.components_.T @ coefficient
    direction /= np.linalg.norm(direction)
    return direction.reshape(8, -1).astype(np.float32)


def _orthogonal_direction(direction: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    candidate = rng.standard_normal(direction.shape).astype(np.float32)
    candidate -= np.sum(candidate * direction) * direction
    candidate /= np.linalg.norm(candidate)
    return candidate


def _build_report(episodes, fit_metadata, args, *, status: str) -> dict:
    report = {
        "schema_version": 1,
        "status": status,
        "task": args.task,
        "checkpoint": str(args.checkpoint),
        "precision": "4bit",
        "episodes": args.episodes,
        "start_seed": args.start_seed,
        "simulator_step_sync": "render_after_step",
        "conditions": episodes,
        "success_rates": {
            condition: float(np.mean([row["success"] for row in rows]))
            if rows
            else None
            for condition, rows in episodes.items()
        },
        "fit": fit_metadata,
    }
    if status != "completed":
        return report
    normal = np.asarray([row["success"] for row in episodes["normal"]], dtype=np.float32)
    memory = np.asarray(
        [row["success"] for row in episodes["memory_velocity"]], dtype=np.float32
    )
    sham = np.asarray(
        [row["success"] for row in episodes["orthogonal_sham"]], dtype=np.float32
    )
    difference = memory - normal
    sham_difference = sham - normal
    report["primary_test"] = {
        "operator": "memory_velocity",
        "success_gain_pp": float(100 * difference.mean()),
        "paired_bootstrap_lower_pp": float(100 * _bootstrap_lower(difference, seed=0)),
        "passes": bool(
            difference.mean() >= 0.10
            and _bootstrap_lower(difference, seed=0) > 0
            and report["success_rates"]["memory_velocity"]
            > report["success_rates"]["orthogonal_sham"]
        ),
    }
    report["oracle_test"] = {
        "success_gain_pp": float(
            100
            * (
                report["success_rates"]["oracle_velocity"]
                - report["success_rates"]["normal"]
            )
        )
    }
    report["orthogonal_sham_test"] = {
        "success_gain_pp": float(100 * sham_difference.mean())
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--probe-report",
        type=Path,
        default=Path("artifacts/reports/intercept_medium_k2_predictive_dynamics.json"),
    )
    parser.add_argument("--episodes-dir", type=Path, default=None)
    parser.add_argument("--task", default="InterceptMedium-VLA-v0")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--start-seed", type=int, default=4242824242)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--pca-batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.task != "InterceptMedium-VLA-v0" or args.episodes != 40:
        raise ValueError("the frozen memory temporal-operator test requires 40 InterceptMedium episodes")
    if args.start_seed < 0 or args.start_seed + args.episodes - 1 >= 2**32:
        raise ValueError("episode seeds must be in NumPy RandomState range [0, 2**32 - 1]")

    episodes_dir = args.episodes_dir or resolve_episodes_dir(args.probe_report)
    pca, velocity_model, fit_metadata = _load_probe_fit(
        episodes_dir,
        args.probe_report,
        pca_dim=args.pca_dim,
        pca_batch_size=args.pca_batch_size,
    )
    velocity_direction = _velocity_direction(pca, velocity_model)
    sham_direction = _orthogonal_direction(velocity_direction)
    fit_metadata = {
        **fit_metadata,
        "memory_injection_fraction": 0.25,
        "memory_injection_norm": 0.25 * fit_metadata["memory_update_norm_median_train"],
        "velocity_standardization_clip": 2.0,
        "operator": "retained_memory_velocity_direction",
    }

    import gymnasium as gym
    import mikasa_robo_suite.vla.memory_envs  # noqa: F401
    import torch
    from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

    from vla_gap_lab.mu_vla_protocol import ProtocolMatchedMuVLAPolicy

    env = gym.make(
        args.task,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        reward_mode="normalized_dense",
        render_mode="rgb_array",
        sim_backend="gpu",
    )
    env = apply_mikasa_vla_wrappers(env, include_overlays=False)
    base = env.unwrapped
    policy = ProtocolMatchedMuVLAPolicy(
        args.checkpoint,
        base.LANGUAGE_INSTRUCTION,
        load_in_4bit=True,
    )
    episodes = {condition: [] for condition in CONDITIONS}
    if args.output.exists():
        if not args.resume:
            raise ValueError(f"output exists; pass --resume to continue: {args.output}")
        previous = json.loads(args.output.read_text())
        for key, value in {
            "task": args.task,
            "checkpoint": str(args.checkpoint),
            "episodes": args.episodes,
            "start_seed": args.start_seed,
            "simulator_step_sync": "render_after_step",
        }.items():
            if previous.get(key) != value:
                raise ValueError(f"resume metadata mismatch for {key}")
        episodes = previous["conditions"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for condition in CONDITIONS:
            for episode_index in range(len(episodes[condition]), args.episodes):
                seed = args.start_seed + episode_index
                obs, _ = env.reset(seed=seed)
                policy.reset()
                success = False
                for step in range(int(env.max_episode_steps)):
                    true_velocity_y = float(base.ball.linear_velocity[0, 1].detach().cpu())
                    memory = policy.memory.float()[0][::8].detach().cpu().numpy()
                    predicted_velocity_y = float(
                        velocity_model.predict(pca.transform(memory.reshape(1, -1)))[0, 1]
                    )
                    mean = fit_metadata["velocity_y_mean_train"]
                    std = fit_metadata["velocity_y_std_train"]
                    predicted_z = float(np.clip((predicted_velocity_y - mean) / std, -2, 2))
                    true_z = float(np.clip((true_velocity_y - mean) / std, -2, 2))
                    override = None
                    if condition == "memory_velocity":
                        edited = apply_memory_velocity_operator(
                            memory=policy.memory.float().detach().cpu().numpy(),
                            direction=velocity_direction,
                            standardized_velocity=predicted_z,
                            gain=fit_metadata["memory_injection_norm"],
                        )
                        override = torch.from_numpy(edited).to(
                            base.device, dtype=policy.memory.dtype
                        )
                    elif condition == "oracle_velocity":
                        edited = apply_memory_velocity_operator(
                            memory=policy.memory.float().detach().cpu().numpy(),
                            direction=velocity_direction,
                            standardized_velocity=true_z,
                            gain=fit_metadata["memory_injection_norm"],
                        )
                        override = torch.from_numpy(edited).to(
                            base.device, dtype=policy.memory.dtype
                        )
                    elif condition == "orthogonal_sham":
                        edited = apply_memory_velocity_operator(
                            memory=policy.memory.float().detach().cpu().numpy(),
                            direction=sham_direction,
                            standardized_velocity=predicted_z,
                            gain=fit_metadata["memory_injection_norm"],
                        )
                        override = torch.from_numpy(edited).to(
                            base.device, dtype=policy.memory.dtype
                        )
                    action = policy.forward(obs, memory_override=override).to(base.device)
                    obs, _, terminated, truncated, info = step_mikasa_env(env, action)
                    success = success or _scalar_bool(info["success"])
                    if _scalar_bool(terminated) or _scalar_bool(truncated):
                        break
                episodes[condition].append(
                    {"episode": episode_index, "seed": seed, "success": success, "steps": step + 1}
                )
                write_json_atomic(
                    args.output,
                    _build_report(episodes, fit_metadata, args, status="running"),
                )
                print(json.dumps({"condition": condition, "episode": episode_index, "success": success}))
    finally:
        env.close()
    report = _build_report(episodes, fit_metadata, args, status="completed")
    write_json_atomic(args.output, report)
    print(json.dumps({key: report[key] for key in ("success_rates", "primary_test", "oracle_test", "orthogonal_sham_test")}, indent=2))


if __name__ == "__main__":
    main()
