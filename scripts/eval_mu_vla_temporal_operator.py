#!/usr/bin/env python3
"""Paired causal test of a frozen velocity readout and action correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA

from vla_gap_lab.artifact_io import write_json_atomic
from vla_gap_lab.dynamics_io import load_episode_npz
from vla_gap_lab.dynamics_probes import select_ridge_regression
from vla_gap_lab.mu_vla_protocol import step_mikasa_env
from vla_gap_lab.temporal_operator import apply_velocity_operator, velocity_action_gain

CONDITIONS = ("normal", "memory_velocity", "oracle_velocity", "sign_sham")


def resolve_episodes_dir(probe_report_path: Path) -> Path:
    """Resolve the Gate-2 cache from a report path independent of the cwd."""
    return probe_report_path.resolve().parent.parent / "mikasa/intercept_medium_k2_dynamics_n40"


def _load_probe_fit(
    episodes_dir: Path,
    probe_report_path: Path,
    *,
    pca_dim: int,
    pca_batch_size: int,
) -> tuple[IncrementalPCA, object, dict]:
    report = json.loads(probe_report_path.read_text())
    if report["gate"]["diagnosis"] != "storage_dynamics_gap":
        raise ValueError("causal temporal operator requires a Storage-Dynamics Gap probe result")
    train_ids = np.asarray(report["train_episode_ids"], dtype=np.int64)
    dev_ids = np.asarray(report["dev_episode_ids"], dtype=np.int64)
    test_ids = np.asarray(report["test_episode_ids"], dtype=np.int64)
    if set(train_ids) & set(dev_ids) or set(train_ids) & set(test_ids) or set(dev_ids) & set(test_ids):
        raise ValueError("probe report contains overlapping episode splits")

    features, targets, groups, actions, updates = [], [], [], [], []
    for episode_id, path in enumerate(sorted(episodes_dir.glob("episode_*.npz"))):
        arrays, metadata = load_episode_npz(path)
        if metadata.get("episode") != episode_id:
            raise ValueError(f"{path}: episode metadata does not match sorted file order")
        mask = arrays["step"] >= int(report["min_step"])
        if report["pre_contact_only"]:
            mask &= arrays["reached_status"] < 0.5
        features.append(arrays["memory_after"][mask].astype(np.float32).reshape(mask.sum(), -1))
        targets.append(arrays["ball_velocity_xy"][mask].astype(np.float32))
        actions.append(arrays["action"][mask].astype(np.float32))
        updates.append(
            np.linalg.norm(
                arrays["memory_after"][mask].astype(np.float32).reshape(mask.sum(), -1)
                - arrays["memory_before"][mask].astype(np.float32).reshape(mask.sum(), -1),
                axis=1,
            )
        )
        groups.append(np.full(mask.sum(), episode_id, dtype=np.int64))
    if len(features) != 40:
        raise ValueError(f"expected 40 source episodes, found {len(features)}")
    features = np.concatenate(features)
    targets = np.concatenate(targets)
    actions = np.concatenate(actions)
    updates = np.concatenate(updates)
    groups = np.concatenate(groups)
    masks = {
        "train": np.isin(groups, train_ids),
        "dev": np.isin(groups, dev_ids),
        "test": np.isin(groups, test_ids),
    }
    components = min(pca_dim, int(masks["train"].sum()) - 1, features.shape[1])
    pca = IncrementalPCA(n_components=components, batch_size=max(components, pca_batch_size))
    pca.fit(features[masks["train"]])
    latent = {name: pca.transform(features[mask]).astype(np.float32) for name, mask in masks.items()}
    velocity_model, velocity_metrics = select_ridge_regression(
        latent["train"],
        targets[masks["train"]],
        latent["dev"],
        targets[masks["dev"]],
        latent["test"],
        targets[masks["test"]],
        alphas=(0.1, 1.0, 10.0, 100.0),
    )
    gain = velocity_action_gain(
        float(np.std(actions[masks["train"], 1])),
        float(np.std(targets[masks["train"], 1])),
    )
    return pca, velocity_model, {
        "source_report_sha256": hashlib.sha256(probe_report_path.read_bytes()).hexdigest(),
        "train_episode_ids": [int(x) for x in train_ids],
        "dev_episode_ids": [int(x) for x in dev_ids],
        "test_episode_ids": [int(x) for x in test_ids],
        "min_step": int(report["min_step"]),
        "pre_contact_only": bool(report["pre_contact_only"]),
        "pca_dim": components,
        "velocity_readout": velocity_metrics,
        "action_y_std_train": float(np.std(actions[masks["train"], 1])),
        "velocity_y_std_train": float(np.std(targets[masks["train"], 1])),
        "velocity_y_mean_train": float(np.mean(targets[masks["train"], 1])),
        "memory_update_norm_median_train": float(np.median(updates[masks["train"]])),
        "gain_fraction": 0.25,
        "gain": gain,
    }


def _bootstrap_lower(differences: np.ndarray, *, seed: int, samples: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    return float(np.quantile(means, 0.025))


def _scalar_bool(value) -> bool:
    if hasattr(value, "detach"):
        return bool(value.detach().reshape(-1)[0].cpu().item())
    return bool(np.asarray(value).reshape(-1)[0])


def _build_report(
    episodes: dict[str, list[dict]],
    fit_metadata: dict,
    args: argparse.Namespace,
    *,
    status: str,
) -> dict:
    report = {
        "schema_version": 2,
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
    memory = np.asarray([row["success"] for row in episodes["memory_velocity"]], dtype=np.float32)
    sham = np.asarray([row["success"] for row in episodes["sign_sham"]], dtype=np.float32)
    memory_difference = memory - normal
    sham_difference = sham - normal
    report.update(
        {
            "primary_test": {
                "operator": "memory_velocity",
                "success_gain_pp": float(100 * memory_difference.mean()),
                "paired_bootstrap_lower_pp": float(
                    100 * _bootstrap_lower(memory_difference, seed=0)
                ),
                "passes": bool(
                    memory_difference.mean() >= 0.10
                    and _bootstrap_lower(memory_difference, seed=0) > 0
                    and report["success_rates"]["memory_velocity"]
                    > report["success_rates"]["sign_sham"]
                ),
            },
            "oracle_test": {
                "success_gain_pp": float(
                    100
                    * (
                        report["success_rates"]["oracle_velocity"]
                        - report["success_rates"]["normal"]
                    )
                )
            },
            "sign_sham_test": {
                "success_gain_pp": float(100 * sham_difference.mean())
            },
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, default=Path("artifacts/reports/intercept_medium_k2_predictive_dynamics.json"))
    parser.add_argument("--episodes-dir", type=Path, default=None)
    parser.add_argument("--task", default="InterceptMedium-VLA-v0")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--start-seed", type=int, default=4242724242)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--pca-batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.task != "InterceptMedium-VLA-v0":
        raise ValueError("the frozen temporal-operator test is defined for InterceptMedium-VLA-v0")
    if args.episodes != 40:
        raise ValueError("the frozen temporal-operator test requires exactly 40 paired episodes")
    if args.start_seed < 0 or args.start_seed + args.episodes - 1 >= 2**32:
        raise ValueError("episode seeds must be in NumPy RandomState range [0, 2**32 - 1]")

    episodes_dir = args.episodes_dir or resolve_episodes_dir(args.probe_report)
    pca, velocity_model, fit_metadata = _load_probe_fit(
        episodes_dir,
        args.probe_report,
        pca_dim=args.pca_dim,
        pca_batch_size=args.pca_batch_size,
    )

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
        if set(previous.get("conditions", {})) != set(CONDITIONS):
            raise ValueError("resume artifact has incompatible conditions")
        episodes = previous["conditions"]
        for condition, rows in episodes.items():
            if len(rows) > args.episodes:
                raise ValueError(f"resume artifact has too many {condition} episodes")
            for index, row in enumerate(rows):
                if row.get("episode") != index or row.get("seed") != args.start_seed + index:
                    raise ValueError(f"resume artifact has non-contiguous {condition} episodes")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for condition in CONDITIONS:
            for episode_index in range(len(episodes[condition]), args.episodes):
                seed = args.start_seed + episode_index
                obs, _ = env.reset(seed=seed)
                policy.reset()
                success = False
                for step in range(int(env.max_episode_steps)):
                    action = policy.forward(obs)
                    memory = policy.memory.float()[0][::8].detach().cpu().numpy().reshape(1, -1)
                    predicted_velocity_y = float(velocity_model.predict(pca.transform(memory))[0, 1])
                    true_velocity_y = float(base.ball.linear_velocity[0, 1].detach().cpu())
                    action_np = action.detach().cpu().numpy()
                    if condition == "memory_velocity":
                        action_np = apply_velocity_operator(
                            action_np, velocity_y=predicted_velocity_y, gain=fit_metadata["gain"]
                        )
                    elif condition == "oracle_velocity":
                        action_np = apply_velocity_operator(
                            action_np, velocity_y=true_velocity_y, gain=fit_metadata["gain"]
                        )
                    elif condition == "sign_sham":
                        action_np = apply_velocity_operator(
                            action_np, velocity_y=-predicted_velocity_y, gain=fit_metadata["gain"]
                        )
                    obs, _, terminated, truncated, info = step_mikasa_env(
                        env, torch.from_numpy(action_np).to(base.device)
                    )
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
    print(json.dumps({key: report[key] for key in ("success_rates", "primary_test", "oracle_test", "sign_sham_test")}, indent=2))


if __name__ == "__main__":
    main()
