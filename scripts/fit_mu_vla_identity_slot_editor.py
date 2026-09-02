#!/usr/bin/env python3
"""Fit a leakage-safe identity-preserving slot editor from mu-VLA trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score

from vla_gap_lab.identity_slot_editor import identity_preserving_slot_direction


def parse_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("grid values must be positive comma-separated numbers")
    return values


def classifier_rows(model: RidgeClassifier) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coef = np.asarray(model.coef_, dtype=np.float32)
    intercept = np.atleast_1d(np.asarray(model.intercept_, dtype=np.float32))
    classes = np.asarray(model.classes_)
    if coef.shape[0] == 1 and len(classes) == 2:
        coef = np.stack([-coef[0], coef[0]])
        intercept = np.asarray([-intercept[0], intercept[0]], dtype=np.float32)
    if coef.shape[0] != len(classes):
        raise ValueError("ridge coefficients do not align with class labels")
    return coef, intercept, classes


def select_ridge(
    train_z: np.ndarray,
    train_y: np.ndarray,
    dev_z: np.ndarray,
    dev_y: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[RidgeClassifier, float, float]:
    best = None
    for alpha in alphas:
        model = RidgeClassifier(alpha=alpha, class_weight="balanced")
        model.fit(train_z, train_y)
        score = float(balanced_accuracy_score(dev_y, model.predict(dev_z)))
        candidate = (score, -alpha, model, alpha)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return best[2], float(best[3]), float(best[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--train-episodes", type=int, default=80)
    parser.add_argument("--dev-episodes", type=int, default=20)
    parser.add_argument("--alpha-grid", type=parse_grid, default=parse_grid("1,10,100,1000"))
    parser.add_argument("--scale-grid", type=parse_grid, default=parse_grid("0.25,0.5,1,2"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.trajectory, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    if metadata.get("memory_pooling") != "strided":
        raise ValueError(
            "identity-slot editor requires a trajectory collected with --pooling strided"
        )
    if metadata.get("target_identity_semantics") != "hidden_target_mug":
        raise ValueError("editor training requires a genuinely hidden persistent target identity")
    memory = data["memory"].astype(np.float32)
    if memory.ndim != 3:
        raise ValueError("strided trajectory memory must have shape (samples, tokens, hidden_dim)")
    episode = data["episode"].astype(np.int64)
    step = data["step"].astype(np.int64)
    identity_y = data["target_mug"].astype(np.int64)
    slot_y = data["target_slot"].astype(np.int64)
    swaps = data["completed_swaps"].astype(np.int64)
    groups = np.unique(episode)
    required = args.train_episodes + args.dev_episodes
    if len(groups) < required:
        raise ValueError(f"need at least {required} episodes, found {len(groups)}")
    rng = np.random.default_rng(args.seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    train_groups = shuffled[: args.train_episodes]
    dev_groups = shuffled[args.train_episodes : required]
    train_mask = np.isin(episode, train_groups)
    dev_mask = np.isin(episode, dev_groups)

    flat = memory.reshape(len(memory), -1)
    n_components = min(args.pca_dim, int(train_mask.sum()) - 1, flat.shape[1])
    if n_components < 2:
        raise ValueError("not enough training rows for PCA")
    pca = PCA(n_components=n_components, whiten=False, random_state=args.seed)
    z = np.empty((len(flat), n_components), dtype=np.float32)
    z[train_mask] = pca.fit_transform(flat[train_mask]).astype(np.float32)
    z[~train_mask] = pca.transform(flat[~train_mask]).astype(np.float32)

    identity_model, identity_alpha, identity_dev = select_ridge(
        z[train_mask], identity_y[train_mask], z[dev_mask], identity_y[dev_mask], args.alpha_grid
    )
    slot_model, slot_alpha, slot_dev = select_ridge(
        z[train_mask], slot_y[train_mask], z[dev_mask], slot_y[dev_mask], args.alpha_grid
    )
    identity_w, identity_b, identity_classes = classifier_rows(identity_model)
    slot_w, slot_b, slot_classes = classifier_rows(slot_model)

    natural_norms = []
    for group in train_groups:
        idx = np.flatnonzero(episode == group)
        idx = idx[np.argsort(step[idx])]
        if len(idx) > 1:
            natural_norms.extend(np.linalg.norm(np.diff(z[idx], axis=0), axis=1).tolist())
    median_norm = float(np.median(natural_norms)) if natural_norms else 1.0
    eval_mask = dev_mask & (swaps >= 1)
    if not np.any(eval_mask):
        raise ValueError("dev split contains no post-swap rows")

    identity_w_t = torch.from_numpy(identity_w)
    slot_w_t = torch.from_numpy(slot_w)
    slot_class_to_index = {int(value): index for index, value in enumerate(slot_classes)}
    scale_rows = []
    for multiplier in args.scale_grid:
        norm = multiplier * median_norm
        edited_z = z[eval_mask].copy()
        before_slot = slot_model.predict(z[eval_mask])
        before_identity = identity_model.predict(z[eval_mask])
        for row_index, (row, predicted, target) in enumerate(
            zip(edited_z, before_slot, slot_y[eval_mask])
        ):
            if int(predicted) == int(target):
                continue
            try:
                direction = identity_preserving_slot_direction(
                    identity_w_t,
                    slot_w_t,
                    target_index=slot_class_to_index[int(target)],
                    predicted_index=slot_class_to_index[int(predicted)],
                ).numpy()
            except ValueError:
                continue
            edited_z[row_index] = row + norm * direction
        after_slot = slot_model.predict(edited_z)
        after_identity = identity_model.predict(edited_z)
        slot_before = float(balanced_accuracy_score(slot_y[eval_mask], before_slot))
        slot_after = float(balanced_accuracy_score(slot_y[eval_mask], after_slot))
        identity_before = float(balanced_accuracy_score(identity_y[eval_mask], before_identity))
        identity_after = float(balanced_accuracy_score(identity_y[eval_mask], after_identity))
        score = (slot_after - slot_before) - 2.0 * max(0.0, identity_before - identity_after)
        scale_rows.append(
            {
                "multiplier": multiplier,
                "edit_norm": norm,
                "score": score,
                "slot_before": slot_before,
                "slot_after": slot_after,
                "identity_before": identity_before,
                "identity_after": identity_after,
            }
        )
    chosen = max(scale_rows, key=lambda row: (row["score"], -row["multiplier"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    token_indices = np.arange(memory.shape[1], dtype=np.int64) * 8
    np.savez_compressed(
        args.output,
        token_indices=token_indices,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        identity_weights=identity_w.astype(np.float32),
        identity_bias=identity_b.astype(np.float32),
        identity_classes=identity_classes,
        slot_weights=slot_w.astype(np.float32),
        slot_bias=slot_b.astype(np.float32),
        slot_classes=slot_classes,
        edit_norm=np.asarray(chosen["edit_norm"], dtype=np.float32),
    )
    report = {
        "schema_version": 1,
        "trajectory": str(args.trajectory),
        "output": str(args.output),
        "pca_dim": n_components,
        "train_episode_ids": [int(value) for value in train_groups],
        "dev_episode_ids": [int(value) for value in dev_groups],
        "identity_alpha": identity_alpha,
        "identity_dev_balanced_accuracy": identity_dev,
        "slot_alpha": slot_alpha,
        "slot_dev_balanced_accuracy": slot_dev,
        "median_natural_latent_update_norm": median_norm,
        "scale_sweep": scale_rows,
        "chosen_scale": chosen,
        "target_identity_semantics": metadata.get("target_identity_semantics"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
