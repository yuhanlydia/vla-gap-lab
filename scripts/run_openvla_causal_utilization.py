#!/usr/bin/env python3
"""Measure whether a decodable action direction causally controls OpenVLA-OFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from vla_gap_lab.openvla_adapter import load_l1_action_head, predict_with_layer_addition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--action-index", type=int, nargs="+", default=[0])
    parser.add_argument("--direction-mode", choices=["probe", "random"], default="probe")
    parser.add_argument("--alpha-multipliers", type=float, nargs="+", default=[-2, -1, 1, 2])
    parser.add_argument("--max-test-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = np.load(args.pairs, allow_pickle=False)
    hidden = np.load(args.hidden, allow_pickle=False)
    metadata = json.loads(str(pairs["metadata"]))
    layers = hidden["layers"].tolist()
    if args.layer not in layers:
        raise ValueError(f"layer {args.layer} is not in cache layers {layers}")
    layer_col = layers.index(args.layer)
    groups = hidden["sample_id"]
    train_indices, test_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed).split(
            hidden["shifted"], groups=groups
        )
    )
    train_x = hidden["clean"][train_indices, layer_col]
    train_y = hidden["actions"][train_indices].reshape(len(train_indices), -1)
    probe = Ridge(alpha=1.0).fit(train_x, train_y)
    directions = {}
    projected_stds = {}
    for action_index in args.action_index:
        if action_index < 0 or action_index >= train_y.shape[1]:
            raise ValueError(f"action index {action_index} is outside [0, {train_y.shape[1] - 1}]")
        if args.direction_mode == "probe":
            direction = probe.coef_[action_index].astype(np.float32)
        else:
            direction = (
                np.random.default_rng(args.seed + action_index)
                .standard_normal(train_x.shape[1])
                .astype(np.float32)
            )
        direction /= max(np.linalg.norm(direction), 1e-8)
        directions[action_index] = direction
        projected_stds[action_index] = float(np.std(train_x @ direction))
    selected = []
    for group in sorted(set(groups[test_indices].tolist())):
        selected.extend(test_indices[groups[test_indices] == group][:1].tolist())
    remaining = [index for index in test_indices if index not in selected]
    test_indices = np.asarray((selected + remaining)[: args.max_test_samples])

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    model.norm_stats = json.loads((args.checkpoint / "dataset_statistics.json").read_text())
    action_head = load_l1_action_head(args.checkpoint, model.llm_dim)
    prompt = f"In: What action should the robot take to {metadata['instruction'].lower()}?\nOut:"
    unnorm_key = f"{metadata['suite']}_no_noops"
    records = []
    for index in test_indices:
        inputs = processor(prompt, Image.fromarray(pairs["shifted_agent"][index])).to(
            "cuda", dtype=torch.bfloat16
        )
        baseline = predict_with_layer_addition(
            model,
            inputs,
            action_head,
            layer=args.layer,
            direction=torch.from_numpy(directions[args.action_index[0]]),
            alpha=0.0,
            unnorm_key=unnorm_key,
        )
        hidden_row = hidden["shifted"][index, layer_col]
        probe_before = probe.predict(hidden_row[None])[0]
        for action_index, direction in directions.items():
            for multiplier in args.alpha_multipliers:
                alpha = multiplier * projected_stds[action_index]
                changed = predict_with_layer_addition(
                    model,
                    inputs,
                    action_head,
                    layer=args.layer,
                    direction=torch.from_numpy(direction),
                    alpha=alpha,
                    unnorm_key=unnorm_key,
                )
                probe_after = probe.predict((hidden_row + alpha * direction)[None])[0]
                probe_delta = float(np.linalg.norm(probe_after - probe_before))
                control_delta = float(np.linalg.norm(changed[0] - baseline[0]))
                records.append(
                    {
                        "index": int(index),
                        "pair_id": str(hidden["pair_ids"][index]),
                        "action_index": action_index,
                        "alpha_multiplier": multiplier,
                        "alpha": alpha,
                        "probe_delta": probe_delta,
                        "control_delta": control_delta,
                        "probe_gain": probe_delta / max(abs(alpha), 1e-12),
                        "control_gain": control_delta / max(abs(alpha), 1e-12),
                        "cur": control_delta / max(probe_delta, 1e-12),
                    }
                )
    per_action = {
        str(action_index): {
            "mean_cur": float(
                np.mean([row["cur"] for row in records if row["action_index"] == action_index])
            ),
            "mean_control_gain": float(
                np.mean(
                    [row["control_gain"] for row in records if row["action_index"] == action_index]
                )
            ),
        }
        for action_index in args.action_index
    }
    report = {
        "schema_version": 1,
        "layer": args.layer,
        "action_indices": args.action_index,
        "direction_mode": args.direction_mode,
        "projected_train_std": projected_stds,
        "train_groups": sorted(set(groups[train_indices].tolist())),
        "test_groups": sorted(set(groups[test_indices].tolist())),
        "records": records,
        "mean_cur": float(np.mean([row["cur"] for row in records])),
        "median_cur": float(np.median([row["cur"] for row in records])),
        "mean_control_gain": float(np.mean([row["control_gain"] for row in records])),
        "per_action": per_action,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
