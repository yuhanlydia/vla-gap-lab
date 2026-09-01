#!/usr/bin/env python3
"""Extract X-VLA hidden states from paired RoboTwin reset renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from vla_gap_lab.xvla_adapter import capture_xvla_joint_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--instruction", default="rank the red, green, and blue blocks in RGB order"
    )
    parser.add_argument("--domain-id", type=int, default=6)
    parser.add_argument("--pooling", choices=["mean", "summary"], default="summary")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = np.load(args.pairs, allow_pickle=False)
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    vlm_layers, action_layers = [0, 3, 6, 9, 11], [0, 4, 8, 12, 16, 20, 23]
    rows = []
    with torch.inference_mode():
        for index, (images, proprio) in enumerate(zip(source["images"], source["proprio"])):
            inputs = processor(images=list(images), language_instruction=args.instruction)
            inputs = {key: value.cuda() for key, value in inputs.items()}
            inputs["image_input"] = inputs["image_input"].to(torch.bfloat16)
            proprio_tensor = torch.from_numpy(proprio[None]).cuda().to(torch.bfloat16)
            torch.manual_seed(1000 + index)
            action, vlm, control = capture_xvla_joint_layers(
                model,
                inputs,
                domain_id=args.domain_id,
                proprio=proprio_tensor,
                vlm_layers=vlm_layers,
                action_layers=action_layers,
                pooling=args.pooling,
            )
            rows.append(
                (
                    vlm.float().cpu().numpy()[0],
                    control.float().cpu().numpy()[0],
                    action.float().cpu().numpy()[0],
                )
            )
    metadata = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "pairs": str(args.pairs),
        "domain_id": args.domain_id,
        "vlm_layers": vlm_layers,
        "action_layers": action_layers,
        "instruction": args.instruction,
        "pooling": args.pooling,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        vlm=np.stack([row[0] for row in rows]),
        control=np.stack([row[1] for row in rows]),
        actions=np.stack([row[2] for row in rows]),
        seed=source["seed"],
        embodiment=source["embodiment"],
        actor_xyz=source["actor_xyz"],
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
