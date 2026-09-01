#!/usr/bin/env python3
"""Extract X-VLA states from official RoboTwin XPolicyLab trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--frames-per-episode", type=int, default=6)
    parser.add_argument("--domain-id", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoModel, AutoProcessor

    from vla_gap_lab.robotwin_data import sample_episode
    from vla_gap_lab.xvla_adapter import capture_xvla_joint_layers

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    vlm_layers, action_layers = [0, 3, 6, 9, 11], [0, 4, 8, 12, 16, 20, 23]
    records = []
    episode_paths = sorted(args.data_dir.glob("episode_*.hdf5"))[: args.episodes]
    with torch.inference_mode():
        for episode_path in episode_paths:
            episode = sample_episode(episode_path, args.frames_per_episode)
            for row, frame_index in enumerate(episode["indices"]):
                inputs = processor(
                    images=list(episode["images"][row]),
                    language_instruction=episode["instruction"],
                )
                inputs = {key: value.cuda() for key, value in inputs.items()}
                inputs["image_input"] = inputs["image_input"].to(torch.bfloat16)
                proprio = torch.from_numpy(episode["proprio"][row : row + 1]).to(
                    device="cuda", dtype=torch.bfloat16
                )
                torch.manual_seed(1000 + len(records))
                action, vlm, control = capture_xvla_joint_layers(
                    model,
                    inputs,
                    domain_id=args.domain_id,
                    proprio=proprio,
                    vlm_layers=vlm_layers,
                    action_layers=action_layers,
                )
                records.append(
                    (
                        vlm.float().cpu().numpy()[0],
                        control.float().cpu().numpy()[0],
                        action.float().cpu().numpy()[0],
                        episode_path.stem,
                        int(frame_index),
                        float(episode["progress"][row]),
                    )
                )
    metadata = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "embodiment": args.data_dir.parent.name,
        "domain_id": args.domain_id,
        "vlm_layers": vlm_layers,
        "action_layers": action_layers,
        "phase_label": "normalized trajectory progress proxy; not semantic ground truth",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        vlm=np.stack([row[0] for row in records]),
        control=np.stack([row[1] for row in records]),
        actions=np.stack([row[2] for row in records]),
        episode_id=np.asarray([row[3] for row in records]),
        frame_index=np.asarray([row[4] for row in records]),
        progress=np.asarray([row[5] for row in records]),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "samples": len(records)}, indent=2))


if __name__ == "__main__":
    main()
