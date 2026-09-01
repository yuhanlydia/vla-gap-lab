#!/usr/bin/env python3
"""Load X-VLA and capture real action-transformer layer states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from vla_gap_lab.xvla_adapter import capture_xvla_action_layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 4, 8, 12, 16, 20, 23])
    args = parser.parse_args()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    images = [np.full((480, 640, 3), value, dtype=np.uint8) for value in (0, 64, 128)]
    inputs = processor(images=images, language_instruction="pick up the block")
    inputs = {key: value.cuda() for key, value in inputs.items()}
    inputs["image_input"] = inputs["image_input"].to(torch.bfloat16)
    proprio = torch.zeros(1, 20, device="cuda", dtype=torch.bfloat16)
    action, features = capture_xvla_action_layers(
        model, inputs, domain_id=6, proprio=proprio, layers=args.layers
    )
    report = {
        "model_class": type(model).__name__,
        "parameters_b": sum(parameter.numel() for parameter in model.parameters()) / 1e9,
        "layers": args.layers,
        "action_shape": list(action.shape),
        "feature_shape": list(features.shape),
        "finite": bool(torch.isfinite(action).all() and torch.isfinite(features).all()),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
