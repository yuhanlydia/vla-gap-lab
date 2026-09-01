#!/usr/bin/env python3
"""Record checkpoint size and current-GPU load feasibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-mode", choices=["bf16", "int8", "int4"], default="bf16")
    parser.add_argument("--reserve-gib", type=float, default=2.0)
    args = parser.parse_args()

    info = HfApi().model_info(args.repo, files_metadata=True)
    checkpoint_bytes = sum(
        sibling.size or 0
        for sibling in info.siblings
        if sibling.rfilename.endswith((".safetensors", ".bin", ".pt"))
    )
    factor = {"bf16": 1.0, "int8": 0.5, "int4": 0.25}[args.load_mode]
    estimated_weight_gib = checkpoint_bytes / 2**30 * factor
    gpu = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "free_gib": free / 2**30,
            "total_gib": total / 2**30,
        }
    feasible = bool(gpu and estimated_weight_gib + args.reserve_gib <= gpu["free_gib"])
    report = {
        "repo": info.id,
        "revision": info.sha,
        "load_mode": args.load_mode,
        "checkpoint_gib": checkpoint_bytes / 2**30,
        "estimated_weight_gib": estimated_weight_gib,
        "reserve_gib": args.reserve_gib,
        "gpu": gpu,
        "feasible_by_weight_estimate": feasible,
        "note": "Estimate excludes transient activations and simulator memory.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

