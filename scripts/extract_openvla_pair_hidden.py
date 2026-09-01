#!/usr/bin/env python3
"""Extract layerwise OpenVLA action-token states from a rendered pair cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28, 32])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    from vla_gap_lab.openvla_adapter import forward_openvla_action_layers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pair_data = np.load(args.pairs, allow_pickle=False)
    pair_meta = json.loads(str(pair_data["metadata"]))
    count = len(pair_data["actions"])
    if args.max_pairs is not None:
        count = min(count, args.max_pairs)

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
    prompt = f"In: What action should the robot take to {pair_meta['instruction'].lower()}?\nOut:"

    def extract(images: np.ndarray) -> np.ndarray:
        rows = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for image in images[:count]:
                inputs = processor(prompt, Image.fromarray(image).convert("RGB")).to(
                    "cuda", dtype=torch.bfloat16
                )
                states = forward_openvla_action_layers(model, inputs, args.layers)
                rows.append(states.mean(dim=2).float().cpu().numpy()[0])
        return np.stack(rows)

    torch.cuda.reset_peak_memory_stats()
    clean = extract(pair_data["clean_agent"])
    shifted = extract(pair_data["shifted_agent"])
    metadata = {
        "schema_version": 1,
        "source_pairs": str(args.pairs),
        "pair_metadata": pair_meta,
        "checkpoint": str(args.checkpoint),
        "layers": args.layers,
        "pooling": "mean over 56 action tokens",
        "samples": count,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    pair_ids = np.asarray(pair_meta["pair_ids"][:count])
    # Group-level IDs make the downstream split hold out entire trajectories,
    # not merely nearby frames from trajectories seen during probe training.
    sample_ids = np.asarray([pair_id.split(":", 1)[0] for pair_id in pair_ids])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        clean=clean,
        shifted=shifted,
        actions=pair_data["actions"][:count, None, :],
        layers=np.asarray(args.layers),
        sample_id=sample_ids,
        shift=np.asarray([pair_meta["category"]] * count),
        pair_ids=pair_ids,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({**metadata, "shape": list(clean.shape)}, indent=2))


if __name__ == "__main__":
    main()
