#!/usr/bin/env python3
"""Fit layerwise probes on a hidden cache and apply the preregistered Gate-0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_gap_lab.cache import HiddenCache
from vla_gap_lab.metrics import latent_action_gate, metrics_dict
from vla_gap_lab.probes import fit_layerwise_ridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--control-retention", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = fit_layerwise_ridge(HiddenCache.load(args.cache), alpha=args.alpha, seed=args.seed)
    gate = latent_action_gate(
        np.array([item.clean.r2 for item in results]),
        np.array([item.shifted.r2 for item in results]),
        args.control_retention,
    )
    report = {
        "layers": [
            {"layer": item.layer, "clean": metrics_dict(item.clean), "shifted": metrics_dict(item.shifted)}
            for item in results
        ],
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()

