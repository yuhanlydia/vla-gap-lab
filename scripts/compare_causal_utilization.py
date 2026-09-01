#!/usr/bin/env python3
"""Compare matched CUR reports with state-cluster bootstrap uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_gap_lab.paired_continuous import paired_cluster_mean_difference


def record_key(row: dict) -> tuple:
    return row["pair_id"], row["action_index"], row["alpha_multiplier"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("shifted", type=Path)
    parser.add_argument("--metric", choices=["cur", "control_gain", "probe_gain"], default="cur")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    shifted = json.loads(args.shifted.read_text())
    reference_rows = {record_key(row): row for row in reference["records"]}
    shifted_rows = {record_key(row): row for row in shifted["records"]}
    if reference_rows.keys() != shifted_rows.keys():
        raise ValueError("reports do not contain identical paired interventions")
    keys = sorted(reference_rows)
    result = paired_cluster_mean_difference(
        np.asarray([reference_rows[key][args.metric] for key in keys]),
        np.asarray([shifted_rows[key][args.metric] for key in keys]),
        np.asarray([key[0] for key in keys]),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report = {
        "schema_version": 1,
        "metric": args.metric,
        "reference": str(args.reference),
        "shifted": str(args.shifted),
        **result,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
