#!/usr/bin/env python3
"""Analyze paired binary conditions emitted by a closed-loop evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gap_lab.paired_binary import paired_binary_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--reference", default="counterfactual_clean")
    parser.add_argument("--shifted", default="benchmark_task")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    by_condition = {result["condition"]: result for result in report["results"]}
    summary = paired_binary_summary(
        by_condition[args.reference]["success_by_episode"],
        by_condition[args.shifted]["success_by_episode"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    summary.update({"reference": args.reference, "shifted": args.shifted})
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
