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
    parser.add_argument(
        "--shifted-report",
        type=Path,
        help="optional second report when conditions were evaluated in separate runs",
    )
    parser.add_argument("--reference", default="counterfactual_clean")
    parser.add_argument("--shifted", default="benchmark_task")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    shifted_report = json.loads(args.shifted_report.read_text()) if args.shifted_report else report
    reference_results = {result["condition"]: result for result in report["results"]}
    shifted_results = {result["condition"]: result for result in shifted_report["results"]}
    summary = paired_binary_summary(
        reference_results[args.reference]["success_by_episode"],
        shifted_results[args.shifted]["success_by_episode"],
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
