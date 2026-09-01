#!/usr/bin/env python3
"""Aggregate an explicit set of layerwise probe reports and apply the Gate-0 threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gap_lab.probe_sweep import load_reports, summarize_probe_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_probe_reports(load_reports(args.reports), args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
