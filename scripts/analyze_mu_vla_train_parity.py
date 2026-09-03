#!/usr/bin/env python3
"""Compare protocol-matched train-task pilots to released mu-VLA references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gap_lab.mu_vla_parity import evaluate_parity

REFERENCE_SR = {
    "ShellGamePush-VLA-v0": 0.96,
    "InterceptMedium-VLA-v0": 0.55,
    "RememberColor5-VLA-v0": 0.94,
}


def parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use TASK=PATH")
    task, path = value.split("=", 1)
    return task, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=parse_report, required=True)
    parser.add_argument("--tolerance-pp", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observed = {}
    details = {}
    for task, path in args.report:
        report = json.loads(path.read_text())
        if report.get("task") != task:
            raise ValueError(f"{path}: task mismatch")
        observed[task] = float(report["success_rate"])
        details[task] = str(path)
    summary = evaluate_parity(
        observed,
        REFERENCE_SR,
        tolerance_pp=args.tolerance_pp,
    )
    summary["reports"] = details
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
