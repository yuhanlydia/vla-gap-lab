#!/usr/bin/env python3
"""Validate the tracked project-level Gate-0 summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gap_lab.results_manifest import load_and_validate_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("results/gate0_summary.yaml"))
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    args = parser.parse_args()
    report = load_and_validate_results(args.manifest, args.repository_root)
    print(
        json.dumps(
            {
                "valid": True,
                "tracks": {
                    name: {"gate_passed": result["gate_passed"]}
                    for name, result in report["tracks"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
