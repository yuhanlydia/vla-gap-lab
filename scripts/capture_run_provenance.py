#!/usr/bin/env python3
"""Write code, dependency, and input-file identities for an experiment run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gap_lab.provenance import capture_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, action="append", default=[])
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument(
        "--package",
        action="append",
        default=["vla-gap-lab", "numpy", "torch", "transformers", "scikit-learn"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repositories = args.repository or [Path.cwd()]
    report = capture_provenance(repositories, args.artifact, args.package)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
