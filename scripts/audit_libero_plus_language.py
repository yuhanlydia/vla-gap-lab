#!/usr/bin/env python3
"""Count task instructions contaminated by perturbation filename metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from vla_gap_lab.libero_language import (
    canonical_instruction_from_name,
    filename_derived_instruction,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    classification = json.loads(args.classification.read_text())

    totals: Counter[str] = Counter()
    contaminated: Counter[str] = Counter()
    examples = {}
    for suite, tasks in classification.items():
        for task in tasks:
            category = task["category"]
            totals[category] += 1
            if category == "Language Instructions":
                continue
            canonical = canonical_instruction_from_name(task["name"], category)
            upstream = filename_derived_instruction(task["name"])
            if upstream != canonical:
                contaminated[category] += 1
                examples.setdefault(
                    category,
                    {"suite": suite, "task": task["name"], "upstream": upstream, "canonical": canonical},
                )

    report = {
        "schema_version": 1,
        "total_tasks": sum(totals.values()),
        "filename_contaminated_tasks": sum(contaminated.values()),
        "by_category": {
            category: {"total": totals[category], "filename_contaminated": contaminated[category]}
            for category in sorted(totals)
        },
        "examples": examples,
        "definition": (
            "non-language task names are converted wholesale from underscores to spaces by "
            "LIBERO-Plus grab_language_from_filename"
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
