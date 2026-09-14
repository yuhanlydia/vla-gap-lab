#!/usr/bin/env python3
"""Create paper-ready tables, plots, and frozen decision for the ICASSP study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vla_gap_lab.state_motion import paper_decision

PRIMARY_RIDGE = "full64__pre_contact__ridge"
PRIMARY_MLP = "full64__pre_contact__mlp"


def _load(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("study") != "icassp_state_motion":
        raise ValueError(f"{path}: not an ICASSP state-motion report")
    return report


def _row(task_label: str, cell: str, values: dict) -> dict:
    token_mode, contact_mode, probe = cell.split("__")
    return {
        "task": task_label,
        "token_mode": token_mode,
        "contact_mode": contact_mode,
        "probe": probe,
        **values,
    }


def _latex_table(rows: list[dict]) -> str:
    selected = [row for row in rows if row["contact_mode"] == "pre_contact"]
    lines = [
        r"\begin{tabular}{llccc}",
        r"\hline",
        r"Task & Setting & $R^2_{\rm pos}$ & $R^2_{v_y}$ & Gap \\",
        r"\hline",
    ]
    for row in selected:
        setting = f'{row["token_mode"]}/{row["probe"]}'
        lines.append(
            f'{row["task"]} & {setting} & '
            f'{row["position_r2_mean_median"]:.3f} & '
            f'{row["velocity_y_r2_median"]:.3f} & '
            f'{row["state_motion_gap_median"]:.3f} \\\\'
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium-report", type=Path, required=True)
    parser.add_argument("--fast-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tracked-summary",
        type=Path,
        default=Path("results/track2_icassp_state_motion_summary.json"),
    )
    args = parser.parse_args()

    medium = _load(args.medium_report)
    fast = _load(args.fast_report)
    if medium["task"] != "InterceptMedium-VLA-v0" or fast["task"] != "InterceptFast-VLA-v0":
        raise ValueError("reports must be Medium then Fast")

    rows = []
    for label, report in (("Medium", medium), ("Fast", fast)):
        for cell, values in report["aggregates"].items():
            rows.append(_row(label, cell, values))

    decision = paper_decision(
        medium["aggregates"][PRIMARY_RIDGE],
        medium["aggregates"][PRIMARY_MLP],
        fast["aggregates"][PRIMARY_RIDGE],
        fast["aggregates"][PRIMARY_MLP],
    )
    summary = {
        "schema_version": 1,
        "study": "icassp_state_motion",
        "medium_success_rate": medium["success_rate"],
        "fast_success_rate": fast["success_rate"],
        "rows": rows,
        "decision": decision,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "icassp_state_motion_summary.json"
    csv_path = args.output_dir / "icassp_state_motion_summary.csv"
    tex_path = args.output_dir / "icassp_state_motion_table.tex"
    png_path = args.output_dir / "icassp_state_motion_primary.png"
    pdf_path = args.output_dir / "icassp_state_motion_primary.pdf"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    args.tracked_summary.parent.mkdir(parents=True, exist_ok=True)
    args.tracked_summary.write_text(json.dumps(summary, indent=2) + "\n")

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tex_path.write_text(_latex_table(rows))

    import matplotlib.pyplot as plt
    import numpy as np

    primary_rows = [
        _row("Medium", PRIMARY_RIDGE, medium["aggregates"][PRIMARY_RIDGE]),
        _row("Medium", PRIMARY_MLP, medium["aggregates"][PRIMARY_MLP]),
        _row("Fast", PRIMARY_RIDGE, fast["aggregates"][PRIMARY_RIDGE]),
        _row("Fast", PRIMARY_MLP, fast["aggregates"][PRIMARY_MLP]),
    ]
    labels = [f'{row["task"]}\n{row["probe"].upper()}' for row in primary_rows]
    pos = np.asarray([row["position_r2_mean_median"] for row in primary_rows])
    vel = np.asarray([row["velocity_y_r2_median"] for row in primary_rows])
    pos_err = np.vstack([
        pos - np.asarray([row["position_r2_mean_q25"] for row in primary_rows]),
        np.asarray([row["position_r2_mean_q75"] for row in primary_rows]) - pos,
    ])
    vel_err = np.vstack([
        vel - np.asarray([row["velocity_y_r2_q25"] for row in primary_rows]),
        np.asarray([row["velocity_y_r2_q75"] for row in primary_rows]) - vel,
    ])
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - width / 2, pos, width, yerr=pos_err, capsize=3, label="Position")
    ax.bar(x + width / 2, vel, width, yerr=vel_err, capsize=3, label="Velocity-y")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_ylabel(r"Held-out $R^2$")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({
        "decision": decision,
        "outputs": [
            str(json_path), str(csv_path), str(tex_path), str(png_path),
            str(pdf_path), str(args.tracked_summary),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
