#!/usr/bin/env python3
"""Create paper-ready summaries for the ICASSP motion-compression study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vla_gap_lab.state_motion import motion_compression_decision

MEM_RIDGE = "memory__pre_contact__ridge"
MEM_MLP = "memory__pre_contact__mlp"
CUR_RIDGE = "visual_current__pre_contact__ridge"
CUR_MLP = "visual_current__pre_contact__mlp"
PAIR_RIDGE = "visual_pair_lag1__pre_contact__ridge"
PAIR_MLP = "visual_pair_lag1__pre_contact__mlp"
PAIR_LAG2_RIDGE = "visual_pair_lag2__pre_contact__ridge"
MEM_STRIDE_RIDGE = "memory_stride8__pre_contact__ridge"
MEM_ALL_RIDGE = "memory__all_steps__ridge"


def _load(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("study") != "icassp_motion_compression":
        raise ValueError(f"{path}: not an ICASSP motion-compression report")
    return report


def _row(task_label: str, representation: str, probe: str, values: dict) -> dict:
    return {"task": task_label, "representation": representation, "probe": probe, **values}


def _latex_primary_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{lllccc}", r"\hline",
        r"Task & Representation & Probe & $R^2_{\rm pos}$ & $R^2_{v_y}$ & $\Delta$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f'{row["task"]} & {row["representation"]} & {row["probe"]} & '
            f'{row["position_r2_mean_median"]:.3f} & {row["velocity_y_r2_median"]:.3f} & '
            f'{row["state_motion_gap_median"]:.3f} \\\\'
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _ablation_rows(label: str, report: dict) -> list[dict]:
    ag = report["aggregates"]
    return [
        {"task": label, "ablation": "Memory full64", **ag[MEM_RIDGE]},
        {"task": label, "ablation": "Memory stride8", **ag[MEM_STRIDE_RIDGE]},
        {"task": label, "ablation": "Memory all steps", **ag[MEM_ALL_RIDGE]},
        {"task": label, "ablation": "Two-frame lag1", **ag[PAIR_RIDGE]},
        {"task": label, "ablation": "Two-frame lag2", **ag[PAIR_LAG2_RIDGE]},
    ]


def _latex_ablation_table(rows: list[dict]) -> str:
    lines = [r"\begin{tabular}{llcc}", r"\hline", r"Task & Ablation & $R^2_{\rm pos}$ & $R^2_{v_y}$ \\", r"\hline"]
    for row in rows:
        lines.append(
            f'{row["task"]} & {row["ablation"]} & '
            f'{row["position_r2_mean_median"]:.3f} & {row["velocity_y_r2_median"]:.3f} \\\\'
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium-report", type=Path, required=True)
    parser.add_argument("--fast-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracked-summary", type=Path, default=Path("results/track2_icassp_motion_compression_summary.json"))
    args = parser.parse_args()

    medium = _load(args.medium_report)
    fast = _load(args.fast_report)
    if medium["task"] != "InterceptMedium-VLA-v0" or fast["task"] != "InterceptFast-VLA-v0":
        raise ValueError("reports must be Medium then Fast")

    primary_rows = []
    representation_map = (
        ("Current visual", CUR_RIDGE, "Ridge"), ("Current visual", CUR_MLP, "MLP"),
        ("Two-frame visual", PAIR_RIDGE, "Ridge"), ("Two-frame visual", PAIR_MLP, "MLP"),
        ("Recurrent memory", MEM_RIDGE, "Ridge"), ("Recurrent memory", MEM_MLP, "MLP"),
    )
    for label, report in (("Medium", medium), ("Fast", fast)):
        for representation, cell, probe in representation_map:
            primary_rows.append(_row(label, representation, probe, report["aggregates"][cell]))

    decision = motion_compression_decision(
        medium_current=medium["aggregates"][CUR_MLP],
        medium_pair=medium["aggregates"][PAIR_MLP],
        medium_memory=medium["aggregates"][MEM_MLP],
        fast_pair=fast["aggregates"][PAIR_MLP],
        fast_memory=fast["aggregates"][MEM_MLP],
    )
    ablations = _ablation_rows("Medium", medium) + _ablation_rows("Fast", fast)
    summary = {
        "schema_version": 2,
        "study": "icassp_motion_compression",
        "paper_question": "does recurrent compression preserve motion available in short visual history?",
        "medium_success_rate": medium["success_rate"],
        "fast_success_rate": fast["success_rate"],
        "primary_rows": primary_rows,
        "ablation_rows": ablations,
        "decision": decision,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "icassp_motion_compression_summary.json"
    csv_path = args.output_dir / "icassp_motion_compression_primary.csv"
    tex_path = args.output_dir / "icassp_motion_compression_primary_table.tex"
    ablation_tex_path = args.output_dir / "icassp_motion_compression_ablation_table.tex"
    hero_png = args.output_dir / "icassp_motion_compression_hero.png"
    hero_pdf = args.output_dir / "icassp_motion_compression_hero.pdf"
    memory_png = args.output_dir / "icassp_motion_compression_memory_state_motion.png"
    memory_pdf = args.output_dir / "icassp_motion_compression_memory_state_motion.pdf"

    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    args.tracked_summary.parent.mkdir(parents=True, exist_ok=True)
    args.tracked_summary.write_text(json.dumps(summary, indent=2) + "\n")

    fieldnames = list(primary_rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(primary_rows)
    tex_path.write_text(_latex_primary_table(primary_rows))
    ablation_tex_path.write_text(_latex_ablation_table(ablations))

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    representations = ["Current\nvisual", "Two-frame\nvisual", "Recurrent\nmemory"]
    x = np.arange(3); width = 0.34
    for offset, (label, report) in zip((-width / 2, width / 2), (("Medium", medium), ("Fast", fast))):
        vals = np.array([
            report["aggregates"][CUR_MLP]["velocity_y_r2_median"],
            report["aggregates"][PAIR_MLP]["velocity_y_r2_median"],
            report["aggregates"][MEM_MLP]["velocity_y_r2_median"],
        ])
        q25 = np.array([
            report["aggregates"][CUR_MLP]["velocity_y_r2_q25"],
            report["aggregates"][PAIR_MLP]["velocity_y_r2_q25"],
            report["aggregates"][MEM_MLP]["velocity_y_r2_q25"],
        ])
        q75 = np.array([
            report["aggregates"][CUR_MLP]["velocity_y_r2_q75"],
            report["aggregates"][PAIR_MLP]["velocity_y_r2_q75"],
            report["aggregates"][MEM_MLP]["velocity_y_r2_q75"],
        ])
        ax.bar(x + offset, vals, width, yerr=np.vstack([vals - q25, q75 - vals]), capsize=3, label=label)
    ax.axhline(0.0, linewidth=0.8); ax.set_ylabel(r"Held-out velocity-$y$ $R^2$")
    ax.set_xticks(x, representations); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(hero_png, dpi=220, bbox_inches="tight"); fig.savefig(hero_pdf, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    labels = ["Medium", "Fast"]; x = np.arange(2); width = 0.34
    pos = np.array([medium["aggregates"][MEM_MLP]["position_r2_mean_median"], fast["aggregates"][MEM_MLP]["position_r2_mean_median"]])
    vel = np.array([medium["aggregates"][MEM_MLP]["velocity_y_r2_median"], fast["aggregates"][MEM_MLP]["velocity_y_r2_median"]])
    ax.bar(x - width / 2, pos, width, label="Position"); ax.bar(x + width / 2, vel, width, label="Velocity-y")
    ax.axhline(0.0, linewidth=0.8); ax.set_ylabel(r"Held-out $R^2$"); ax.set_xticks(x, labels); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(memory_png, dpi=220, bbox_inches="tight"); fig.savefig(memory_pdf, bbox_inches="tight"); plt.close(fig)

    print(json.dumps({"decision": decision, "outputs": [str(json_path), str(csv_path), str(tex_path), str(ablation_tex_path), str(hero_png), str(hero_pdf), str(memory_png), str(memory_pdf), str(args.tracked_summary)]}, indent=2))


if __name__ == "__main__":
    main()
