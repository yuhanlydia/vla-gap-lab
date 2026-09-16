#!/usr/bin/env python3
"""Summarize v3 only, without mixing Ridge headlines with an MLP decision."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vla_gap_lab.icassp_v3 import evaluate_claim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--medium-report', type=Path, required=True)
    parser.add_argument('--fast-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--tracked-summary', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in (args.medium_report, args.fast_report)]
    if any(not r.get('canonical_protocol', False) for r in reports):
        raise ValueError('noncanonical/synthetic reports cannot produce a formal claim decision')
    decision = evaluate_claim(*reports)
    rows = []
    for report in reports:
        for cell in report['cells']:
            rows.append({'task': report['task'], **{k: cell[k] for k in (
                'split_seed', 'representation', 'mask', 'probe', 'lag', 'position_r2_mean',
                'velocity_y_r2', 'fit_converged', 'primary_eligible')}})
    compact = {'schema_version': 3, 'study': 'icassp_motion_compression_v3',
               'protocol_hash': reports[0]['protocol_hash'], 'decision': decision,
               'tasks': {r['task']: {'success_rate': r['success_rate'], 'aggregates': r['aggregates'],
                                     'source_manifest': r['source_manifest']} for r in reports},
               'rows': rows,
               'caution': 'Post-audit reanalysis. No score clipping, no failed-fit omission, no new claim of causal information loss.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.output_dir/'summary.json', args.tracked_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(compact, indent=2, allow_nan=False) + '\n')
    with (args.output_dir/'all_cells.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Row labels explicitly identify the probe; MLP diagnostics cannot masquerade as Ridge.
    table = [r'\begin{tabular}{llllrr}', r'\hline',
             r'Task & Representation & Mask & Probe & $R^2_p$ & $R^2_{v_y}$ \\', r'\hline']
    for report in reports:
        for name, stats in report['aggregates'].items():
            rep, mask, probe, lag = name.split('__')
            pos, vel = stats['position_r2_mean_median'], stats['velocity_y_r2_median']
            if pos is None or vel is None:
                pair = '-- & --'
            else:
                pair = f'{pos:.3f} & {vel:.3f}'
            escaped_rep = (rep + '/' + lag).replace('_', r'\_')
            escaped_mask = mask.replace('_', r'\_')
            table.append(f'{report["task"].split("-")[0]} & {escaped_rep} & {escaped_mask} & {probe} & {pair} \\\\')
    table.extend([r'\hline', r'\end{tabular}'])
    (args.output_dir/'table.tex').write_text('\n'.join(table) + '\n')
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
