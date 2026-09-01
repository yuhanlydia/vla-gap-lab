# Track 1 large-angle Camera variant (task 611)

This replication tests a difficulty-2 Camera variant of the same LIBERO
Spatial base task used in the earlier task-609 pilot. Natural-language
instructions and initial states are held fixed between clean and shifted runs.

## Closed-loop control

| Condition | Success |
|---|---:|
| Clean BDDL | 9/10 |
| Camera `view_13_15_100_0_0` | 1/10 |

- Control retention: **0.111**
- Paired difference (shift - clean): **-0.80**
- Episode-cluster bootstrap 95% interval: **[-1.00, -0.50]**
- Discordant pairs: clean-only 8, shifted-only 0
- Exact two-sided McNemar p: **0.0078125**

Unlike the mild task-609 camera change, this variant creates a clear
instruction-matched control failure.

## Representation replication

The offline cache contains 480 identical-state pairs from 30 demonstration
trajectories. Both views are re-rendered in one process.

- Clean/shifted pixel MAE: **29.420 / 255**
- Demonstration/base-rerender audit MAE: **5.143 / 255**
- Layers: 4, 8, 12, 16, 20, 24, 28, 32
- Sweep: alpha `{0.01, 0.1, 1, 10, 100, 1000}` × trajectory split seed
  `{7, 21, 42, 84}`

At alpha below 1, no shifted layer had non-negative R². Across the 16
configurations with any valid shifted probe, the best retention was:

| Alpha | Best retention range |
|---:|---:|
| 1 | 0.026–0.396 |
| 10 | 0.326–0.490 |
| 100 | 0.349–0.475 |
| 1000 | 0.523–0.608 |

The maximum over all layers, alphas, and splits was **0.608**, below the
preregistered 0.8 threshold.

The sweep can be regenerated without manually selecting runs:

```bash
PYTHONPATH=src python3 scripts/summarize_probe_sweep.py \
  artifacts/reports/camera611_n480_a*_s*.json \
  --output artifacts/reports/camera611_n480_sweep_summary.json
```

The aggregator requires every input to reference the same hidden cache,
retains runs whose shifted R² is negative, and records invalid runs instead of
silently excluding them. This produces 24/24 structurally valid runs; the 16
count above is the stricter subset with positive shifted R².

## Gate decision

**Not passed.** This variant satisfies the control-failure half of the
hypothesis but not representation retention. The clean action signal is
linearly decodable, while its shifted-view decodability falls substantially.
The failure is therefore better described as a representation robustness
failure than a latent-to-action utilization failure. ControlSkip remains
untrained.

This is still a single-task 10-episode closed-loop pilot with 4-bit inference,
not a category-level leaderboard reproduction.
