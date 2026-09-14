# Track 2 ICASSP 2027 — State–Motion Accessibility Study

## Paper question

Does recurrent VLA memory make instantaneous spatial state substantially more accessible than object motion?

This is a diagnostic paper, not a new-memory-method paper. The stopped temporal-operator intervention remains a negative supporting result and is not tuned further.

## One-command run

From the repository root:

```bash
git pull origin main
git submodule update --init --recursive
bash scripts/run_track2_icassp.sh
```

Optional checkpoint override:

```bash
bash scripts/run_track2_icassp.sh /absolute/path/to/mu-vla-m64-k2
```

The runner installs the Track-2 extras in the pinned MIKASA environment, validates the exact μVLA memory-aware Transformers runtime, collects both dynamic regimes, runs five whole-episode splits, executes all frozen ablations, and generates paper-ready summaries.

## Frozen collection

Checkpoint: released μVLA K=2, NF4 4-bit, batch-1 recurrent rollout.

| Task | Episodes | Start seed |
|---|---:|---:|
| `InterceptMedium-VLA-v0` | 60 | 4243024242 |
| `InterceptFast-VLA-v0` | 60 | 4243124242 |

Every timestep stores all 64 post-update recurrent memory tokens plus ball position/velocity, goal, TCP, contact state, action, reward, and success. Collection is episode-atomic and `--resume` safe.

## Probe protocol

Five split seeds: `0,1,2,3,4`.

Each split:

- train: 36 whole episodes;
- dev: 12 whole episodes;
- test: 12 untouched whole episodes;
- minimum step: 2;
- primary rows: pre-contact only.

Dimensionality reduction is train-only:

1. token PCA: 4096 → 32 per memory token;
2. flatten selected tokens;
3. sample-level PCA: → 256.

Primary probes:

- full 64 memory tokens + Ridge;
- full 64 memory tokens + fixed 2-layer MLP `(128,64)`.

Ablations:

- stride-8 tokens + Ridge (token-coverage ablation);
- full 64 tokens + all steps + Ridge (contact-filter ablation).

Metrics are held-out position XY R², velocity XY R², task-relevant velocity-y R², and the state-motion accessibility gap:

```text
Gap = R2(position_xy mean) - R2(velocity_y)
```

All paper numbers are median across the five split seeds; IQR is retained for uncertainty/error bars.

## Frozen GO / STOP rule

The ICASSP claim is allowed only if both full-memory probe families preserve the state-favoring result and the ordering replicates on Fast.

Required:

```text
Medium position R2 >= 0.65                 (Ridge and MLP)
Medium state-motion gap >= 0.25           (Ridge and MLP)
Fast state-motion gap > 0                 (Ridge and MLP)
Medium MLP velocity-y R2 < 0.60
```

If the nonlinear full-memory probe makes velocity strongly accessible or Fast reverses the ordering, stop the paper rather than add more tasks.

## Outputs

Raw episode caches:

```text
artifacts/mikasa/icassp_state_motion_medium_n60/
artifacts/mikasa/icassp_state_motion_fast_n60/
```

Paper outputs:

```text
artifacts/reports/icassp_state_motion/medium.json
artifacts/reports/icassp_state_motion/fast.json
artifacts/reports/icassp_state_motion/icassp_state_motion_summary.json
artifacts/reports/icassp_state_motion/icassp_state_motion_summary.csv
artifacts/reports/icassp_state_motion/icassp_state_motion_table.tex
artifacts/reports/icassp_state_motion/icassp_state_motion_primary.png
artifacts/reports/icassp_state_motion/icassp_state_motion_primary.pdf
```

A compact summary is also written to:

```text
results/track2_icassp_state_motion_summary.json
```

After the run, the GPU agent should commit and push that single tracked summary so the results can be reviewed without committing multi-GB caches:

```bash
git add results/track2_icassp_state_motion_summary.json
git commit -m "results: add ICASSP state-motion study"
git push
```
