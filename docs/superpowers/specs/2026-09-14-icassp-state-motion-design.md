# ICASSP State–Motion Accessibility Study Design

## Goal

Turn the stopped Track-2 method line into a narrow diagnostic ICASSP study answering one question:

> Does recurrent VLA memory make instantaneous spatial state substantially more accessible than object motion?

The paper does **not** claim that velocity is absent from memory or that the observed accessibility asymmetry is a causal performance bottleneck. It reports a controlled representation diagnostic on released μVLA under protocol-matched MIKASA-Robo evaluation.

## Primary tasks

Use the released μVLA K=2 checkpoint on two in-distribution dynamic environments:

- `InterceptMedium-VLA-v0`
- `InterceptFast-VLA-v0`

Run 60 episodes per task, one environment at a time, NF4 4-bit, with the exact memory-aware Transformers fork and `render_after_step` synchronization already validated in Track 2.

## Collection

Each timestep stores:

- all 64 recurrent memory tokens after the policy update, float16;
- ball XY position;
- ball XY linear velocity;
- goal XY position;
- TCP XY position;
- `reached_status`;
- executed 7-D action;
- reward and success.

Only `memory_after` is stored for the new ICASSP collection. The stopped `memory_before`/`memory_delta` causal line is retained as prior Track-2 evidence and is not rerun.

## Leakage-safe probe protocol

For each task use five deterministic whole-episode split seeds. Each split is 36 train / 12 dev / 12 test episodes. PCA is fit on train rows only. Hyperparameters are selected on dev only. Test episodes are untouched until final scoring.

Primary rows exclude the first two timesteps and post-contact rows (`reached_status >= 0.5`).

### Primary representation

- token coverage: all 64 memory tokens;
- PCA dimension: 256;
- probe families: Ridge and fixed two-layer MLP;
- targets: current ball position XY and current ball velocity XY;
- report mean XY R² and per-axis R², including task-relevant velocity-y.

### Reviewer-facing ablations

1. **Token coverage:** full 64 tokens vs. legacy stride-8 tokens `[0,8,...,56]`.
2. **Probe capacity:** Ridge vs. nonlinear 2-layer MLP.
3. **Contact filtering:** pre-contact only vs. all valid timesteps after step 2.
4. **Dynamic regime replication:** Medium vs. Fast.

No additional architecture search, kernel sweep, temporal operator tuning, or new VLA training is authorized.

## Frozen interpretation / kill rule

The ICASSP diagnostic is considered supported only if the full-memory result remains state-favoring after the stronger probe and replicates on Fast.

For each probe family compute the median across five split seeds.

Go condition:

- Medium: position mean R² >= 0.65;
- Medium: position mean R² - velocity-y R² >= 0.25;
- Fast: position mean R² > velocity-y R²;
- neither full-memory MLP velocity-y R² reaches 0.60 on Medium.

If full-memory MLP reaches velocity-y R² >= 0.60 on Medium, or Fast reverses the ordering, the paper claim is abandoned rather than rescued with more tasks.

## Output contract

A single shell entrypoint must:

1. validate the μVLA runtime;
2. collect Medium and Fast caches with resume support;
3. run all primary and ablation probes;
4. emit per-task JSON reports;
5. emit a combined JSON summary, CSV, LaTeX table, and PDF/PNG figure under `artifacts/reports/`;
6. copy a small machine-readable summary to `results/track2_icassp_state_motion_summary.json` so the GPU agent can commit and push it for review.

## Hardware

Primary target: 16 GB GPU. Closed-loop rollout stays batch 1. PCA and probes run offline after caching.
