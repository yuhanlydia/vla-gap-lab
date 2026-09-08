# Track 2 Gate-2 — Storage–Dynamics Gap

Date frozen: 2026-09-03  
Stage-0 root-cause resolution: 2026-09-08

Status: completed. See
`docs/results/track2_predictive_dynamics_gate2_2026-09-08.md`.

## Why Gate-1 is stopped

The Identity–Location Gate-1 is complete and failed. On
`ShellGameShuffleTouch-VLA-v0`, the released K=2/K=8 policies themselves are at
or near the floor. IPSI changed the slot probe strongly but did not causally
rescue closed-loop success. Do not train IPSI or Dual-Timescale Memory from that
result.

The active question is moved **in distribution** to the released checkpoint's
training task `InterceptMedium-VLA-v0`.

## New question

Does recurrent VLA memory retain current physical state while failing to encode
the temporal dynamics needed for predictive control?

```text
I(M_t ; position_t) is high
but
I(M_t ; velocity_t) is low.
```

No new VLA method is authorized until this diagnostic is established.

## Hardware policy

Primary target: **16 GB GPU**.

- 7B checkpoint in NF4 4-bit;
- one recurrent simulator environment at a time;
- crash-safe one-episode-at-a-time state collection;
- cached high-dimensional analysis uses `IncrementalPCA(batch_size=256)`;
- 24 GB BF16 is only a confirmation tier for a parity discrepancy.

## Stage 0 — released-checkpoint parity (mandatory)

Three released training tasks are used as an evaluator sanity gate:

- `ShellGamePush-VLA-v0` — released receding-horizon SR about 0.96;
- `InterceptMedium-VLA-v0` — about 0.55;
- `RememberColor5-VLA-v0` — about 0.94.

The original 2026-09-08 parity run obtained 0%, 45%, and 90%. A post-hoc trace
found that the local rollout omitted the official evaluator's per-step
`env.render()` synchronization. Without it, ShellGamePush mugs fell through the
table. The fixed same-seed NF4 replacement scored 20/20 (100%), so all three
tasks pass the frozen 20pp tolerance.

The YCB archive mismatch was real, but all `025_mug` files are byte-identical
between the two archives, so it was not causal. Exact YCB provenance remains
pinned for reproducibility. See
`docs/results/track2_ycb_asset_audit_2026-09-08.md`.

### Stage 0A — runtime check

```bash
cd external/MIKASA-Robo
PYTHONPATH=../../src uv run python ../../scripts/check_mu_vla_runtime.py
```

It must report the memory-aware mu-VLA Transformers fork at
`9dbc09f574912a45dd0d71354c035e3c37bcce9e` with Transformers 4.40.1 and
Tokenizers 0.19.1.

### Stage 0B — install the exact YCB archive

```bash
cd external/MIKASA-Robo
PYTHONPATH=../../src uv run python ../../scripts/install_track2_ycb_asset.py
```

The installer pins:

```text
repo      haosulab/ManiSkill2
revision  4c134e2e33b11b1fa751c1d7b6afb2e89231b875
file      data/mani_skill2_ycb.zip
sha256    174001ba1003cc0c5adda6453f4433f55ec7e804f0f0da22d015d525d02262fb
```

`eval_mu_vla_protocol.py` now refuses to run ShellGamePush without the matching
asset provenance marker.

### Stage 0C — completed ShellGamePush replacement

The fixed evaluator was run with the same checkpoint, precision, seeds, and
preprocessing:

```bash
cd external/MIKASA-Robo
PYTHONPATH=../../src uv run python ../../scripts/eval_mu_vla_protocol.py \
  --checkpoint ../../models/mu-vla-m64-k2 \
  --task ShellGamePush-VLA-v0 \
  --precision 4bit \
  --episodes 20 --start-seed 4242424242 \
  --output ../../artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_4bit_ycbpinned_n20.json
```

### Stage-0 decision

ShellGamePush scored 20/20 (100%), above the required 76%. Combined with
InterceptMedium 45% and RememberColor5 90%, Stage-0 passes. Proceed to Stage 1.
The previous BF16 run is obsolete because it used the broken local rollout.

## Stage 1 — collect in-distribution predictive trajectories

Only after Stage-0 parity passes:

```bash
cd external/MIKASA-Robo
PYTHONPATH=../../src uv run python ../../scripts/collect_mu_vla_dynamics_trajectory.py \
  --checkpoint ../../models/mu-vla-m64-k2 \
  --task InterceptMedium-VLA-v0 \
  --episodes 40 --start-seed 4242624242 \
  --precision 4bit --pooling strided --resume \
  --output-dir ../../artifacts/mikasa/intercept_medium_k2_dynamics_n40
cd ../..
```

Each episode stores memory before/after the recurrent update and simulator-only
diagnostic labels for ball position, current velocity, initial velocity, goal,
TCP position, contact status, executed action and reward. Simulator labels are
never policy inputs.

## Stage 2 — leakage-safe dynamics probe

Use whole-episode splits: 24 train / 8 dev / 8 test. Exclude the first two
steps and post-contact rows by default.

```bash
PYTHONPATH=src python scripts/probe_mu_vla_predictive_dynamics.py \
  --episodes-dir artifacts/mikasa/intercept_medium_k2_dynamics_n40 \
  --train-episodes 24 --dev-episodes 8 --test-episodes 8 \
  --pca-dim 128 --pca-batch-size 256 \
  --alpha-grid 0.1,1,10,100 --min-step 2 \
  --output artifacts/reports/intercept_medium_k2_predictive_dynamics.json
```

Probe independently:

1. `memory_before`;
2. `memory_after`;
3. `memory_delta = memory_after - memory_before`.

Predict current position, current velocity, and initial velocity. PCA is fit on
train episodes only, ridge alpha is selected on dev only, and final R2 is on
untouched test episodes.

## Decision tree

### A. Storage–Dynamics Gap

```text
R2(position | memory_after) >= 0.50
R2(velocity | memory_after) <= 0.20
```

Then state is stored but useful dynamics are not. Next run a minimal causal
temporal operator before training any larger memory architecture.

### B. Dynamics already represented

```text
R2(velocity | memory_after) >= 0.50
```

Then the gap is not representation. Next test a **Dynamics-to-Control
Utilization Gap** by intervening on a validated velocity direction and measuring
action sensitivity.

### C. Dynamics concentrated in the update

```text
R2(velocity | memory_delta) - R2(velocity | memory_after) >= 0.15
```

Then motion information is more available in the recurrent update than in the
committed state. Test a minimal update-readout rather than a new memory stack.

### D. No clear signal

If none of A/B/C holds, stop Track 2.

## Not authorized in this stage

- no Dual-Timescale Memory training;
- no new tracking benchmark;
- no tuning on failed ShuffleTouch seeds;
- no simulator-label input to the policy;
- no multi-environment recurrent rollout before the batch-1 parity path is
  established.
