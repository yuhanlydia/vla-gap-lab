# Track 2 Gate-2 — Storage–Dynamics Gap

Date frozen: 2026-09-03

## Why Gate-1 is stopped

The Identity–Location Gate-1 is complete and failed. On
`ShellGameShuffleTouch-VLA-v0`, the released K=2/K=8 policies themselves are at
or near the floor (0/30 and 1/30 in the Short pilot; 2/30 and 1/30 in the Long
pilot). The IPSI intervention changes the slot probe strongly (+67.95pp) but
moves closed-loop SR only from 0% to 2%. That task is also not one of the five
tasks used to train the released mu-VLA checkpoint. A local memory edit cannot
cleanly diagnose control when the underlying policy has almost no task
competence.

A separate code audit found one protocol mismatch in the historical local
adapter: it delegated 128px image resizing to the Hugging Face processor and did
not explicitly reproduce the released evaluator's 224px resize + 0.9 center
crop used for `image_aug=True`. New experiments therefore use
`ProtocolMatchedMuVLAPolicy`. Historical Gate-0/Gate-1 artifacts are not
rewritten.

## New question

The released K=2 memory checkpoint strongly improves static/persistent recall
training tasks, but its released score on the in-distribution
`InterceptMedium-VLA-v0` is about the same as the no-memory baseline. Intercept
requires predictive control of a moving ball. This motivates a narrower
phenomenon-first question:

> Does recurrent memory retain the current physical state while failing to
> encode the temporal dynamics needed to predict motion?

Formally, we test whether

```text
I(M_t ; position_t) is high
but
I(M_t ; velocity_t) is low.
```

No new VLA method is authorized until this diagnostic is established.

## Hardware policy

The primary target is a **16 GB GPU**.

- Load the 7B checkpoint in NF4 4-bit.
- Use one simulator environment at a time. Recurrent memory must advance once
  per simulator tick; vectorizing asynchronous closed-loop episodes before
  parity is established is not worth the protocol risk.
- Every dynamics episode is written atomically, so long runs can resume without
  retaining a large cache in GPU memory.
- After states are cached, use `IncrementalPCA(batch_size=256)` so the expensive
  high-dimensional work is batched on CPU/RAM. The probe itself is small.
- If 4-bit train-task parity fails, repeat only the failed parity task in BF16
  on a 24 GB GPU before blaming the scientific hypothesis.

## Stage 0 — released-checkpoint parity (mandatory)

Use the official K=2 checkpoint and official seeds. Three tasks span the
released checkpoint's train distribution:

- `ShellGamePush-VLA-v0` — released receding-horizon SR about 0.96;
- `InterceptMedium-VLA-v0` — about 0.55;
- `RememberColor5-VLA-v0` — about 0.94.

Run 20 episodes per task first. A 20pp absolute tolerance is deliberately loose:
this stage only detects a broken evaluation path, not paper-level replication.

```bash
cd external/MIKASA-Robo
CKPT=../../models/mu-vla-m64-k2
for task in ShellGamePush-VLA-v0 InterceptMedium-VLA-v0 RememberColor5-VLA-v0; do
  PYTHONPATH=../../src uv run python ../../scripts/eval_mu_vla_protocol.py \
    --checkpoint "$CKPT" --task "$task" --precision 4bit \
    --episodes 20 --start-seed 4242424242 --resume \
    --output "../../artifacts/mikasa/parity_${task}_k2_4bit_n20.json"
done
cd ../..

PYTHONPATH=src python scripts/analyze_mu_vla_train_parity.py \
  --report ShellGamePush-VLA-v0=artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_4bit_n20.json \
  --report InterceptMedium-VLA-v0=artifacts/mikasa/parity_InterceptMedium-VLA-v0_k2_4bit_n20.json \
  --report RememberColor5-VLA-v0=artifacts/mikasa/parity_RememberColor5-VLA-v0_k2_4bit_n20.json \
  --tolerance-pp 20 \
  --output artifacts/reports/mu_vla_k2_protocol_parity_4bit_n20.json
```

**Stop rule:** Gate-2 cannot run unless parity passes. If only one task fails,
rerun that task in BF16 on a 24 GB GPU with the same 20 seeds. If BF16 passes
but 4-bit fails, record quantization as the blocker and do not mix precisions in
the main experiment.

## Stage 1 — collect in-distribution predictive trajectories

Only after parity passes:

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

Each episode stores, before and after the recurrent update:

- memory tokens `0,8,...,56` (float16);
- ball XY position;
- current ball XY velocity;
- initial XY velocity (`oracle_info`, diagnostic label only);
- goal and TCP XY position;
- contact/reached status;
- executed action and reward.

Simulator labels are never inputs to the policy.

## Stage 2 — leakage-safe dynamics probe

Use 24/8/8 whole episodes for train/dev/test. By default exclude the first two
steps and all post-contact rows, because the cleanest question is whether the
memory can infer free-flight velocity from visual history before the robot
changes the ball dynamics.

```bash
PYTHONPATH=src python scripts/probe_mu_vla_predictive_dynamics.py \
  --episodes-dir artifacts/mikasa/intercept_medium_k2_dynamics_n40 \
  --train-episodes 24 --dev-episodes 8 --test-episodes 8 \
  --pca-dim 128 --pca-batch-size 256 \
  --alpha-grid 0.1,1,10,100 --min-step 2 \
  --output artifacts/reports/intercept_medium_k2_predictive_dynamics.json
```

Three feature sets are tested independently:

1. `memory_before`;
2. `memory_after`;
3. `memory_delta = memory_after - memory_before`.

For each, ridge probes predict current position, current velocity, and initial
velocity. PCA is fit **only on train episodes**, ridge alpha is selected on dev,
and the reported R2 is untouched test performance.

## Decision tree

### A. Storage–Dynamics Gap

Continue only if:

```text
R2(position | memory_after) >= 0.50
R2(velocity | memory_after) <= 0.20
```

This would support the specific claim that recurrence stores state but does not
form a useful dynamics representation. Do not immediately train a large memory
model; first perform a causal test of a minimal temporal operator.

### B. Dynamics already represented

If:

```text
R2(velocity | memory_after) >= 0.50
```

then the gap is not storage. The next experiment becomes a
**Dynamics-to-Control Utilization** intervention: change a validated velocity
direction in memory and measure action sensitivity. This is analogous to the
original latent-to-action idea, but now on an in-distribution task with usable
policy competence.

### C. Dynamics concentrated in the update

If:

```text
R2(velocity | memory_delta) - R2(velocity | memory_after) >= 0.15
```

then the recurrent *change* carries more motion information than the stored
state. The next minimal experiment is an update-readout, not a new memory
architecture.

### D. No clear signal

If none of A/B/C holds, stop Track 2. Do not manufacture a method around the
failed probe.

## Not authorized in this stage

- no Dual-Timescale Memory training;
- no new tracking benchmark;
- no tuning on the failed ShuffleTouch Gate-1 seeds;
- no simulator-label input to the policy;
- no vectorized multi-environment rollout until the 16GB batch-1 path matches
  the released train-task checkpoint behavior.
