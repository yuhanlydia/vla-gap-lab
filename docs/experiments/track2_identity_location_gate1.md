# Track 2 Gate-1 — Identity–Location Revision

Date frozen: 2026-09-02

## Research question

Gate-0 separated two memory contents after the first cup swap on
`ShellGameShuffleTouch-VLA-v0`: hidden target identity remains decodable, while
current target slot is approximately at three-way chance. Gate-1 tests the
minimal causal claim:

> If recurrent memory is corrected toward the true current slot while its
> identity-probe logits are preserved to first order, does closed-loop success
> recover?

This is a privileged **diagnostic intervention**, not a deployable method. The
simulator target slot is used only to test whether the observed representation
failure is causal. No trainable memory mechanism is authorized until this gate
passes.

## 0. Environment and checkpoints

Use the pinned `external/MIKASA-Robo` stack and the isolated environment in
`scripts/README.md`. Download both official mu-VLA checkpoints if needed:

```bash
huggingface-cli download \
  mu-vla/mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k2-tbptt \
  --local-dir models/mu-vla-m64-k2
huggingface-cli download \
  mu-vla/mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k8-tbptt \
  --local-dir models/mu-vla-m64-k8
```

All runs use `pd_ee_delta_pose`, receding-horizon execution, and `success_once`
(the evaluator ORs the simulator `success` signal over the episode). Do not
change action decoding, image preprocessing, memory-token count, or released
checkpoint weights.

## 1. Horizon-control pilot: K=2 versus K=8

Before fitting any editor, rule out the simplest explanation: K=2 may fail
because its training credit-assignment horizon is too short.

Tasks:

- `ShellGameShuffleTouch-VLA-v0` — primary, 60-step Short Tracking task;
- `ShellGameShuffleTouch-Long-VLA-v0` — 600-step Medium Tracking stress test.

Use the same 30 seeds for K=2 and K=8, starting at `4242474242`.

```bash
cd external/MIKASA-Robo
for ckpt in k2 k8; do
  model=../../models/mu-vla-m64-${ckpt}
  for task in ShellGameShuffleTouch-VLA-v0 ShellGameShuffleTouch-Long-VLA-v0; do
    PYTHONPATH=../../src uv run python ../../scripts/eval_mu_vla_intervention.py \
      --checkpoint "$model" --task "$task" --mode normal \
      --episodes 30 --start-seed 4242474242 --resume \
      --output "../../artifacts/mikasa/${task}_${ckpt}_normal_n30.json"
  done
done
```

**Stop rule:** if K=8 exceeds K=2 by at least 20 percentage points on the
primary Short task, stop IPSI and reinterpret the failure primarily as a
TBPTT/temporal-credit problem. The Long task is a stress replication, not a
substitute for this primary stop rule.

## 2. Collect leakage-safe editor data

Only if the K=8 stop rule does not fire, fit the editor on the primary Short
task using the K=2 checkpoint. Collect 100 episodes beginning at seed
`4242424242`; these seeds are disjoint from the horizon pilot and causal test.
Retain every eighth recurrent memory token (`0, 8, ..., 56`).

```bash
cd external/MIKASA-Robo
PYTHONPATH=../../src uv run python ../../scripts/collect_mu_vla_memory_trajectory.py \
  --checkpoint ../../models/mu-vla-m64-k2 \
  --task ShellGameShuffleTouch-VLA-v0 \
  --episodes 100 --start-seed 4242424242 \
  --pooling strided --checkpoint-every 5 --resume \
  --output ../../artifacts/mikasa/shell_shuffle_k2_strided_n100.npz
```

The collector stores simulator identity/slot labels only for offline subspace
fitting and causal diagnostics. They are never inputs to the normal policy.

## 3. Fit the held-out identity/slot editor

The editor makes no semantic assumption about the 64 memory tokens. It:

1. flattens the eight retained tokens;
2. fits PCA-256 on **train episodes only**;
3. fits balanced ridge classifiers for hidden identity and current slot;
4. chooses ridge alpha from `{1,10,100,1000}` on 20 held-out dev episodes;
5. estimates median natural latent update norm on train trajectories;
6. chooses edit scale from `{0.25,0.5,1,2}` times that median using only dev
   probe behavior, never closed-loop success.

Split: 80 train episodes + 20 dev episodes, grouped by episode.

```bash
cd ../..
PYTHONPATH=src python scripts/fit_mu_vla_identity_slot_editor.py \
  --trajectory artifacts/mikasa/shell_shuffle_k2_strided_n100.npz \
  --train-episodes 80 --dev-episodes 20 --pca-dim 256 \
  --alpha-grid 1,10,100,1000 --scale-grid 0.25,0.5,1,2 \
  --output artifacts/mikasa/shell_shuffle_k2_ipsi_editor.npz \
  --report artifacts/reports/shell_shuffle_k2_ipsi_editor.json
```

For wrong predicted slot `s_hat` and simulator target slot `s*`:

```text
d_slot = w_slot[s*] - w_slot[s_hat]
P_identity_perp = I - pinv(W_identity) W_identity
d_IPSI = normalize(P_identity_perp d_slot)
```

The PCA-space delta is mapped back only to the retained memory tokens. A
`slot_only` control omits identity projection. A `random_orthogonal` control
samples a norm-matched direction in the same identity-orthogonal subspace.

## 4. Offline sanity before causal GPU evaluation

Inspect `artifacts/reports/shell_shuffle_k2_ipsi_editor.json`. Do not proceed if
there are no post-swap dev rows, an identity/slot classifier is degenerate, or
the selected IPSI scale fails to improve the dev slot probe while preserving
the identity probe. This is a sanity check, not a paper result.

## 5. Primary paired causal Gate-1

Use 50 completely new paired seeds beginning at `4242524242`. Run four
conditions on `ShellGameShuffleTouch-VLA-v0`:

- `normal`: untouched recurrent memory;
- `random_orthogonal`: norm-matched random edit orthogonal to identity probe;
- `slot_only`: oracle slot direction without identity preservation;
- `ipsi`: oracle slot direction projected away from identity probe.

Edits occur exactly once after each newly completed swap and before the next
VLA action. The editor evaluator deliberately rejects tasks without the hidden
`cup_with_ball_number` identity, so the color-lamp task cannot silently be used
as an identity-preserving test.

```bash
cd external/MIKASA-Robo
TASK=ShellGameShuffleTouch-VLA-v0
EDITOR=../../artifacts/mikasa/shell_shuffle_k2_ipsi_editor.npz
for mode in normal random_orthogonal slot_only ipsi; do
  extra=()
  if [ "$mode" != normal ]; then
    extra=(--editor "$EDITOR" --edit-seed 0)
  fi
  PYTHONPATH=../../src uv run python \
    ../../scripts/eval_mu_vla_identity_slot_intervention.py \
    --checkpoint ../../models/mu-vla-m64-k2 --task "$TASK" --mode "$mode" \
    --episodes 50 --start-seed 4242524242 --resume \
    "${extra[@]}" \
    --output "../../artifacts/mikasa/${TASK}_${mode}_gate1_n50.json"
done
cd ../..

PYTHONPATH=src python scripts/analyze_mu_vla_identity_location_gate.py \
  --normal artifacts/mikasa/${TASK}_normal_gate1_n50.json \
  --random artifacts/mikasa/${TASK}_random_orthogonal_gate1_n50.json \
  --slot-only artifacts/mikasa/${TASK}_slot_only_gate1_n50.json \
  --ipsi artifacts/mikasa/${TASK}_ipsi_gate1_n50.json \
  --output artifacts/reports/${TASK}_identity_location_gate1.json
```

### Gate-1 criteria

All must hold:

1. `SR(IPSI) - SR(normal) >= 10 pp`;
2. paired-bootstrap 95% CI lower bound for IPSI-minus-normal is `> 0`;
3. `SR(IPSI) - SR(random_orthogonal) >= 8 pp`;
4. slot probe accuracy after IPSI improves by at least 15 pp;
5. identity probe accuracy drops by no more than 5 pp;
6. IPSI beats `slot_only`, **or** `slot_only` damages identity by more than 5 pp.

If the gate fails, stop this mechanism. Do not train a memory architecture just
because the probe result is interesting.

## 6. Long-horizon replication only after the primary gate passes

Reuse the **same Short-task editor** on `ShellGameShuffleTouch-Long-VLA-v0`;
do not refit it on Long test episodes. Run the same four conditions on 50
paired seeds starting at `4242524242`. This asks whether local memory geometry
transfers across horizon rather than merely fitting one task duration.

## 7. Explicitly not authorized yet

- no full memory reset as an identity-preserving oracle;
- no hand assignment of memory tokens to identity versus location roles;
- no `ConflictAdaptiveRefresh` training;
- no Dual-Timescale Memory training;
- no use of color-lamp target color as a hidden identity claim;
- no method selection using causal-test success.

If Gate-1 passes, the next method hypothesis is a fixed-capacity dual-timescale
memory separating slowly changing identity content from fast dynamic state.
That method is a separate decision and must not be implemented before causal
rescue is established.
