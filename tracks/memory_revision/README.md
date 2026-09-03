# Track 2 — Memory Structure in VLA Control

The original Persistence–Revision / Identity–Location line has completed its
causal test and is **stopped**. On `ShellGameShuffleTouch-VLA-v0`, the released
mu-VLA policy appeared at the floor (K=2: 0/30, K=8: 1/30). IPSI corrected the
slot probe strongly but changed closed-loop success only from 0% to 2% on 50
paired seeds. The preregistered Gate-1 therefore failed; do not train
Dual-Timescale Memory from that result.

However, a later runtime audit found that the historical Track-2 setup
instructions installed the non-memory `moojink/transformers-openvla-oft` fork
instead of the exact memory-aware mu-VLA Transformers fork. Those absolute
scores are therefore **protocol-compromised until replicated**. The qualitative
negative result remains useful for deciding what to retest, but it is not a
faithful released-mu-VLA benchmark claim.

See
[`docs/results/track2_runtime_root_cause.md`](../../docs/results/track2_runtime_root_cause.md)
for the root-cause audit.

The earlier ShuffleTouch setup also has a task/model mismatch: ShuffleTouch is
not one of the five tasks used to train the released MIKASA mu-VLA checkpoint
and requires a different tracking-style memory structure. It is not a clean
test bed for a small causal memory edit even after the runtime is corrected.

## Active question: Storage–Dynamics Gap

The next phenomenon is deliberately moved **in distribution**. The released
K=2 checkpoint strongly benefits persistent/static memory tasks but shows only
a small released gain over no-memory on `InterceptMedium-VLA-v0`, one of its
five training tasks. Intercept requires predictive motion information rather
than cue storage.

The diagnostic asks:

```text
Does recurrent memory encode current physical state
without encoding the temporal dynamics needed for predictive control?
```

Before testing this, the local evaluator must pass a train-task protocol parity
check. New runs use `ProtocolMatchedMuVLAPolicy`, which:

- requires the exact official memory-aware Transformers VCS revision;
- checks `transformers==4.40.1` and `tokenizers==0.19.1`;
- reproduces the released 224px resize plus 0.9 center crop;
- clips actions to `[-1, 1]` before the simulator step.

Execution is frozen in
[`docs/experiments/track2_predictive_dynamics_gate2.md`](../../docs/experiments/track2_predictive_dynamics_gate2.md).

Order:

1. reinstall Track-2 requirements and run `scripts/check_mu_vla_runtime.py`;
2. 16GB 4-bit parity on `ShellGamePush`, `InterceptMedium`, and
   `RememberColor5`;
3. only if parity passes, collect 40 `InterceptMedium` recurrent trajectories;
4. probe position, velocity, and initial velocity from memory-before,
   memory-after, and memory-delta using episode-held-out splits;
5. choose the next causal experiment from the diagnostic branch; do not train a
   new memory architecture before that branch is established.

Historical IPSI and `ConflictAdaptiveRefresh` code remains only for
reproducibility.
