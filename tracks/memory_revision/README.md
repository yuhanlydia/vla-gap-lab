# Track 2 — Memory Structure in VLA Control

The original Persistence–Revision / Identity–Location line has completed its
causal test and is **stopped**. On `ShellGameShuffleTouch-VLA-v0`, the released
mu-VLA policy itself is at the floor (K=2: 0/30, K=8: 1/30). IPSI corrected the
slot probe strongly but changed closed-loop success only from 0% to 2% on 50
paired seeds. The preregistered Gate-1 therefore failed; do not train
Dual-Timescale Memory from that result.

The failure is also confounded by task/model mismatch: ShuffleTouch is not one
of the five tasks used to train the released MIKASA mu-VLA checkpoint and
requires a different tracking-style memory structure. It is not a clean test
bed for a small causal memory edit.

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
check. New runs use `ProtocolMatchedMuVLAPolicy`, which explicitly reproduces
the released evaluator's 224px resize plus 0.9 center crop for checkpoints
trained with image augmentation.

Execution is frozen in
[`docs/experiments/track2_predictive_dynamics_gate2.md`](../../docs/experiments/track2_predictive_dynamics_gate2.md).

Order:

1. 16GB 4-bit parity on `ShellGamePush`, `InterceptMedium`, and
   `RememberColor5`;
2. only if parity passes, collect 40 `InterceptMedium` recurrent trajectories;
3. probe position, velocity, and initial velocity from memory-before,
   memory-after, and memory-delta using episode-held-out splits;
4. choose the next causal experiment from the diagnostic branch; do not train a
   new memory architecture before that branch is established.

Historical IPSI and `ConflictAdaptiveRefresh` code remains only for
reproducibility.
