# Track 2 direction reassessment after Gate-1

Date: 2026-09-03

## Latest evidence

The completed horizon and causal experiments do not support continuing the
Identity–Location intervention as a method line.

### Horizon pilot

`ShellGameShuffleTouch-VLA-v0`, 30 paired seeds:

- K=2: 0/30 (0.00%)
- K=8: 1/30 (3.33%)

`ShellGameShuffleTouch-Long-VLA-v0`, 30 paired seeds:

- K=2: 2/30 (6.67%)
- K=8: 1/30 (3.33%)

Longer TBPTT therefore does not rescue the task.

### Identity–Location Gate-1

`ShellGameShuffleTouch-VLA-v0`, 50 paired seeds:

- normal: 0/50
- random-orthogonal: 0/50
- slot-only: 1/50
- IPSI: 1/50

IPSI changed the representation strongly (slot-probe gain +67.95pp with 0pp
identity-accuracy loss) but improved success only +2pp. The paired-bootstrap
interval included zero and IPSI did not beat slot-only. The preregistered gate
therefore failed.

## Why this is not evidence for a trainable IPSI method

The released mu-VLA MIKASA checkpoint was trained on five environments:
`ShellGamePush`, `InterceptMedium`, `RememberColor5`, `TakeItBack`, and
`RememberShapeAndColor3x3`. `ShellGameShuffleTouch` is a held-out task requiring
a different tracking-style memory structure. The released checkpoint is already
at the control floor on this task. A local latent edit cannot isolate a control
bottleneck when the base policy has essentially no task competence.

The correct conclusion is therefore narrower than "memory revision is false":

> the current ShuffleTouch setup is a bad causal test bed for a small memory
> intervention on this released checkpoint.

## Code audit finding

The historical local `MuVLAPolicy` is close to the released evaluator in action
normalization, recurrent stepping, camera splitting, and attention masking, but
it did not explicitly reproduce the upstream 224px resize plus 0.9 center crop
used for checkpoints trained with `image_aug=True`. This is a real evaluation
protocol mismatch even though the strong `RememberColor3` result shows that the
adapter is not globally broken.

New experiments therefore use `ProtocolMatchedMuVLAPolicy` and must pass a
three-task in-distribution parity check before any new scientific claim.

## New candidate gap

The released K=2 memory checkpoint shows large benefits on persistent/static
memory tasks but only a very small released gain over no-memory on the
in-distribution `InterceptMedium` training task. Intercept requires prediction
of a moving object's dynamics, not merely retention of a cue.

The new diagnostic asks whether generic recurrence has a **Storage–Dynamics
Gap**:

```text
current physical state is decodable from recurrent memory
but temporal dynamics / velocity are not.
```

This is still phenomenon-first. No temporal-memory method is authorized unless
the in-distribution probe supports the gap.

See `docs/experiments/track2_predictive_dynamics_gate2.md` for the frozen 16GB
protocol.
