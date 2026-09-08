# Track 2 ShellGamePush root-cause audit — 2026-09-08

## Outcome

The same-seed NF4 replacement run scored **20/20 (100%)**, above the frozen
76% parity threshold. Stage-0 parity is restored when combined with
InterceptMedium 9/20 (45%) and RememberColor5 18/20 (90%).

The causal root cause was not the YCB archive revision. The local evaluator
omitted the official evaluator's per-step `env.render()` call. In this SAPIEN
GPU simulation path, that call flushes actor-pose updates needed by
ShellGamePush. Without it, the mugs fell through the table and every episode
failed. The shared rollout helper now performs `env.step(action)` followed by
`env.render()`.

## Evidence

The original corrected-runtime run produced a task-specific pattern:

| Task | Original observed | Released reference |
|---|---:|---:|
| `ShellGamePush-VLA-v0` | 0/20 = 0% | ~96% |
| `InterceptMedium-VLA-v0` | 9/20 = 45% | ~55% |
| `RememberColor5-VLA-v0` | 18/20 = 90% | ~94% |

ShellGamePush also remained 0/20 in BF16, excluding NF4 as the primary cause.

An official-vs-local same-seed trace then established:

1. Reset observations, proprioception, task state, and initial object poses
   matched exactly.
2. Model/config and policy preprocessing matched the official source.
3. The official rollout rendered after every simulator step; the local rollout
   did not.
4. A zero-action A/B test isolated this difference. Without rendering, mug Z
   fell from `-0.064148` at step 3 to `-0.750854` at step 8. With per-step
   rendering, mug Z stabilized at `0.052946` from step 4 onward.
5. After adding the official synchronization behavior, a one-seed NF4 test
   passed, followed by the formal 20-seed run at 20/20.

Formal artifact:

```text
artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_4bit_ycbpinned_n20.json
```

Its SHA-256 is
`a70d2a9577854bdac70f99d11bc93fe1a0042ebf160164bd1271e8d22e1354b4`.
The episode-level compact record is tracked at
`results/track2_stage0_parity_fixed_2026_09_08.json`.

It records the 20 seeds beginning at `4242424242`, NF4 precision, success rate
1.0, and the pinned archive SHA-256
`174001ba1003cc0c5adda6453f4433f55ec7e804f0f0da22d015d525d02262fb`.

## YCB hypothesis result

The archive mismatch was real and remains worth guarding for reproducibility:

- ManiSkill-b15 archive: `174001ba1003cc0c5adda6453f4433f55ec7e804f0f0da22d015d525d02262fb`
- mutable HF-main archive: `1551724fd1ac7bad9807ebcf46dd4a788caed5c9499c1225b9bfa080ffbefcb3`
- pinned revision: `4c134e2e33b11b1fa751c1d7b6afb2e89231b875`

However, archive inspection showed that all seven files under `025_mug` are
byte-identical. The only changed archive file was `info_pick_v0.json`, which
added metadata for four other objects. Therefore the archive difference could
not explain ShellGamePush's mug physics and the asset-causality hypothesis is
rejected.

The pinned installer and provenance marker remain in place so future runs do
not depend on mutable dataset state.

## Decision

Stage-0 parity passes:

| Task | Valid observed SR | Reference SR | Absolute error | Result |
|---|---:|---:|---:|---|
| `ShellGamePush-VLA-v0` | 20/20 = 1.00 | 0.96 | 4pp | PASS |
| `InterceptMedium-VLA-v0` | 9/20 = 0.45 | 0.55 | 10pp | PASS |
| `RememberColor5-VLA-v0` | 18/20 = 0.90 | 0.94 | 4pp | PASS |

The subsequent 40-episode InterceptMedium Storage-Dynamics Gate-2 completed and
supported the gap; see
`docs/results/track2_predictive_dynamics_gate2_2026-09-08.md`. The original
ShellGamePush 0/20 runs remain invalid evaluator evidence and must not be used
as evidence against the research hypothesis.
