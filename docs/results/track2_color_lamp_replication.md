# Track 2 color-lamp tracking replication

Date: 2026-09-02

This is a 50-seed closed-loop replication on
`ShellGameShuffleColorLampTouch-VLA-v0`, using the released
`mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k2-tbptt` checkpoint and the same
seeds (`4242424242`–`4242424291`) as the primary Tracking pilot.

## Closed-loop controls

| condition | episodes | successes | SR | mean dense return |
|---|---:|---:|---:|---:|
| Normal recurrent memory | 50 | 0 | 0% | 2.377 |
| Freeze memory from cue end | 50 | 2 | 4% | 2.697 |
| Reset memory once at cue end | 50 | 0 | 0% | 2.356 |

Normal-minus-freeze is not significant: both successes are freeze-only,
McNemar exact two-sided `p=0.5`, with a paired bootstrap SR-difference interval
of `[0.00, 0.10]`. Reset has no successes in either condition.

## Why this is not the privileged identity intervention

The color-lamp task exposes `target_color` as the hidden label, but the target
color is only revealed by the lamp during the manipulation phase. It does not
provide a persistent target identity that is hidden through the shuffle. The
collector therefore records the label semantics explicitly as
`lamp_target_color_revealed_at_manipulation` and this task is treated as a
negative/control replication, not as evidence for an identity-preserving slot
edit.

The accompanying 50-episode memory cache is
`artifacts/mikasa/shell_shuffle_color_lamp_memory_strided_n50.npz`, with probe
report `artifacts/reports/shell_shuffle_color_lamp_memory_probe_n50_a100.json`.
During shuffle, target-color and target-slot probes are near three-way chance;
at manipulation target-color becomes decodable (balanced accuracy `0.756`)
while target-slot remains low (`0.365`). This is consistent with the task
revealing color late, rather than with a memory that preserves identity while
revising location.

## Decision

This replication does not authorize Conflict-Adaptive Refresh training. A
valid follow-up still requires a task exposing a hidden, identity-preserving
target label together with a privileged intervention that changes only its
simulator-derived slot estimate.
