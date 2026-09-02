# Track 2 horizon pilot — Short task

The primary Short-task horizon pilot is complete on
`ShellGameShuffleTouch-VLA-v0`, using the official mu-VLA K=2 and K=8
checkpoints, identical seeds `4242474242..4242474271`, and normal (unmodified)
memory. The raw JSON artifacts remain local under `artifacts/mikasa/` because
large experiment outputs are git-ignored.

| checkpoint | episodes | successes | success rate |
| --- | ---: | ---: | ---: |
| K=2 | 30 | 0 | 0.000 |
| K=8 | 30 | 1 | 0.033 |

Paired difference (K=8 minus K=2) is **+3.33 percentage points**. A paired
bootstrap with 20,000 resamples (seed 42) gives a 95% interval of
`[0.00pp, 10.00pp]`; the exact McNemar test has `p=1.0` (0 K2-only wins,
1 K8-only win). The frozen `+20pp` stop rule therefore does **not** fire, so
the runbook permits proceeding to the editor stage after the Long stress
replication completes.

Raw artifacts:

- `ShellGameShuffleTouch-VLA-v0_k2_normal_n30.json`
- `ShellGameShuffleTouch-VLA-v0_k8_normal_n30.json`
