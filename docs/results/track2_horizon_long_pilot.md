# Track 2 horizon pilot — Long task

The Long-task horizon replication is complete on
`ShellGameShuffleTouch-Long-VLA-v0`, using the official mu-VLA K=2 and K=8
checkpoints, identical seeds `4242474242..4242474271`, and normal (unmodified)
memory. The raw JSON artifacts remain local under `artifacts/mikasa/` because
large experiment outputs are git-ignored.

| checkpoint | episodes | successes | success rate |
| --- | ---: | ---: | ---: |
| K=2 | 30 | 2 | 0.067 |
| K=8 | 30 | 1 | 0.033 |

Paired difference (K=8 minus K=2) is **-3.33 percentage points**. A paired
bootstrap with 20,000 resamples (seed 42) gives a 95% interval of
`[-10.00pp, 0.00pp]`; the exact McNemar test has `p=1.0` (1 K2-only win,
0 K8-only wins). The frozen `+20pp` stop rule therefore does **not** fire.

Raw artifacts:

- `ShellGameShuffleTouch-Long-VLA-v0_k2_normal_n30.json`
- `ShellGameShuffleTouch-Long-VLA-v0_k8_normal_n30.json`
