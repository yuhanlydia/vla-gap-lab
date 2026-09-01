# Track 2 static-memory intervention smoke

Date: 2026-09-01

This is a 10-seed causal pilot, not a benchmark-wide result. The released mu-VLA
M64/K2 checkpoint was evaluated closed-loop on `RememberColor3-VLA-v0` with
receding horizon (one action per model call) and memory reset at every episode.

| Intervention | Success | Paired difference vs normal | Exact McNemar p |
|---|---:|---:|---:|
| Normal | 10/10 | — | — |
| Freeze after step 5 | 9/10 | -0.10 | 1.0 |
| Reset refresh at step 5 | 2/10 | -0.80 | 0.0078125 |

All conditions use the same seeds 4242424242–4242424251. The paired bootstrap
95% interval for reset-minus-normal SR is `[-1.00, -0.50]`; there are eight
normal-only successes and no reset-only successes. Freezing the post-cue state
barely changes success, whereas discarding it causes a large, paired loss.
This is direct evidence that persistent history is useful on static recall and
also confirms why a full reset is not a valid “oracle” for hidden-object
tracking: it destroys the identity that must be retained. The earlier 1/3-seed
implementation smoke is superseded by this run.
