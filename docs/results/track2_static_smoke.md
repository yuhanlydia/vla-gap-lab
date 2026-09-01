# Track 2 static-memory intervention smoke

Date: 2026-09-01

This is an implementation check, not a benchmark result. The released mu-VLA
M64/K2 checkpoint was evaluated closed-loop on `RememberColor3-VLA-v0` with
receding horizon (one action per model call) and memory reset at every episode.

| Intervention | Episodes | Success rate | Mean memory inertia |
|---|---:|---:|---:|
| Normal | 1 | 1.000 | 0.890 |
| Freeze after step 5 | 3 | 0.667 | 0.932 |
| Oracle refresh at step 5 | 3 | 0.000 | 0.859 |

The seeds start at the checkpoint's published evaluation seed `4242424242`.
The sample is deliberately too small for claims. It nevertheless validates
that interventions alter the actual recurrent state and gives the expected
negative control: discarding old memory after the visual cue destroys static
recall. The reset-refresh diagnostic must therefore be evaluated only at a
predefined revision event where current evidence can identify the revised
state; applying it indiscriminately is not a valid diagnostic.
