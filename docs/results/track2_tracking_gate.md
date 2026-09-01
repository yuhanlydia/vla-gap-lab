# Track 2 dynamic-tracking Gate-0

Date: 2026-09-01

This is a closed-loop pilot on official MIKASA-Robo-VLA, not the canonical 50-episode task result.

## Setup

- Task: `ShellGameShuffleTouch-VLA-v0` (held out from checkpoint training).
- Checkpoint: released `mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k2-tbptt`.
- Official GPU simulator, wrapper stack, `pd_ee_delta_pose`, receding horizon, and seeds beginning at `4242424242`.
- Normal and cue-end freeze use the same 10 seeds.
- Cue-end refresh uses the first 3 seeds.
- The evaluator now reads each episode's sampled `cue_steps_per_env`, `shuffle_steps_per_env`, and `num_swaps_per_env`, so interventions occur at the true per-episode boundary rather than a guessed fixed step.

## Results

| condition | episodes | successes | SR | mean dense return |
|---|---:|---:|---:|---:|
| Normal recurrent memory | 10 | 0 | 0% | 1.862 |
| Freeze memory from cue end | 10 | 1 | 10% | 2.994 |
| Reset memory once at cue end | 3 | 0 | 0% | 1.690 |

Normal-minus-freeze paired return difference is -1.133 with a seed-bootstrap 95% interval of [-3.419, 0.451]. The lone discordant success is not significant (exact paired sign/McNemar evidence is uninformative at n=1 discordance). Thus neither SR nor dense return supports the claim that ordinary recurrent updates reliably revise state during shuffling.

For seed `4242424242` (3 cue steps, 28 shuffle steps, 4 swaps), normal memory candidate cosine averaged 0.947 during shuffle. Freezing the cue memory makes actual cosine exactly 1.0, while the updater's unused candidate cosine drops to 0.874 and its candidate update norm increases from 153.9 to 226.3. The updater responds to changing images, but those changes do not translate into reliable success.

Cosines are computed in float32. Earlier bfloat16 values slightly above 1.0 were a numerical artifact and are superseded.

## Gate decision

The preregistered continuation criterion was at least +10 percentage points from oracle refresh on Tracking. It did not pass: cue-end refresh is 0/3 and normal is 0/10. Conflict-Adaptive Refresh training must not start from this evidence.

There is also a conceptual issue with full refresh on this particular task: after the ball is hidden, the current frame does not identify the target cup. Clearing memory at cue end destroys target identity rather than providing a true oracle revision. A scientifically valid follow-up would preserve identity while revising location, for example with factorized/token-selective memory interventions or privileged simulator labels used only for diagnostics. It should not relabel full reset as an oracle.

## Reproducibility note

ManiSkill's pinned `ycb` downloader expected SHA-256 `174001...62fb`, while the current official Hugging Face archive returned `155172...cb3`. The current archive contained the expected `mani_skill2_ycb/info_pick_v0.json` and 470 asset files. It was manually extracted to ManiSkill's standard data directory without altering the benchmark code. The 25 MB temporary ZIP was deleted after extraction and is recoverable from the official URL.
