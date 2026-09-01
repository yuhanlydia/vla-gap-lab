# Gate-0 decision matrix

This page is the human-readable handoff for the local pilots. The authoritative
machine-readable values and protocol deviations are in
`results/gate0_summary.yaml`. None of these runs is an official benchmark-wide
result.

| Track | Strongest positive evidence | Decisive counterevidence | Decision |
|---|---|---|---|
| Latent-to-Action | Three paired Camera views collapse from clean 9/10 to shifted 1/10, 0/10, 0/10 | Their best 480-state representation retentions are 0.608, 0.595, 0.585, all below 0.8 | Do not train ControlSkip; observed failure is representation robustness |
| Persistence–Revision | At 50 episodes, first-swap identity is 0.651 [0.531, 0.776] while current slot is 0.326 [0.240, 0.429] | Policy is 0/50 and no identity-preserving causal intervention has produced +10 pp SR | Do not train adaptive refresh; representation diagnostic passed, causal Gate did not |
| State Transport | A few layer/alpha choices exceed raw fold chance | Correct seed-42 alpha-100 test is nonsignificant; the minimum across 48 exploratory tests has Bonferroni p=1.0 and all margins are negative | Do not distill transport; semantic-phase data are still absent |

## What is established

1. OpenVLA-OFT has a reproducible, instruction-matched failure on three
   adjacent large-angle LIBERO-Plus Camera variants. The action signal itself
   degrades in the captured representation, so these cases do not demonstrate
   a latent-to-action utilization gap.
2. Released μVLA memory retains hidden target identity longer than it tracks
   the target's revised slot. Static `RememberColor3` also depends on continued
   updates: normal is 48/50 versus post-cue freeze 32/50 (paired p=0.000145).
   The n=10 destructive-reset control is 2/10 versus normal 10/10, confirming
   that clearing persistent history is harmful.
3. Reset-state X-VLA transport does not robustly exceed its correspondence-null
   distribution. Reset layouts are not semantic task phases, so this diagnostic
   cannot satisfy the formal cross-embodiment hypothesis.

## Next experiments, in order

1. For Track 2, design an identity-preserving privileged intervention that
   changes only the tracked slot estimate. Run it on at least two Tracking tasks
   with 50 paired seeds. Only a ≥10 pp success gain authorizes method training.
2. For Track 1, sample a different semantic base task and search for a shift
   with representation retention ≥0.8 and control retention <0.7. Do not select
   layers, ridge alpha, or shift severity on the same evaluation episodes.
3. For Track 3, acquire planned trajectories on a Curobo-capable machine and
   derive task-phase labels before any further transport optimization. Apply
   layer-family multiple-testing correction by default.

## Reproduction safeguards added

- Long MIKASA collection and intervention evaluation are atomic and resumable.
- Heavy CLI imports occur after argument parsing; all 22 entrypoints expose
  help in the base environment.
- Probe sweeps preserve failed/negative-R² runs and record invalid runs.
- Transport reports record cache, alpha, seed, and adjusted p-values.
- The result manifest distinguishes incomplete pilots from completed formal
  protocols and validates thresholds against configs.
- `scripts/release_check.sh` runs 61 tests, builds a wheel, installs it in an
  isolated environment, and verifies import from site-packages.
