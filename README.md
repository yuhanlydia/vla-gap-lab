# VLA Gap Lab

Reproducible phenomenon-first experiments for failure hypotheses in existing VLA benchmarks:

1. **Latent-to-Action Utilization Gap** — LIBERO-Plus + OpenVLA-OFT (Gate not passed).
2. **VLA Memory Structure** — MIKASA-Robo-VLA + mu-VLA. Identity–Location Gate-1 failed; Predictive-Dynamics Gate-2 supports a **Storage–Dynamics Gap** on an in-distribution task.
3. **Cross-Embodiment State Transport** — RoboTwin 2.0 + X-VLA (exploratory scene-matched proxy completed; formal gate remains closed pending semantic-phase trajectories).

The project deliberately does not introduce a new benchmark. Each track starts with a cheap diagnostic and only trains a minimal mechanism when its preregistered gate passes.

## Execution policy

```text
environment + asset check -> model/protocol parity -> cheap diagnostic
    -> STOP if threshold fails
    -> causal intervention
    -> STOP again if causal rescue fails
    -> only then train a minimal mechanism
```

## Hardware policy

The primary development target is a **16 GB RTX A4000-class GPU**; 24 GB is the optional confirmation tier.

For mu-VLA:

- primary inference is NF4 4-bit;
- recurrent closed-loop evaluation remains `num_envs=1`;
- long collections are crash-safe and store one episode at a time;
- cached PCA/probe work is offline after rollout;
- BF16 is a confirmation tier, not the primary protocol.

## Current status

- [x] Track 1 original Latent-to-Action hypothesis not supported; no ControlSkip training authorized
- [x] Track 2 real MIKASA simulator + released mu-VLA checkpoint path
- [x] Track 2 Identity–Location IPSI Gate-1 completed and failed
- [x] Track 2 corrected memory-aware Transformers runtime + action clipping
- [x] Track 2 Stage-0 parity restored: ShellGamePush 20/20, InterceptMedium 9/20, RememberColor5 18/20
- [x] Track 2 Predictive-Dynamics Gate-2: position R² 0.765, velocity R² 0.169
- [x] Track 2 minimal causal temporal operator: no control rescue; larger memory training stopped
- [x] Track 2 ICASSP 2027 motion-compression v2 run completed: same-backbone current/two-frame visual controls vs recurrent memory, Medium/Fast, MLP/Ridge, targeted ablations
- [ ] Track 2 ICASSP 2027 corrected-v3 evaluation (v2 is provisional after the September 16 audit below)
- [x] Track 3 X-VLA reset-state diagnostic completed
- [x] Track 3 scene-matched normalized-progress phase-proxy diagnostic completed (exploratory; formal Gate-0 closed)
- [ ] Track 3 semantic-phase portability probes on real paired RoboTwin trajectories

## Track 2 ICASSP 2027 run

The stopped method line is now tested as a narrow four-page diagnostic paper: **does recurrent compression preserve motion information that is already available in short visual history?** This is not a claim that VLA systems generally cannot understand motion, and no new VLA architecture is trained.

The frozen experiment uses the same μVLA visual projector for all visual controls. It compares current-step visual tokens, two-frame visual tokens, and the post-update recurrent memory on 60 `InterceptMedium` and 60 `InterceptFast` episodes. Five whole-episode splits use Ridge and a fixed 2-layer MLP. Ablations cover full64-vs-stride8 memory tokens, pre-contact-vs-all-step rows, and two-frame lag1-vs-lag2.

Run everything from the repository root:

```bash
git pull origin main
git submodule update --init --recursive
bash scripts/run_track2_icassp.sh
```

The exact protocol and kill rule are in:

```text
docs/experiments/track2_icassp_state_motion.md
configs/memory_revision/icassp_state_motion.yaml
```

The runner writes a commit-friendly result summary to:

```text
results/track2_icassp_motion_compression_summary.json
```

The compression-loss claim is allowed only if two-frame visual features first make task-relevant velocity strongly accessible, recurrent memory remains clearly worse on velocity while preserving current position, and the ordering replicates on `InterceptFast`. Otherwise the paper is stopped rather than expanded.

### September 16 evaluation audit: v2 is provisional

The 60-episode Medium and 60-episode Fast v2 run completed, but its formal
`1/6` decision must not be interpreted as a clean rejection of recurrent
memory. A code and data audit found two evaluation defects and one genuine
remaining scientific risk:

1. **The formal gate and the reported headline numbers used different
   probes.** The six-condition decision in
   `scripts/summarize_icassp_state_motion.py` uses the MLP aggregates, whereas
   the cited Fast recurrent-memory values (position R² `0.067`, velocity-y R²
   `-0.161`) are Ridge aggregates. The v2 explanation mixed those two result
   families.
2. **The frozen MLP is not a reliable gate.** It standardizes probe inputs but
   not the two-dimensional position or velocity targets. Some fits reached the
   500-iteration limit without convergence, and all primary MLP cells had
   strongly negative R². In a diagnostic rerun of Fast two-frame split 0,
   target standardization changed position R² from `[-5.61, 0.01]` to
   `[-0.45, 0.47]` and velocity R² from `[-8.00, -0.24]` to
   `[-3.05, 0.01]`. This does not establish a positive result; it establishes
   that the original MLP-based kill rule is invalid.
3. **`pre_contact` is not a physical-contact mask.** The code retains every
   row with `reached_status < 0.5`, but the environment sets that status from
   TCP distance to an intended hit pose, not from robot-ball contact. Thus
   collisions, rebounds, and out-of-view motion can remain in the nominal
   pre-contact set. In Fast, at least 517 of 3,218 retained rows across 26 of
   60 episodes show reversed/large lateral velocity or out-of-range position;
   observed extremes include velocity-y `-2.12` and position-y `-2.14`.

The collection alignment itself passed the audit: the environment launches
the ball primarily along the y axis, stored position differences correlate
with stored velocity at approximately `0.99`, and the inferred simulator
interval is approximately `0.05` seconds. No velocity-axis swap or one-step
label offset was found.

The contaminated Fast tail materially affects position decoding. On the same
Fast split-0 two-frame Ridge diagnostic, limiting evaluation to steps 0--30
changed per-axis position R² from `[-0.62, 0.37]` to `[0.72, 0.77]`. Velocity-y
remained weak at approximately `0.01`, so the audit does **not** rescue the
motion-preservation hypothesis. It shows that Fast position failure is largely
an evaluation-window/mask problem while velocity accessibility remains the
main unresolved scientific issue.

The original v2 artifacts are retained for auditability. A corrected v3 will
be versioned separately, will standardize and invert-transform MLP targets,
record convergence diagnostics, use Ridge as the primary gate and a converged
MLP only as a consistency check, and replace `reached_status` with a valid
contact/visibility definition. Its mask, window, thresholds, and analysis will
be fixed before looking at corrected aggregate results; positive and negative
outcomes will both be reported.

## Track 2 completed diagnostic

Stage-0 parity passes. The same-seed pinned-asset ShellGamePush replacement scored 20/20 after matching the official evaluator's per-step render call. The old and new archives contain byte-identical `025_mug` files, so YCB provenance was a real protocol mismatch but was not the cause of the 0/20 result.

Gate-2 completed on 40 InterceptMedium episodes. The policy scored 22/40 (55%). On leakage-safe held-out episodes, `memory_after` decoded position at R² 0.765 but velocity at only R² 0.169; `memory_delta` velocity R² was 0.082. The minimal causal temporal operator then failed: memory-injected velocity was 18/40 versus 19/40 normal, and oracle velocity was also 19/40. Larger memory training remains unauthorized.

## Reproducibility

Run first-party checks with:

```bash
scripts/check.sh
```

Before a long run, capture exact revisions and artifact hashes:

```bash
PYTHONPATH=src python3 scripts/capture_run_provenance.py \
  --repository . --repository external/MIKASA-Robo \
  --artifact results/gate0_summary.yaml \
  --output artifacts/reports/run_provenance.json
```

## Sources

- [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- [MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo)
- [mu-VLA](https://github.com/CognitiveAISystems/muVLA)
- [mu-VLA K=2 checkpoint](https://huggingface.co/mu-vla/mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k2-tbptt)
- [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)
- [X-VLA](https://github.com/2toinf/X-VLA)

## Results log

- [Gate-0 decision matrix](docs/results/gate0_decision_matrix.md)
- [Machine-readable Gate-0 summary](results/gate0_summary.yaml)
- [Track 1 large-angle Camera failure](docs/results/track1_camera611_n480.md)
- [Track 2 failed Identity–Location Gate-1](docs/results/track2_identity_location_gate1.md)
- [Track 2 runtime root-cause audit](docs/results/track2_runtime_root_cause.md)
- [Track 2 September 8 Stage-0 parity run](docs/results/track2_stage0_parity_2026-09-08.md)
- [Track 2 ShellGamePush root-cause audit](docs/results/track2_ycb_asset_audit_2026-09-08.md)
- [Track 2 Predictive-Dynamics Gate-2 result](docs/results/track2_predictive_dynamics_gate2_2026-09-08.md)
- [Track 2 causal temporal operator result](docs/results/track2_temporal_operator_causal_2026-09-08.md)
- [Track 3 paired reset-state diagnostic](docs/results/track3_paired_reset.md)
- [Track 3 scene-matched phase-proxy diagnostic](docs/results/track3_scene_matched_phase_proxy_2026-09-09.md)
