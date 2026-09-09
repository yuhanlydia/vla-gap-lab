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
- cached PCA/probe work uses `IncrementalPCA(batch_size=256)`;
- BF16 is a confirmation tier, not the primary protocol.

## Current status

- [x] Track 1 original Latent-to-Action hypothesis not supported; no ControlSkip training authorized
- [x] Track 2 real MIKASA simulator + released mu-VLA checkpoint path
- [x] Track 2 Identity–Location IPSI Gate-1 completed and failed
- [x] Track 2 corrected memory-aware Transformers runtime + action clipping
- [x] Track 2 Stage-0: InterceptMedium 9/20 (45%) and RememberColor5 18/20 (90%) pass parity tolerance
- [x] Track 2 ShellGamePush root cause: local rollout omitted the official per-step render/simulator synchronization
- [x] Track 2 Stage-0 parity restored: ShellGamePush 20/20, InterceptMedium 9/20, RememberColor5 18/20
- [x] Track 2 Predictive-Dynamics Gate-2: position R² 0.765, velocity R² 0.169 — Storage–Dynamics Gap supported
- [x] Track 2 minimal causal temporal operator: no control rescue; larger memory training stopped
- [x] Track 3 X-VLA reset-state diagnostic completed
- [x] Track 3 scene-matched normalized-progress phase-proxy diagnostic completed (exploratory; formal Gate-0 closed)
- [ ] Track 3 semantic-phase portability probes on real paired RoboTwin trajectories

## Track 2 result

Stage-0 parity passes. The same-seed pinned-asset ShellGamePush replacement
scored 20/20 after matching the official evaluator's per-step render call. The
old and new archives contain byte-identical `025_mug` files, so YCB provenance
was a real protocol mismatch but was not the cause of the 0/20 result.

Follow:

```text
docs/results/track2_ycb_asset_audit_2026-09-08.md
docs/experiments/track2_predictive_dynamics_gate2.md
```

Gate-2 completed on 40 InterceptMedium episodes. The policy scored 22/40
(55%). On leakage-safe held-out episodes, `memory_after` decoded position at
R² 0.765 but velocity at only R² 0.169; `memory_delta` velocity R² was 0.082.
This meets the frozen Storage–Dynamics Gap criterion. The next authorized work
was the minimal causal temporal operator. It failed: memory-injected velocity
was 18/40 versus 19/40 normal, oracle velocity was also 19/40, and the paired
bootstrap lower bound was −20 pp. Larger memory training is therefore not
authorized by this run.

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
