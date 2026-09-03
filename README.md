# VLA Gap Lab

Reproducible phenomenon-first experiments for failure hypotheses in existing VLA benchmarks:

1. **Latent-to-Action Utilization Gap** — LIBERO-Plus + OpenVLA-OFT (Gate not passed).
2. **VLA Memory Structure** — MIKASA-Robo-VLA + mu-VLA. Identity–Location Gate-1 failed; active diagnostic is the **Storage–Dynamics Gap** on an in-distribution predictive task.
3. **Cross-Embodiment State Transport** — RoboTwin 2.0 + X-VLA (paused until semantic-phase trajectories are available).

The project deliberately does not introduce a new benchmark. Each track starts with a cheap diagnostic and only trains a minimal mechanism when its preregistered gate passes.

## Repository layout

```text
configs/                    Experiment manifests and preregistered thresholds
src/vla_gap_lab/            Shared extraction, probes, interventions, statistics
tracks/latent_action/       LIBERO-Plus diagnostics
tracks/memory_revision/     mu-VLA memory diagnostics
tracks/state_transport/     X-VLA probes and transport diagnostics
scripts/                    Setup, collection, probes, and evaluation entrypoints
tests/                      Unit and CLI regression tests
external/                   Pinned official repositories (git submodules)
artifacts/                  Local logs/checkpoints (gitignored)
data/                       Local datasets (gitignored)
```

## Execution policy

```text
environment check -> model/protocol parity -> cheap diagnostic
    -> STOP if threshold fails
    -> causal intervention
    -> STOP again if causal rescue fails
    -> only then train a minimal mechanism
```

No result is reported as an official benchmark result unless it comes from the official simulator protocol. Offline probes, privileged interventions, and synthetic smoke tests are labeled separately.

Run first-party CPU checks without traversing pinned upstream submodules:

```bash
scripts/check.sh
```

Before tagging or handing off a snapshot:

```bash
scripts/release_check.sh
```

## Hardware policy

The primary development target is a **16 GB RTX A4000-class GPU**; 24 GB is the optional confirmation tier.

For mu-VLA:

- primary inference is NF4 4-bit;
- recurrent closed-loop evaluation remains `num_envs=1` because memory must advance exactly once per simulator tick and asynchronous vectorization can change the protocol;
- long collections are crash-safe and store one episode at a time;
- after state caching, high-dimensional PCA/probe work is batched (`IncrementalPCA` batch 256) so throughput is recovered where batching is scientifically safe;
- if a 4-bit released-checkpoint parity task fails, repeat only that task in BF16 on a 24 GB GPU before interpreting a scientific failure.

A failed memory preflight is recorded as a hardware/protocol block rather than silently changing the experiment.

## Current status

- [x] Track 1: paired LIBERO-Plus rendering, hidden extraction, ridge probes, causal utilization, and three-view Camera replication
- [x] Track 1: original Latent-to-Action hypothesis not supported; no ControlSkip training authorized
- [x] Track 2: real MIKASA simulator + released mu-VLA checkpoint path
- [x] Track 2: static memory controls and identity-vs-location probe
- [x] Track 2: K=2/K=8 ShuffleTouch horizon pilot
- [x] Track 2: Identity–Location IPSI Gate-1 completed and **failed**
- [ ] Track 2: released K=2 train-task protocol parity with training-matched preprocessing
- [ ] Track 2: Predictive-Dynamics Gate-2 on `InterceptMedium-VLA-v0`
- [x] Track 3: X-VLA load/action/layer capture and reset-state diagnostic
- [ ] Track 3: semantic-phase portability probes on real paired RoboTwin trajectories

## Track 2 next run

Do **not** rerun or tune IPSI. Follow:

```text
docs/experiments/track2_predictive_dynamics_gate2.md
```

The first required run is a 20-episode 4-bit parity check on each of:

```text
ShellGamePush-VLA-v0
InterceptMedium-VLA-v0
RememberColor5-VLA-v0
```

Only after parity passes may the 40-episode Intercept dynamics collection start.

## Reproducibility

Before a long run, record exact revisions and artifact hashes:

```bash
PYTHONPATH=src python3 scripts/capture_run_provenance.py \
  --repository . --repository external/MIKASA-Robo \
  --artifact results/gate0_summary.yaml \
  --output artifacts/reports/run_provenance.json
```

The report intentionally excludes environment variables and credentials.

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
- [Track 2 static-memory control](docs/results/track2_static_smoke.md)
- [Track 2 memory-content probe](docs/results/track2_memory_content_probe.md)
- [Track 2 Short horizon pilot](docs/results/track2_horizon_short_pilot.md)
- [Track 2 Long horizon pilot](docs/results/track2_horizon_long_pilot.md)
- [Track 2 failed Identity–Location Gate-1](docs/results/track2_identity_location_gate1.md)
- [Track 2 direction reassessment](docs/results/track2_direction_reassessment.md)
- [Track 3 paired reset-state diagnostic](docs/results/track3_paired_reset.md)
