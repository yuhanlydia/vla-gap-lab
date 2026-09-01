# VLA Gap Lab

Reproducible Gate-0 experiments for three failure hypotheses in existing VLA benchmarks:

1. **Latent-to-Action Utilization Gap** — LIBERO-Plus + OpenVLA-OFT.
2. **Persistence-Revision Gap** — MIKASA-Robo-VLA + mu-VLA.
3. **Cross-Embodiment State Transport** — RoboTwin 2.0 + X-VLA.

The project deliberately does not introduce a new benchmark. Each track starts with a cheap diagnostic and only trains a minimal mechanism when its preregistered gate passes.

## Repository layout

```text
configs/                    Experiment manifests and Gate-0 thresholds
src/vla_gap_lab/            Shared extraction, probes, interventions, statistics
tracks/latent_action/       LIBERO-Plus diagnostic and ControlSkip
tracks/memory_revision/     mu-VLA interventions and adaptive refresh
tracks/state_transport/     X-VLA probes, transport, and distillation
scripts/                    Setup, download, smoke, train, and evaluation entrypoints
tests/                      Unit and integration tests with synthetic tensors
external/                   Pinned official repositories (git submodules)
artifacts/                  Local logs/checkpoints (gitignored)
data/                       Local datasets (gitignored)
```

## Execution policy

Each track follows the same progression:

```text
environment check -> data/model manifest -> smoke test -> Gate-0 diagnostic
    -> STOP if threshold fails
    -> minimal mechanism training if threshold passes -> official closed-loop SR
```

No result is reported as an official benchmark result unless it comes from the official simulator protocol. Offline metrics and synthetic smoke tests are labeled separately.

## Hardware note

The current development machine has a 16 GB RTX A4000. Configs therefore support hidden-state caching, batch size 1, CPU offload, and optional 4-bit model loading. The target reference setup remains a 24 GB GPU. A failed memory preflight is recorded as a hardware block rather than silently changing the scientific protocol.

## Status

- [x] Umbrella repository and official source pins
- [x] Track 1: paired LIBERO-Plus rendering, extraction adapter, ridge probes, and causal intervention
- [ ] Track 1: ControlSkip and LIBERO-Plus evaluation
- [x] Track 2: real MIKASA simulator + mu-VLA memory/action/proprio smoke path
- [x] Track 2: memory interventions + inertia metrics
- [ ] Track 2: conflict-adaptive refresh
- [x] Track 3: real X-VLA checkpoint load, action forward, and layer capture
- [ ] Track 3: layerwise portability probes on paired RoboTwin trajectories
- [ ] Track 3: hidden-state transport and distillation

Validated Track 1 model-load path on the development machine:

```bash
pip install -r requirements/track1-inference.txt
pip install -e external/openvla-oft --no-deps
python3 scripts/smoke_openvla_load.py \
  --checkpoint models/openvla-7b-oft-combined \
  --output artifacts/smoke/openvla_load.json
```

Generate a small paired clean/shift cache from the official simulator and
original demonstration states:

```bash
pip install -r requirements/track1-simulator.txt
PYTHONPATH=external/LIBERO-plus MUJOCO_GL=egl \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python3 scripts/render_paired_libero_states.py --task-id 1 --num-demos 10 --num-frames 8 \
  --output artifacts/pairs/libero_spatial_task_1.npz
```

## Sources

- [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- [MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo)
- [mu-VLA checkpoint](https://huggingface.co/mu-vla/mu-vla-openvla-oft-mikasa-robo-5-tasks-m64-k2-tbptt)
- [RoboTwin 2.0](https://github.com/HashimHS/robotwin)
- [X-VLA](https://github.com/2toinf/X-VLA)

## Results log

- [Track 1 smoke-scale preliminary diagnostic](docs/results/track1_preliminary.md)
- [Track 2 static-memory intervention smoke](docs/results/track2_static_smoke.md)
