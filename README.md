# VLA Gap Lab

Reproducible phenomenon-first experiments for failure hypotheses in existing VLA benchmarks:

1. **Latent-to-Action Utilization Gap** — LIBERO-Plus + OpenVLA-OFT (Gate not passed).
2. **VLA Memory Structure** — MIKASA-Robo-VLA + mu-VLA. Identity–Location Gate-1 failed; active diagnostic is the **Storage–Dynamics Gap** on an in-distribution predictive task.
3. **Cross-Embodiment State Transport** — RoboTwin 2.0 + X-VLA (paused until semantic-phase trajectories are available).

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
- [x] Track 2 Stage-0 ShellGamePush 0/20 isolated to a run with non-pinned YCB asset provenance
- [ ] Track 2 Stage-0: rerun **only ShellGamePush 20 seeds** with pinned ManiSkill-b15 YCB archive
- [ ] Track 2 Predictive-Dynamics Gate-2 on `InterceptMedium-VLA-v0` — blocked until the ShellGame replacement parity passes
- [x] Track 3 X-VLA reset-state diagnostic completed
- [ ] Track 3 semantic-phase portability probes on real paired RoboTwin trajectories

## Track 2 next run

Do **not** rerun InterceptMedium, RememberColor5, IPSI, or Gate-2 yet.

The September 8 parity run used the current Hugging Face YCB archive after
ManiSkill 3.0.0b15 rejected it against its historical checksum. ShellGamePush
is the parity task that directly uses YCB `025_mug`, so the next experiment
changes only this asset variable.

Follow:

```text
docs/results/track2_ycb_asset_audit_2026-09-08.md
docs/experiments/track2_predictive_dynamics_gate2.md
```

Run:

```bash
git pull origin main
git submodule update --init --recursive

cd external/MIKASA-Robo
uv sync --frozen
uv pip install -r ../../requirements/track2-extra.txt --python .venv/bin/python
PYTHONPATH=../../src uv run python ../../scripts/check_mu_vla_runtime.py
PYTHONPATH=../../src uv run python ../../scripts/install_track2_ycb_asset.py

PYTHONPATH=../../src uv run python ../../scripts/eval_mu_vla_protocol.py \
  --checkpoint ../../models/mu-vla-m64-k2 \
  --task ShellGamePush-VLA-v0 \
  --precision 4bit --episodes 20 --start-seed 4242424242 \
  --output ../../artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_4bit_ycbpinned_n20.json
```

Decision:

- ShellGamePush `SR >= 0.76` -> Stage-0 parity restored; proceed to the 40-episode InterceptMedium Gate-2 collection.
- ShellGamePush `< 0.76` -> keep Gate-2 blocked and compare official vs local ShellGamePush step-by-step.

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
- [Track 2 YCB asset audit and one-task replacement plan](docs/results/track2_ycb_asset_audit_2026-09-08.md)
- [Track 3 paired reset-state diagnostic](docs/results/track3_paired_reset.md)
