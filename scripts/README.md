# Scripts

Executable setup/download/train/evaluate entrypoints are added per track. Scripts must write resolved configs, git SHAs, hardware metadata, logs, and checkpoints beneath `artifacts/`.

## ICASSP 2027 state-motion study

The Track-2 method line is stopped. The ICASSP study is a narrow diagnostic asking whether recurrent μVLA memory makes instantaneous state more accessible than motion.

Run the complete frozen experiment from the repository root:

```bash
bash scripts/run_track2_icassp.sh
```

Optional checkpoint override:

```bash
bash scripts/run_track2_icassp.sh /absolute/path/to/mu-vla-m64-k2
```

The runner performs, in order:

1. pinned MIKASA environment sync and Track-2 dependency install;
2. exact μVLA runtime validation;
3. 60 full-memory episodes on `InterceptMedium-VLA-v0`;
4. 60 full-memory episodes on `InterceptFast-VLA-v0`;
5. five leakage-safe 36/12/12 episode splits per task;
6. full64 Ridge and MLP primary probes;
7. stride8 token-coverage and all-step contact-filter ablations;
8. combined JSON/CSV/LaTeX table and PDF/PNG figure generation;
9. tracked summary at `results/track2_icassp_state_motion_summary.json`.

Collections are `--resume` safe. Multi-GB episode caches remain under `artifacts/` and should not be committed. After a completed run, commit only the tracked summary so the result can be reviewed remotely.

Full protocol: `docs/experiments/track2_icassp_state_motion.md`.

## Generate action-preserving LIBERO-Plus pairs

Run from the repository root. The explicit environment variables work around upstream package discovery and PyTorch's changed trusted-checkpoint default:

```bash
PYTHONPATH=external/LIBERO-plus:src MUJOCO_GL=egl \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-openvla/bin/python scripts/render_paired_libero_states.py \
  --task-id 1 --num-demos 10 --num-frames 8 \
  --output artifacts/pairs/libero_spatial_task_1.npz
```

The script deliberately accepts only Camera, Lighting, Background, and Sensor Noise tasks. Robot initialization and layout perturbations do not preserve the expert action at a fixed state and therefore cannot support the paired offline probe assumption.

Then extract real layerwise OpenVLA action-token states (4-bit model load):

```bash
python3 scripts/extract_openvla_pair_hidden.py \
  --checkpoint models/openvla-7b-oft-combined \
  --pairs artifacts/pairs/libero_spatial_task_1.npz \
  --output artifacts/hidden/libero_spatial_task_1.npz
```

## MIKASA + mu-VLA memory smoke

MIKASA is kept in its own locked environment because its official stack pins Torch 2.2.1 and NumPy 1.23.5. Track 2 additionally pins the exact **memory-aware** Transformers fork used by the released mu-VLA checkpoint.

```bash
cd external/MIKASA-Robo
uv sync --frozen
uv pip install -r ../../requirements/track2-extra.txt --python .venv/bin/python
PYTHONPATH=../../src uv run python ../../scripts/check_mu_vla_runtime.py
PYTHONPATH=../../src uv run python ../../scripts/smoke_mu_vla_mikasa.py \
  --checkpoint ../../models/mu-vla-m64-k2 \
  --output ../../artifacts/smoke/mu_vla_mikasa.json
```

If this environment was created before the runtime fix, reinstall the Track-2 requirements before running any parity or ICASSP experiment.

## OpenVLA causal utilization

Use a separate system-site-packages environment. OpenVLA's Prismatic vision guard requires `timm<1`, while the X-VLA environment uses `timm==1.0.12`:

```bash
python3 -m venv --system-site-packages .venv-openvla
uv pip install --no-deps -r requirements/track1-causal.txt \
  --python .venv-openvla/bin/python
TRANSFORMERS_NO_TF=1 PYTHONPATH=src .venv-openvla/bin/python \
  scripts/run_openvla_causal_utilization.py --help
```

## X-VLA layer capture

Use an isolated environment with the host's CUDA-compatible Torch. Installing a fresh Torch currently selects CUDA 13 wheels, which are incompatible with the development machine's CUDA 12.8 driver:

```bash
python3 -m venv --system-site-packages .venv-xvla
uv pip install --no-deps -r requirements/track3-inference.txt \
  tokenizers==0.21.4 --python .venv-xvla/bin/python
TRANSFORMERS_NO_TF=1 .venv-xvla/bin/python scripts/smoke_xvla_layers.py \
  --checkpoint models/x-vla-robotwin2 \
  --output artifacts/smoke/xvla_layers.json
```
