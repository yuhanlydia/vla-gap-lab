# Scripts

Executable setup/download/train/evaluate entrypoints are added per track. Scripts must write resolved configs, git SHAs, hardware metadata, logs, and checkpoints beneath `artifacts/`.
## Generate action-preserving LIBERO-Plus pairs

Run from the repository root. The explicit environment variables work around
upstream package discovery and PyTorch's changed trusted-checkpoint default:

```bash
PYTHONPATH=external/LIBERO-plus:src MUJOCO_GL=egl \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-openvla/bin/python scripts/render_paired_libero_states.py \
  --task-id 1 --num-demos 10 --num-frames 8 \
  --output artifacts/pairs/libero_spatial_task_1.npz
```

The script deliberately accepts only Camera, Lighting, Background, and Sensor
Noise tasks. Robot initialization and layout perturbations do not preserve the
expert action at a fixed state and therefore cannot support the paired offline
probe assumption. Both conditions are re-rendered in the same process; stored
demonstration frames are retained only to quantify renderer drift.

Then extract real layerwise OpenVLA action-token states (4-bit model load):

```bash
python3 scripts/extract_openvla_pair_hidden.py \
  --checkpoint models/openvla-7b-oft-combined \
  --pairs artifacts/pairs/libero_spatial_task_1.npz \
  --output artifacts/hidden/libero_spatial_task_1.npz
```

On minimal Ubuntu images, install `python3.10-dev` first. The current PyTorch
Triton backend compiles a small CUDA helper and requires `Python.h`.

## MIKASA + mu-VLA memory smoke

MIKASA is kept in its own locked environment because its official stack pins
Torch 2.2.1 and NumPy 1.23.5:

```bash
cd external/MIKASA-Robo
uv sync --frozen
uv pip install -r ../../requirements/track2-extra.txt --python .venv/bin/python
uv pip install \
  'transformers @ git+https://github.com/moojink/transformers-openvla-oft.git' \
  --python .venv/bin/python
PYTHONPATH=../../src uv run python ../../scripts/smoke_mu_vla_mikasa.py \
  --checkpoint ../../models/mu-vla-m64-k2 \
  --output ../../artifacts/smoke/mu_vla_mikasa.json
```

The adapter supplies only the two tiny Prismatic modules imported by the
checkpoint's remote code. This avoids importing the unrelated TensorFlow/RLDS
training pipeline into simulator inference.

Run an explicitly reset, receding-horizon closed-loop episode:

```bash
PYTHONPATH=../../src uv run python ../../scripts/eval_mu_vla_intervention.py \
  --checkpoint ../../models/mu-vla-m64-k2 --task RememberColor3-VLA-v0 \
  --mode normal --episodes 1 --output ../../artifacts/mikasa/remember_color_normal.json
```

## OpenVLA causal utilization

Use a separate system-site-packages environment. OpenVLA's Prismatic vision
guard requires `timm<1`, while the X-VLA environment uses `timm==1.0.12`:

```bash
python3 -m venv --system-site-packages .venv-openvla
uv pip install --no-deps -r requirements/track1-causal.txt \
  --python .venv-openvla/bin/python
TRANSFORMERS_NO_TF=1 PYTHONPATH=src .venv-openvla/bin/python \
  scripts/run_openvla_causal_utilization.py --help
```

Run a paired closed-loop Camera task while holding the natural-language
instruction and initial states fixed. Do not omit `--clean-language`: upstream
variant task metadata includes perturbation bookkeeping in the language field.

```bash
PYTHONPATH=external/openvla-oft:external/LIBERO-plus:src MUJOCO_GL=egl \
TRANSFORMERS_NO_TF=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
.venv-openvla/bin/python scripts/eval_openvla_libero_tasks.py \
  --checkpoint models/openvla-7b-oft-combined --suite libero_spatial \
  --task-ids 609 --trials 10 \
  --clean-bddl pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl \
  --clean-language 'pick up the black bowl between the plate and the ramekin and place it on the plate' \
  --also-benchmark-variant \
  --output artifacts/libero_closed_loop/camera609_instruction_matched_n10.json

PYTHONPATH=src python3 scripts/analyze_paired_binary.py \
  artifacts/libero_closed_loop/camera609_instruction_matched_n10.json
```

Audit filename-derived perturbation metadata in LIBERO-Plus instructions:

```bash
PYTHONPATH=src python3 scripts/audit_libero_plus_language.py \
  --classification external/LIBERO-plus/libero/libero/benchmark/task_classification.json \
  --output artifacts/reports/libero_plus_language_audit.json
```

## X-VLA layer capture

Use an isolated environment with the host's CUDA-compatible Torch. Installing
a fresh Torch currently selects CUDA 13 wheels, which are incompatible with
the development machine's CUDA 12.8 driver:

```bash
python3 -m venv --system-site-packages .venv-xvla
uv pip install --no-deps -r requirements/track3-inference.txt \
  tokenizers==0.21.4 --python .venv-xvla/bin/python
TRANSFORMERS_NO_TF=1 .venv-xvla/bin/python scripts/smoke_xvla_layers.py \
  --checkpoint models/x-vla-robotwin2 \
  --output artifacts/smoke/xvla_layers.json
```

The smoke captures action-token states after seven of the 24 flow-transformer
blocks. This is the domain-conditioned control stack; Florence encoder layers
will be captured separately for the domain-agnostic task-state probe.
