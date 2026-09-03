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
Torch 2.2.1 and NumPy 1.23.5. Track 2 additionally pins the exact
**memory-aware** Transformers fork used by the released mu-VLA checkpoint.
Do not install `moojink/transformers-openvla-oft` in this environment: it is a
different fork and does not implement the released memory attention contract.

```bash
cd external/MIKASA-Robo
uv sync --frozen
uv pip install -r ../../requirements/track2-extra.txt --python .venv/bin/python
PYTHONPATH=../../src uv run python ../../scripts/check_mu_vla_runtime.py
PYTHONPATH=../../src uv run python ../../scripts/smoke_mu_vla_mikasa.py \
  --checkpoint ../../models/mu-vla-m64-k2 \
  --output ../../artifacts/smoke/mu_vla_mikasa.json
```

If this environment was created before the runtime fix, reinstall the Track-2
requirements before running any parity or Gate-2 experiment. The runtime check
fails if Transformers is not the exact
`CognitiveAISystems/transformers-mu-openvla-oft` commit used by the official
mu-VLA release.

The adapter supplies only the two tiny Prismatic modules imported by the
checkpoint's remote code. This avoids importing the unrelated TensorFlow/RLDS
training pipeline into simulator inference.

Run an explicitly reset, receding-horizon closed-loop episode:

```bash
PYTHONPATH=../../src uv run python ../../scripts/eval_mu_vla_intervention.py \
  --checkpoint ../../models/mu-vla-m64-k2 --task RememberColor3-VLA-v0 \
  --mode normal --episodes 10 --resume \
  --output ../../artifacts/mikasa/remember_color_normal.json
```

The evaluator atomically checkpoints after every completed episode. `--resume`
requires the checkpoint, task, mode, intervention timing, and starting seed to
match, then continues with the next unused seed.

Collect and probe dynamic ShellGame memory without erasing target identity:

```bash
cd external/MIKASA-Robo
PYTHONPATH=../../src uv run python ../../scripts/collect_mu_vla_memory_trajectory.py \
  --checkpoint ../../models/mu-vla-m64-k2 --episodes 24 --pooling summary \
  --checkpoint-every 5 --resume \
  --output ../../artifacts/mikasa/shell_shuffle_memory_summary_n24.npz
cd ../..
PYTHONPATH=src .venv-openvla/bin/python scripts/probe_mu_vla_memory_tracking.py \
  --trajectory artifacts/mikasa/shell_shuffle_memory_summary_n24.npz --alpha 100 \
  --output artifacts/reports/shell_shuffle_memory_summary_probe_n24_a100.json
```

Use `--pooling strided` during collection to retain every eighth memory token
instead of four summary statistics. The downstream probe flattens any retained
token axes and omits the redundant delta-from-reset probe when learned initial
memory is identical across episodes.

Collection checkpoints atomically every five completed episodes by default.
Re-running with `--resume` validates checkpoint/task/seed/pooling metadata and
continues from the next episode without repeating seeds. Lower
`--checkpoint-every` for more frequent recovery points.

The collector also accepts `ShellGameShuffleColorLampTouch-VLA-v0`; in that
task `target_color` is revealed by the lamp during manipulation, so metadata
marks it as a negative/control replication rather than a hidden
identity-preserving tracking task.

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

Use `--condition clean` and `--condition shifted` with identical intervention
settings, then compare the reports with state-cluster bootstrap uncertainty:

```bash
PYTHONPATH=src python3 scripts/compare_causal_utilization.py \
  artifacts/reports/causal_clean.json artifacts/reports/causal_shifted.json \
  --metric cur --output artifacts/reports/causal_clean_vs_shifted.json
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

For reset-pair transport, summary pooling retains token mean/std and endpoint
tokens. The probe uses an N×N dual ridge solve and reports an empirical
permutation p-value:

```bash
TRANSFORMERS_NO_TF=1 PYTHONPATH=src .venv-xvla/bin/python \
  scripts/extract_xvla_robotwin_pairs.py \
  --checkpoint models/x-vla-robotwin2 \
  --pairs artifacts/robotwin/blocks_ranking_rgb_aloha_franka_reset_32.npz \
  --pooling summary \
  --output artifacts/hidden/xvla_blocks_ranking_rgb_aloha_franka_reset_32_summary.npz

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
python3 scripts/probe_xvla_pair_transport.py \
  --cache artifacts/hidden/xvla_blocks_ranking_rgb_aloha_franka_reset_32_summary.npz \
  --alpha 100 --no-standardize --null-permutations 500 \
  --output artifacts/results/xvla_summary_n32_a100_raw_null500.json
```