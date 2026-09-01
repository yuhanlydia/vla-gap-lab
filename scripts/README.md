# Scripts

Executable setup/download/train/evaluate entrypoints are added per track. Scripts must write resolved configs, git SHAs, hardware metadata, logs, and checkpoints beneath `artifacts/`.
## Generate action-preserving LIBERO-Plus pairs

Run from the repository root. The explicit environment variables work around
upstream package discovery and PyTorch's changed trusted-checkpoint default:

```bash
PYTHONPATH=external/LIBERO-plus MUJOCO_GL=egl \
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python3 scripts/render_paired_libero_states.py \
  --task-id 1 --num-demos 10 --num-frames 8 \
  --output artifacts/pairs/libero_spatial_task_1.npz
```

The script deliberately accepts only Camera, Lighting, Background, and Sensor
Noise tasks. Robot initialization and layout perturbations do not preserve the
expert action at a fixed state and therefore cannot support the paired offline
probe assumption.

Then extract real layerwise OpenVLA action-token states (4-bit model load):

```bash
python3 scripts/extract_openvla_pair_hidden.py \
  --checkpoint models/openvla-7b-oft-combined \
  --pairs artifacts/pairs/libero_spatial_task_1.npz \
  --output artifacts/hidden/libero_spatial_task_1.npz
```

On minimal Ubuntu images, install `python3.10-dev` first. The current PyTorch
Triton backend compiles a small CUDA helper and requires `Python.h`.
