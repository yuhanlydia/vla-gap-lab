# Track 1 causal-utilization diagnostic

Date: 2026-09-01

This is an offline causal diagnostic on 4-bit OpenVLA-OFT, not a LIBERO-Plus closed-loop result.

## Intervention

For the Camera-shift cache, a ridge probe is fitted on clean states from seven demonstrations. Three entire demonstrations are held out. At hidden-state layer 20, each action coordinate's normalized probe coefficient defines a direction. The perturbation is added to all 56 action tokens after the matching transformer block, at ±1 and ±2 times the training feature standard deviation along that direction. The official released L1 regression head then predicts the 8x7 action chunk.

The denominator is the norm of the ridge probe's predicted current-action change. The numerator is the norm of the official head's first-action change. Both are in unnormalized LIBERO action units:

`CUR = ||delta official action[0]|| / ||delta ridge action||`.

## Result

Across six shifted states balanced over three held-out demonstrations and all seven action-coordinate directions:

- Mean CUR: **0.0229**.
- Median CUR: **0.0188**.
- Per-coordinate mean CUR range: **0.0107–0.0409**.
- Mean official control gain per unit hidden perturbation: **0.00463**.

For action coordinate 0, expanding to all 24 held-out states gives mean CUR **0.0140**. Demonstration-level means are 0.0151, 0.0133, and 0.0137; a three-group bootstrap interval is [0.0133, 0.0151]. At the final normalized hidden state (layer 32), the same coordinate gives CUR **0.0314** on six samples. The final-layer intervention is attached after the final RMSNorm so it matches the cached layer-32 coordinate system.

A random hidden direction gives a much larger numerical CUR (~0.45), but this is not a valid positive control: its probe-predicted denominator is near zero by construction. Its control sensitivity per hidden-unit perturbation (0.00644) is only modestly above the layer-20 probe-direction value (0.00422). We therefore do not use random-direction CUR to support the claim.

## Interpretation

The causal effect is nonzero and approximately linear across ±1/±2 standard deviations, but only about 1–4% of the probe-predicted action change reaches the official first action. This is direct evidence that action decodability and action-head sensitivity are not equivalent in this smoke setting.

It is not yet sufficient to train ControlSkip. The separate representation-retention Gate-0 used only 80 states and Camera retention was 0.717, below the preregistered 0.8 threshold. The correct next step is a larger, multi-task causal replication and local closed-loop control retention—not method training based on this result alone.

## Large-angle Camera clean/shift control

Task 611 supplies the requested failure case (9/10 clean versus 1/10 shifted),
but its 480-state representation retention reaches only 0.608. At layer 4,
12 held-out trajectories × 7 action coordinates × 4 intervention magnitudes
give:

| Condition | Mean CUR | Median CUR |
|---|---:|---:|
| Clean | 0.02097 | 0.01826 |
| Camera shift | 0.01806 | 0.01563 |

The paired shifted-minus-clean mean is -0.00291 with a state-cluster bootstrap
95% interval of [-0.00519, -0.00073]. The shift modestly reduces utilization,
but absolute CUR is already very low in the clean condition where closed-loop
success is high. Low CUR is therefore not, by itself, a failure-specific
indicator. Together with the failed representation-retention Gate, this result
does not justify ControlSkip training.

## Reproduction

```bash
TRANSFORMERS_NO_TF=1 PYTHONPATH=src .venv-xvla/bin/python \
  scripts/run_openvla_causal_utilization.py \
  --checkpoint models/openvla-7b-oft-combined \
  --pairs artifacts/pairs/libero_spatial_camera_n80.npz \
  --hidden artifacts/hidden/libero_spatial_camera_n80.npz \
  --layer 20 --action-index 0 1 2 3 4 5 6 --max-test-samples 6 \
  --output artifacts/reports/libero_spatial_camera_layer20_cur_all_actions.json
```
