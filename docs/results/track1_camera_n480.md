# Track 1 Camera representation replication (N=480)

This expands the earlier 80-state diagnostic to 480 paired states sampled from
30 official LIBERO Spatial demonstration trajectories (16 states each). Splits
hold out whole trajectories. The clean and Camera-variant images are both
re-rendered from the identical simulator state in one process.

## Render audit

- Camera variant: task index 609, difficulty 1
- Mean clean/shifted agent-view pixel MAE: **38.676 / 255**
- Mean official-demonstration/base-rerender MAE: **4.912 / 255**
- Hidden cache shape: **480 × 8 layers × 4096 dimensions**
- Captured layers: 4, 8, 12, 16, 20, 24, 28, 32
- Pooling: mean over the 56 OpenVLA-OFT action-token positions

The second number audits renderer drift. It is substantially smaller than the
intended camera perturbation, and neither side of the probe uses the stored
demonstration image.

## Ridge and split sensitivity

The sweep uses ridge alpha `{0.01, 0.1, 1, 10, 100, 1000}` and trajectory split
seeds `{7, 21, 42, 84}`. For every run, the table below reports the layer with
the best non-negative shifted R².

| Alpha | Best retention range across seeds |
|---:|---:|
| 0.01 | no valid layer to 0.187 |
| 0.1 | no valid layer to 0.189 |
| 1 | no valid layer to 0.210 |
| 10 | 0.470–0.502 |
| 100 | 0.611–0.680 |
| 1000 | 0.678–0.736 |

Across all 24 configurations, 17 had any layer with non-negative shifted R².
The maximum representation retention was **0.736**, and the median of the 17
valid per-run maxima was **0.502**. At the preregistered alpha 1 / seed 42,
clean R² reached 0.509 at layer 8 while every shifted R² was negative.

## Interpretation and Gate decision

**Gate not passed.** No hyperparameter/split combination reached the
preregistered representation-retention threshold of 0.8. The separately run
instruction-matched closed-loop pilot also showed no control loss (9/10 clean,
10/10 shifted). This task therefore exhibits neither required half of the
proposed Latent-to-Action Utilization Gap.

The sensitivity to ridge alpha is itself informative: a retention ratio is not
interpretable when either R² is negative. The Gate implementation now marks
such layers ineligible instead of allowing a ratio of two negative values to
look robust.

These are still single-task diagnostics under 4-bit inference and a
noncanonical Transformers version, not an official category-level result.
