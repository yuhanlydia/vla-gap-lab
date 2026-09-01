# Track 3 paired reset-state diagnostic

This is an offline diagnostic, not a RoboTwin success-rate result and not yet the preregistered semantic-phase Gate-0.

## What ran

- Official RoboTwin 2.0 `blocks_ranking_rgb` environment.
- Aloha-AgileX and Franka-Panda, 16 seeds each (32 observations).
- Each pair uses the official task sampler with NumPy re-seeded exactly at `load_actors`, after embodiment-dependent robot construction.
- Every pair passed an `atol=1e-6` equality check on all three block positions.
- Each observation contains the official head, left-wrist, and right-wrist RGB cameras plus the official-client-compatible 20D end-effector proprioception.
- Official `X-VLA-0.9B` RoboTwin checkpoint, shared RoboTwin domain ID 6.
- Florence encoder layers 0/3/6/9/11 and action-transformer layers 0/4/8/12/16/20/23.

The machine has no `nvcc`, while upstream RoboTwin imports Curobo even for reset-only use. `patches/robotwin-reset-without-curobo.patch` makes the optional import lazy in effect and skips planner construction/gripper planning only when `need_plan=False`. It does not provide or emulate motion planning.

## Cross-validated transport result

Four-fold seed-disjoint evaluation gives chance retrieval of 25%. A ridge map is trained from Aloha features to paired Franka features on each training fold. The permutation baseline shuffles training correspondences ten times.

| stack | layer | mapped top-1 | raw top-1 | permuted-map top-1 |
|---|---:|---:|---:|---:|
| VLM | 0 | 25.0% | 43.8% | 25.0% |
| VLM | 11 | 18.8% | 31.2% | 25.6% |
| Control | 0 | 31.2% | 25.0% | 26.9% |
| Control | 4 | 31.2% | 37.5% | 27.5% |
| Control | 8 | **37.5%** | **50.0%** | 25.6% |
| Control | 12 | **37.5%** | **50.0%** | 26.9% |
| Control | 20 | 31.2% | 43.8% | 26.2% |
| Control | 23 | 31.2% | 37.5% | 30.6% |

The earlier single split showed 60% at control layers 4/8 and 20% at layers 20/23, but this did not survive cross-validation. The honest conclusion is therefore weak evidence of paired-state information around control layers 8/12, not a demonstrated mid-to-late shared-state collapse. Paired cosine is 0.92–1.00 but all retrieval margins remain negative, confirming cosine alone is misleading because these features are strongly anisotropic.

## Reproduction

```bash
git -C external/RoboTwin apply ../../patches/robotwin-reset-without-curobo.patch
PYTHONPATH=src python3 scripts/render_robotwin_cross_embodiment_pairs.py \
  --robotwin-root external/RoboTwin --seeds 16 \
  --output artifacts/robotwin/blocks_ranking_rgb_aloha_franka_reset_16.npz

TRANSFORMERS_NO_TF=1 PYTHONPATH=src .venv-xvla/bin/python \
  scripts/extract_xvla_robotwin_pairs.py \
  --checkpoint models/x-vla-robotwin2 \
  --pairs artifacts/robotwin/blocks_ranking_rgb_aloha_franka_reset_16.npz \
  --output artifacts/hidden/xvla_blocks_ranking_rgb_aloha_franka_reset_16.npz

PYTHONPATH=src python3 scripts/probe_xvla_pair_transport.py \
  --cache artifacts/hidden/xvla_blocks_ranking_rgb_aloha_franka_reset_16.npz \
  --null-permutations 10 \
  --output artifacts/results/xvla_pair_transport_reset_16_cv.json
```

## Next gate

Reset states vary object layout but do not supply task-phase labels. The formal Gate-0 still requires paired or phase-aligned trajectories and cross-embodiment phase accuracy above 80%. Next work should either generate official planned trajectories on a machine with Curobo available, or replay matched joint paths if compatible official paths can be obtained. No state-transport distillation should begin from this result alone.

## N=32 summary-pooling replication

The reset-pair sample was subsequently doubled to 32 seeds (64 observations).
Instead of token means alone, each layer uses concatenated token mean, standard
deviation, first token, and last token. A dual-form ridge implementation makes
the small-N/high-dimensional mapping and 500 correspondence permutations
tractable without materializing a feature-by-target coefficient matrix.

At N=16, the best summary-pooled result was control layer 8: 43.75% retrieval
against 25% fold chance, with an uncorrected 200-permutation p-value of 0.0547.
Its retrieval margin was still negative. On the N=32 replication:

| stack/layer | mapped top-1 | fold chance | null mean | null 95th percentile | p | margin |
|---|---:|---:|---:|---:|---:|---:|
| Control 8 | 18.75% | 12.5% | 12.3% | 18.75% | 0.136 | -0.0088 |

No tested layer exceeded its permutation 95th percentile. The best mapped
retrieval at N=32 was only 18.75%, shared by several control layers and VLM
layer 11. The earlier signal was therefore a small-sample fluctuation rather
than evidence for portable state. The Track 3 Gate remains closed.
