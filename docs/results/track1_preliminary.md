# Track 1 preliminary diagnostic (not a Gate-0 result)

Date: 2026-09-01

This smoke-scale run uses 80 states from 10 held-out-by-episode LIBERO Spatial
demonstrations for one variant in each action-preserving visual category. It is
far below the preregistered minimum of 2,000 states and is **not** an official
Gate-0 result.

The OpenVLA-OFT combined checkpoint was loaded in NF4. We mean-pooled its 56
action-token states at layers `[4, 8, 12, 16, 20, 24, 28, 32]`, trained clean
ridge probes with the preregistered `alpha=1`, and evaluated the same probes on
paired shifted renders of identical simulator states.

| Shift | Best layer | Clean R² | Shift R² | Representation retention | Published control retention* |
|---|---:|---:|---:|---:|---:|
| Camera | 20 | 0.324 | 0.233 | 0.717 | 0.581 |
| Light | 4 | 0.459 | 0.429 | 0.934 | 0.913 |
| Background | 4 | 0.459 | 0.333 | 0.725 | 0.961 |
| Sensor Noise | 32 | 0.388 | 0.423 | 1.090 | 0.781 |

\* Control retention uses the official LIBERO-Plus OpenVLA-OFT category SR
divided by its published clean average (97.1). It is contextual only: our local
closed-loop protocol has not yet reproduced those numbers.

At this scale no category satisfies both preregistered conditions
simultaneously (`representation > 0.8`, `control < 0.7`). Camera has the needed
control failure but insufficient representation retention; Light and Noise
retain probe information but do not have a sufficiently large published
control collapse. The correct current decision is therefore **inconclusive / do
not train ControlSkip**, not a positive claim.

Artifacts are gitignored and reproducible with:

```bash
python3 scripts/render_paired_libero_states.py --task-id TASK_ID \
  --num-demos 10 --num-frames 8 --output artifacts/pairs/PAIR.npz
python3 scripts/extract_openvla_pair_hidden.py \
  --checkpoint models/openvla-7b-oft-combined --pairs artifacts/pairs/PAIR.npz \
  --output artifacts/hidden/HIDDEN.npz
python3 scripts/run_latent_action_gate.py --cache artifacts/hidden/HIDDEN.npz \
  --alpha 1 --output artifacts/reports/REPORT.json
```
