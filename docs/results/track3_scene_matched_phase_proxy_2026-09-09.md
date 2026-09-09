# Track 3 scene-matched phase-proxy diagnostic

This is an exploratory diagnostic, not the preregistered semantic-phase Gate-0
and not evidence that cross-embodiment state transport is solved.

## What ran

- Official X-VLA RoboTwin 2.0 checkpoint on `blocks_ranking_rgb`.
- Aloha-AgileX source paired with Franka-Panda and ARX-X5 targets.
- 50 downloaded demonstrations per embodiment; 40 exact pairs per target after
  matching the nested `scene_info.info` signature.
- Six normalized-progress samples per episode, with four episode-level held-out
  folds. These ordinal labels are a phase proxy, not semantic task-phase ground
  truth.
- VLM layers 0/3/6/9/11 and action/control layers 0/4/8/12/16/20/23.

The target-archive downloader needed to recognize the official clean target
layout (`<embodiment>_clean_50/{data,instructions,...}`) and infer embodiment
from the archive name. The reproducible submodule patch is
`patches/robotwin-target-archive-layout.patch`.

## Exploratory transport result

The scene-matched transport probe is positive at the best VLM layer, relative
to its correspondence permutation null. This is useful evidence that the
paired data contain transferable scene information, but it does not establish
semantic phase portability.

| target | best VLM layer | mapped top-1 | null 95th percentile | margin | Holm p across layers |
|---|---:|---:|---:|---:|---:|
| Franka | 6 | 62.1% | 3.3% | +0.0070 | 0.024 |
| ARX-X5 | 6 | 67.9% | 3.3% | +0.0103 | 0.024 |

## Phase-proxy diagnostic and gate decision

| target | best VLM layer | cross-phase accuracy | embodiment accuracy | shuffled-label max | formal Gate-0 |
|---|---:|---:|---:|---:|---:|
| Franka | 6 | 85.4% | 100.0% | 22.5% | **closed** |
| ARX-X5 | 0 | 96.7% | 98.1% | 25.0% | **closed** |

The preregistered gate requires high phase accuracy together with materially
lower embodiment accuracy than the late-layer/control comparison. Neither
target satisfies both conditions. The apparent phase signal is therefore
confounded by embodiment information under this proxy. No state-transport
distillation or causal intervention is authorized from this result.

## Reproduction

The raw downloaded demonstrations, checkpoint, and feature caches are local
ignored artifacts. The tracked machine-readable reports are:

- `results/track3_scene_matched_transport_2026-09-09_franka.json`
- `results/track3_scene_matched_transport_2026-09-09_arx_x5.json`
- `results/track3_scene_matched_phase_proxy_2026-09-09_franka.json`
- `results/track3_scene_matched_phase_proxy_2026-09-09_arx_x5.json`
- `results/track3_scene_matched_phase_shuffle_2026-09-09_franka.json`
- `results/track3_scene_matched_phase_shuffle_2026-09-09_arx_x5.json`

Apply the RoboTwin archive-layout patch before using the target archive:

```bash
git -C external/RoboTwin apply --unidiff-zero ../../patches/robotwin-target-archive-layout.patch
```
