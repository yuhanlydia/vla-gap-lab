# Track 2 ICASSP 2027 — Motion Preservation under Recurrent Compression

## Paper question

**Does recurrent compression preserve motion information that is available in short visual history?**

The revised paper does not claim that VLA systems generally fail to understand motion. The diagnostic is narrower: μVLA observes a temporal stream, but repeatedly compresses that stream into a fixed recurrent state. We test whether motion that is accessible from the model's own two-frame visual representation remains equally accessible after that recurrent compression.

## Why the previous state-vs-motion result is not enough

The earlier in-distribution probe found strong current-position accessibility and weaker velocity accessibility in μVLA memory. By itself, that result cannot distinguish two explanations:

1. velocity is simply hard to infer from the visual stream; or
2. temporal evidence contains velocity, but recurrent compression preferentially preserves current state.

The new control directly separates them using the **same frozen μVLA visual projector**.

## One-command run

```bash
git pull origin main
git submodule update --init --recursive
bash scripts/run_track2_icassp.sh
```

Optional checkpoint override:

```bash
bash scripts/run_track2_icassp.sh /absolute/path/to/mu-vla-m64-k2
```

## Frozen collection

Released μVLA K=2, NF4 4-bit, batch-1 recurrent rollout.

| Task | Episodes | Start seed |
|---|---:|---:|
| `InterceptMedium-VLA-v0` | 60 | 4243224242 |
| `InterceptFast-VLA-v0` | 60 | 4243324242 |

At each timestep the collector stores:

- all 64 post-update recurrent memory tokens;
- eight coarse visual tokens captured from the **same μVLA multimodal projector** used during action prediction (two camera views × a 2×2 spatial grid);
- ball position and velocity, goal, TCP, contact state, executed action, reward, and success.

The visual tokens are captured with a forward hook during the existing policy call, so no second encoder or external video model is introduced.

## Primary representations

All representations use train-only token PCA (hidden dimension → 32) and the same downstream probe budget.

1. **Current visual**: current-step visual tokens only. This is the single-time-step control.
2. **Two-frame visual**: concatenated frozen visual tokens from `t-1` and `t`. No learned temporal module is added.
3. **Recurrent memory**: all 64 post-update memory tokens at `t`, which summarize the full history seen by μVLA.

The primary comparison is velocity-y accessibility. Position XY is the retention control.

## Leakage-safe protocol

Five deterministic whole-episode splits: `0,1,2,3,4`, each with 36 train / 12 dev / 12 test episodes. Primary rows are step ≥2 and pre-contact. PCA is fit only on train episodes; Ridge alpha is selected only on dev; the MLP architecture is frozen in advance. Test episodes are untouched until final scoring.

## Primary claim gate

The paper may claim **motion loss under recurrent compression** only if all conditions hold on the median across five splits:

```text
Medium two-frame MLP velocity-y R2 >= 0.60
Medium (two-frame - current) velocity-y R2 >= 0.20
Medium (two-frame - recurrent-memory) velocity-y R2 >= 0.15
Medium recurrent-memory position R2 >= 0.65
Fast two-frame velocity-y R2 > recurrent-memory velocity-y R2
Fast recurrent-memory position R2 >= 0.60
```

If two-frame visual features do not recover velocity, temporal observability has not been established and the compression-loss claim is stopped. If recurrent memory matches the two-frame representation, there is no compression gap and the paper is stopped rather than rescued with more tasks.

## Frozen ablations

Only reviewer-facing ablations are authorized:

- full 64 memory tokens vs legacy stride-8 memory tokens;
- pre-contact rows vs all steps;
- two-frame lag 1 vs lag 2;
- Ridge vs fixed 2-layer MLP;
- Medium vs Fast dynamic regime.

The previously failed velocity-injection intervention is retained as a **limitation/supporting negative result** and is not tuned again.

## Outputs

```text
artifacts/mikasa/icassp_motion_compression_medium_n60/
artifacts/mikasa/icassp_motion_compression_fast_n60/
artifacts/reports/icassp_motion_compression/medium.json
artifacts/reports/icassp_motion_compression/fast.json
artifacts/reports/icassp_motion_compression/icassp_motion_compression_summary.json
artifacts/reports/icassp_motion_compression/icassp_motion_compression_primary_table.tex
artifacts/reports/icassp_motion_compression/icassp_motion_compression_ablation_table.tex
artifacts/reports/icassp_motion_compression/icassp_motion_compression_hero.pdf
artifacts/reports/icassp_motion_compression/icassp_motion_compression_memory_state_motion.pdf
results/track2_icassp_motion_compression_summary.json
```

After the run:

```bash
git add results/track2_icassp_motion_compression_summary.json
git commit -m "results: add ICASSP motion-compression study"
git push
```

## Explicitly not authorized

- no new VLA training;
- no world-model or optical-flow method;
- no post-hoc threshold changes;
- no extra benchmarks to rescue a failed primary claim;
- no tuning of the failed temporal operator.
