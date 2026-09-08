# Track 2 Minimal Causal Temporal Operator — 2026-09-08

## Decision

The Storage–Dynamics Gap probe replicated, but the preregistered minimal causal
rescue did **not** pass. No new memory method or training run is authorized from
this result.

The intervention was intentionally small and diagnostic, not a deployable method:
the velocity readout was fit using only the Gate-2 training episodes, mapped back
to the 128-dimensional PCA memory space, reshaped to the eight retained memory
tokens, and injected at a fixed 25% of the train-set median memory-update norm.
The evaluation used 40 new paired seeds per condition, with simulator
`render_after_step` synchronization.

## Results

| Operator family | Condition | Success | Rate |
|---|---|---:|---:|
| action correction | normal | 21/40 | 52.5% |
| action correction | predicted velocity | 8/40 | 20.0% |
| action correction | oracle velocity | 6/40 | 15.0% |
| action correction | sign sham | 9/40 | 22.5% |
| memory injection | normal | 19/40 | 47.5% |
| memory injection | predicted velocity | 18/40 | 45.0% |
| memory injection | oracle velocity | 19/40 | 47.5% |
| memory injection | orthogonal sham | 18/40 | 45.0% |

For the primary memory intervention, the paired success gain was **−2.5 pp**
with a paired-bootstrap 2.5th-percentile lower bound of **−20.0 pp**. The oracle
condition had **0.0 pp** gain, and the orthogonal sham had the same −2.5 pp gain
as the velocity direction. The frozen criterion required at least +10 pp, a
positive bootstrap lower bound, and improvement over the sham; it therefore
failed.

The action-level operator was also negative (−32.5 pp; bootstrap lower bound
−50.0 pp). Because the oracle action correction was negative as well, that result
is recorded as a failed fixed action coupling, not evidence that velocity is
irrelevant to every possible controller.

## Interpretation and stop rule

The earlier predictive probe remains valid: position is decodable from committed
memory while velocity is weakly decodable. The new result says that this
representation-level gap was not rescued by the tested fixed linear temporal
operator. In particular, the zero oracle-memory gain argues against simply
adding a scalar velocity signal along this frozen direction and expecting the
existing policy to use it.

This is not evidence that the entire Storage–Dynamics research direction is
mathematically impossible. It is a negative result for the preregistered cheap
causal rescue. Per the execution policy, stop before training a larger memory
architecture or spending on a broader sweep. A future continuation would need a
newly justified controller-level intervention or a separately preregistered
method, not post-hoc tuning of these seeds.

## Artifacts

- action full report:
  `artifacts/reports/intercept_medium_k2_temporal_operator_causal_n40.json`
- action report SHA-256:
  `a85c45f840f6b8904ce8e8876e9ea02f40d204b8c9dfaa6ff25a9d4fa2af89bb`
- memory full report:
  `artifacts/reports/intercept_medium_k2_memory_temporal_operator_causal_n40.json`
- memory report SHA-256:
  `ad00d0cffc979fa422a672d6a6ece1f908e10406f0a6b5cff74df3122385c3db`
- tracked summary:
  `results/track2_temporal_operator_causal_2026_09_08.json`
