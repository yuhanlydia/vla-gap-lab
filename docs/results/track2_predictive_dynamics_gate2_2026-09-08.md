# Track 2 Predictive-Dynamics Gate-2 — 2026-09-08

## Decision

**Gate-2 supports the Storage–Dynamics Gap.**

The frozen decision rule was:

```text
R2(position | memory_after) >= 0.50
R2(velocity | memory_after) <= 0.20
```

The held-out test results were:

| Probe | Test R² |
|---|---:|
| position from `memory_after` | 0.7649 |
| velocity from `memory_after` | 0.1693 |
| velocity from `memory_delta` | 0.0823 |
| initial velocity from `memory_after` | -0.2468 |

Position is strongly decodable while current and initial velocity are weak or
absent under the preregistered mean-R² metric. The memory update does not
concentrate velocity information: `0.0823 - 0.1693 = -0.0870`, below the
required +0.15.

This result authorizes the next diagnostic: a minimal causal temporal operator.
It does not yet authorize training a larger memory architecture.

## Collection

- task: `InterceptMedium-VLA-v0` (in-distribution training task)
- checkpoint: released μVLA K=2
- precision: NF4 4-bit
- episodes: 40, seeds `4242624242..4242624281`
- rows: 2,400 total, 60 per episode
- policy successes: 22/40 = 55%
- simulator synchronization: `render_after_step`
- pooling: memory tokens `[0, 8, 16, 24, 32, 40, 48, 56]`
- collection size: 242,054,507 bytes
- ordered collection-manifest SHA-256:
  `1c4245e66e4a2a925fea685ad0a2f9669f6300f698d308fea93771ca42dd2326`

All episode files passed checks for contiguous episode/seed metadata, schema 2,
60-row agreement across arrays, finite values, and the required simulator sync
marker. The observed 55% task success also matches the released reference
success rate of approximately 55%.

## Leakage controls

The probe used deterministic whole-episode splits:

- train: 24 episodes
- dev: 8 episodes
- test: 8 untouched episodes
- split seed: 0
- minimum step: 2
- pre-contact rows only
- PCA fit on train episodes only
- ridge alpha selected on dev episodes only
- final R² evaluated once on test episodes

PCA used 128 dimensions. The selected ridge alpha was 0.1 for position and 100
for current velocity.

The current-velocity result is anisotropic: per-axis test R² is
`[-0.0109, 0.3496]`. The frozen gate uses their mean (0.1693), so the formal
decision is Storage–Dynamics Gap; the directional difference should be retained
when designing the causal operator.

## Artifacts

- episode cache:
  `artifacts/mikasa/intercept_medium_k2_dynamics_n40/`
- full probe report:
  `artifacts/reports/intercept_medium_k2_predictive_dynamics.json`
- full probe report SHA-256:
  `321090bb07bc57ac0a7703e0d65e282707776c01e94caa1e016431bda78cee07`
- tracked probe report:
  `results/track2_predictive_dynamics_gate2_2026_09_08.json`

The episode cache and full report are gitignored; the complete probe metrics and
split IDs are preserved in the tracked report.
