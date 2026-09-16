# VLA Gap Lab

Phenomenon-first diagnostics on existing VLA benchmarks. Track 1
(LIBERO-Plus latent-to-action) and Track 3 (RoboTwin state transport) remain
paused. Track 2 studies μVLA recurrent memory on MIKASA-Robo.

## Current task: corrected ICASSP v3 evaluation

**Do not treat the v2 GO/STOP output as a clean scientific verdict.** The
September 16 audit found a probe-family mismatch, unstandardized/unreliable
MLP fits, and a `reached_status` mask incorrectly called physical pre-contact.
The corrected pipeline is now implemented; **real v3 results are pending**.

The paper question remains: does recurrent memory retain motion information
that is accessible from matched two-frame visual features? This is a diagnostic
protocol, not a new VLA policy or demonstrated performance improvement.

```bash
git pull --ff-only origin main
git submodule update --init --recursive
V3_PHASE=smoke bash scripts/run_track2_icassp.sh
# Only after the smoke verifies replay alignment and sensor/contact APIs:
bash scripts/run_track2_icassp.sh
```

The runner reuses the existing 60 Medium and 60 Fast v2 feature trajectories.
It replays saved actions **without loading the 7B model**, validates the saved
state/reward/success traces, and adds contact/visibility annotations. Old NPZs
and v2 reports are never overwritten. Replay mismatch blocks the run.

Full instructions: [corrected v3 runbook](docs/experiments/track2_icassp_v3.md).
Frozen settings: `configs/memory_revision/icassp_v3.json`.
Repair evidence: [September 16 audit](docs/results/track2_v3_repair_2026-09-16.md).
The older [v2 runbook](docs/experiments/track2_icassp_state_motion.md) is retained
for history, not the current execution authority.

### What v3 changes

- Physical robot-ball contact is sampled at every physics substep and latched;
  `reached_status` is no longer treated as a physical-contact sensor.
- Visibility is measured from ball segmentation in the actual policy-camera
  crop. Current/two-frame/memory probes share identical valid time rows.
- MLP X and Y scaling use training data only; outputs are inverse-transformed
  before metrics. Early stopping uses explicit whole-episode dev data, with
  loss curves, stop reasons, and epoch-limit failures recorded.
- Ridge is the labelled primary gate; converged MLP is a separate consistency
  check. Failed optimization is not automatically a scientific rejection.
- Six effect-size thresholds are unchanged. The correction is explicitly a
  **post-audit reanalysis**, not independent preregistered confirmation.
- No data are filtered by desired velocity, probe accuracy, or task success.
  The previously inspected steps-0--30 window is not the formal population.

### What the latest audit actually says

The README audit at `f4e4ced` reports that the v2 six-condition decision used
MLP, whereas Fast memory position R² `.067` and velocity-y R² `-.161` cited in
the explanation were Ridge values. The old MLP standardized inputs but not
position/velocity targets, and some fits exhausted 500 iterations.

`reached_status` marks reaching an intended hit pose rather than collision.
The audit found at least 517/3,218 nominal pre-contact Fast rows across 26/60
episodes with reversed/large lateral velocity or out-of-range position.
Restricting one diagnostic to steps 0--30 improved position prediction, but
velocity-y remained around `.01`. Thus **the motion hypothesis is not rescued
by that audit**. Saved position/velocity alignment was reported to be sound
(approximately `.99` correlation and `.05s` interval); no axis or one-step
correction is justified.

These values are from the committed README audit. Raw v2 motion-compression
JSONs/caches were not present in committed `results/` when the repair was
prepared. No newly recomputed real GPU score is claimed here.

## Outputs and next decision

```text
artifacts/mikasa/icassp_v3_annotations/{medium,fast}/
artifacts/reports/icassp_motion_compression_v3/
results/track2_icassp_motion_compression_v3_summary.json
```

Push the compact tracked summary after the GPU agent completes the run:

```bash
git add results/track2_icassp_motion_compression_v3_summary.json
git commit -m "results: corrected ICASSP v3 reanalysis"
git push origin main
```

`not_supported` means the corrected primary Ridge gate failed.
`inconclusive_data`, `inconclusive_mlp`, and `inconclusive_probe_disagreement`
mean the specified validity/consistency requirement was not met.
`review_candidate` only means the controlled result merits review: it is not a
claim of information-theoretic loss, new causal control ability, or acceptance.
The outcome must be reported even when negative. No extra architecture/gain
sweep is authorized to rescue the story.

## Hardware and test status

Target: the existing **16GB** setup. Replays use one simulator environment and
no VLA checkpoint. Offline mini-batch MLP and bounded-buffer PCA run on CPU.
No 24GB/BF16 rerun is requested. A fresh GPU replay has not been performed in
the editing environment; the mandatory two-per-task smoke verifies that path.

CPU repair checks:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest -q tests/test_icassp_v3*.py
```

The repository-wide legacy check remains `scripts/check.sh`.

## Historical tracks and evidence

Historical absolute scores must be read with their protocol status, not treated
as conclusions about all VLA memory systems.

- Track 1 original latent-to-action hypothesis did not pass; no ControlSkip
  training is authorized.
- Track 2 Identity–Location IPSI did not show useful behavioral rescue.
- Corrected runtime/parity work restored ShellGamePush 20/20,
  InterceptMedium 9/20, and RememberColor5 18/20 in the small parity runs.
- The original 40-episode dynamics pilot recorded position R² .765, XY-mean
  velocity R² .169 and memory-delta velocity R² .082. Those are limited
  readout results, not proof that memory contains no dynamics.
- The minimal temporal operator did not rescue control. Larger memory-policy
  training remains stopped.
- Track 3 proxy experiments are exploratory; semantic-phase portability is
  still unverified.

[Gate matrix](docs/results/gate0_decision_matrix.md) ·
[Results manifest](results/gate0_summary.yaml) ·
[Gate-2 report](docs/results/track2_predictive_dynamics_gate2_2026-09-08.md) ·
[Temporal operator result](docs/results/track2_temporal_operator_causal_2026-09-08.md) ·
[Runtime audit](docs/results/track2_runtime_root_cause.md) ·
[Asset/render audit](docs/results/track2_ycb_asset_audit_2026-09-08.md) ·
[Track-3 proxy](docs/results/track3_scene_matched_phase_proxy_2026-09-09.md).
