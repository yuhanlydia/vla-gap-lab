# ICASSP v3: correct the evaluation before judging the hypothesis

## What this fixes

The September 16 README audit identified an MLP/Ridge interpretation mismatch,
unscaled MLP targets with unreliable optimization, and a false physical-contact
interpretation of `reached_status`. v3 is a separately versioned **post-audit
reanalysis**, not a newly preregistered independent confirmation. The v2
observations, including negative velocity results, remain intact.

No new memory architecture, VLA training, action steering, or threshold sweep
is authorized. Track 1 and Track 3 are unchanged.

## One-command handoff

From the repository root in the existing GPU checkout:

```bash
git pull --ff-only origin main
git submodule update --init --recursive
V3_PHASE=smoke bash scripts/run_track2_icassp.sh
# Only if smoke completes without replay drift or missing sensor/contact APIs:
bash scripts/run_track2_icassp.sh
```

The first command replays only two Medium and two Fast episodes. The second
reuses those annotations and completes the remaining saved trajectories,
probes, and summary. **It does not rerun the 7B policy.** There is no background
job or automatic git push in this script.

Expected existing caches:

```text
artifacts/mikasa/icassp_motion_compression_medium_n60/
artifacts/mikasa/icassp_motion_compression_fast_n60/
```

Override their locations, without moving/deleting original files:

```bash
MED_CACHE=/absolute/medium FAST_CACHE=/absolute/fast bash scripts/run_track2_icassp.sh
```

Use the existing pinned `external/MIKASA-Robo/.venv/bin/python`. The runner uses
that interpreter directly, not `uv run`, so dependency synchronization cannot
silently undo the μVLA fork installation. On a clean environment only:

```bash
cd external/MIKASA-Robo
uv sync --frozen
uv pip install -r ../../requirements/track2-extra.txt --python .venv/bin/python
```

Do not upgrade ManiSkill to make an API work. Replay requires
`mani_skill==3.0.0b15` and MIKASA commit
`16634db18bef08128ed79346469c86fc12169aed`.

## 1. Simulator-only annotation of the saved actions

`annotate_icassp_v3.py` resets each saved seed and executes its original 7-D
clipped action sequence. It retains the original wrappers and per-step render
synchronization. Before each action it checks saved/current ball XY position,
ball XY velocity, TCP XY position, and goal XY position (absolute tolerance
`1e-4`). Reward is checked at `1e-5`; success is checked exactly. An earlier
termination or mismatch raises an error and leaves the source untouched.
**Do not loosen these tolerances to get a passing run.**

A successful replay writes a small sidecar, not a replacement memory cache.
Every sidecar records the source-file SHA-256, protocol hash, task/seed/episode,
maximum replay errors, and exact simulator revision.

Contact means a nonzero robot-link/ball interaction: force norm greater than
`0.001 N`, queried for every robot link at **every physics substep**. The flag
is latched through the episode. This catches transient contact that has ended
before the next camera frame. It deliberately does not count the ball/table
support force as robot contact. `reached_status` is not used.

Visibility means at least four pixels belonging to the actual ball's
segmentation ID in one policy camera after the same 224px resize and 0.9-area
center crop. Categorical masks use nearest-neighbour interpolation. This is
not a color heuristic, position bounding box, or filter on velocity sign.
Missing segmentation/contact APIs cause an error, not fake all-visible or
no-contact labels.

**Replay limitation:** v2 did not save RGB hashes or the full robot joint
state. Matching the saved state/reward/success trace is a strong consistency
check, but it is not a bitwise proof of identical images. Report this limit;
if replay disagrees, the old features cannot be certified and a reviewed fresh
collection would be required. Such recollection is not launched automatically.

## 2. Frozen masks and matched rows

Formal primary population: steps `t >= 2`, before any recorded robot/ball
contact, with the ball visible in at least one **common camera** at `t`, `t-1`,
and `t-2`. All three primary representations and lag controls use those exact
same `(episode, step)` rows. Missing or nonconsecutive steps are never padded.
Using the lag-2 visibility intersection for all cells makes the lag ablation
paired as well as the primary comparison.

There is **no upper time cutoff**, no filtering by velocity/position target,
and no filtering by success. The README's steps-0--30 diagnostic is not reused
as a formal criterion. A visibility-matched all-contact-state memory cell is
reported separately as a sensitivity analysis, never substituted into the
primary gate.

## 3. Fixed probes and preprocessing

Each task retains 60 episodes with split seeds `0,1,2,3,4` and 36/12/12
whole-episode train/dev/test partitioning. These are repeated partitions of the
same cohort, not five independent data collections.

The existing probe budget is retained: token PCA to 32 dimensions, then sample
PCA to at most 256 dimensions. Token PCA uses only eligible training rows and
their history endpoints; actual incremental batches are bounded, unlike a
constructor `batch_size` value that is ignored by manual `partial_fit` calls.
The modality-specific visual transform is shared by current and paired visual
features. No reducer is fit on dev/test data. All-contacts sensitivity analysis
fits its own train-only transforms.

Both input **and output** scalers are fit only on training data. Predicted
outputs are inverse-transformed before physical-unit errors and R² are
computed. Ridge uses the existing alpha grid `{0.1,1,10,100}`, chosen by dev
standardized MSE. MLP keeps `(128,64)`, Adam, learning rate `1e-3`, and L2
`1e-4`; it uses batch 256 (or all rows if fewer), maximum 500 epochs, patience
25, and tolerance `1e-4`.

MLP early stopping evaluates the **explicit whole-episode dev set** after each
epoch. It never randomly holds out frames from training episodes. The best dev
checkpoint is restored. Epoch-limit termination is recorded as unconverged;
it is not silently considered a scientific failure. Train/dev loss curves,
best epoch, stop reason, scalers, per-axis R²/RMSE/MAE, target variance, and the
train-mean prediction baseline are retained. Negative R² is not clipped.
Constant targets produce undefined R² (`null`), not artificial zero/one scores.

Primary comparisons also require at least 32 rows per split and at least 80%
of each split's episodes to retain eligible rows. Otherwise interpretation is
inconclusive rather than quietly based on a few surviving episodes.

## 4. Experiments executed

For each of five splits and each of Medium/Fast:

| Cell | Representation | Probe | Rows |
|---|---|---|---|
| Main | Current same-projector visual | Ridge + MLP | primary common mask |
| Main | Two-frame visual, lag 1 | Ridge + MLP | same primary mask |
| Main | Full 64-token recurrent memory | Ridge + MLP | same primary mask |
| Token sensitivity | Stride-8 recurrent memory | Ridge | same primary mask |
| Time-span sensitivity | Two-frame visual, lag 2 | Ridge | same primary mask |
| Contact sensitivity | Full memory | Ridge | visibility matched, contact permitted |

Medium is the training-regime task; Fast is a velocity-regime transfer test,
not a different trained checkpoint or independent benchmark family.

## 5. Decision contract

Ridge is now the **named primary gate**, as specified by the README audit. It
retains the six numerical v2 effect-size criteria: Medium pair velocity-y R²
at least .60, pair-minus-current at least .20, pair-minus-memory at least .15,
Medium memory position R² at least .65, Fast pair velocity better than memory,
and Fast memory position R² at least .60. Paired differences are computed per
split before aggregation, not by mixing probe families.

MLP results are reported separately. A valid Ridge failure cannot be rescued
by swapping to MLP. A Ridge pass with unconverged MLP is **inconclusive**, not a
rejection or automatic success. Converged probe disagreement also stays
inconclusive. Only full agreement creates `review_candidate`; that flag does
not establish novelty, causal information loss, statistical significance, or
conference acceptance. Reusing audited cohorts must be disclosed.

## Outputs, resume, and result delivery

```text
artifacts/mikasa/icassp_v3_annotations/{medium,fast}/
artifacts/reports/icassp_motion_compression_v3/{medium,fast}.json
artifacts/reports/icassp_motion_compression_v3/{summary.json,all_cells.csv,table.tex}
results/track2_icassp_motion_compression_v3_summary.json
```

Per-cell probe outputs are checkpointed atomically. Resume requires matching
source hashes, annotations, and protocol. After valid annotations already
exist, `V3_PHASE=probe bash scripts/run_track2_icassp.sh` skips replay.

```bash
git add results/track2_icassp_motion_compression_v3_summary.json
git commit -m "results: corrected ICASSP v3 reanalysis"
git push origin main
```

Do not overwrite/delete v2 summaries or multi-GB caches. The compact v3 summary
includes all probe-family labels, all negative scores, convergence status,
paired-row IDs via report hashes, cohort coverage, and the exact decision.

## Hardware and verification boundary

Designed for the existing 16GB GPU: simulator replay uses one environment and
loads no 7B model. Offline probes run on CPU with bounded token-PCA buffers and
mini-batches; do not co-load a VLA to run these regressors. Four BLAS threads
are the runner default; override thread environment variables for the host.
24GB/BF16 is not required by this repair.

CPU regression tests, synthetic full-pipeline tests, and CLI/shell checks have
been run in the editing environment. **Actual ManiSkill GPU replay and real
v3 metrics remain unrun here.** The two-per-task smoke is mandatory before the
full replay. No v3 scientific result is asserted by this code update.
