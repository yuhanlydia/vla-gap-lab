# September 16 repair status: evaluation correction, not a positive result

## Evidence inspected

The inspected repository main was `f4e4cedeba3a8ba555d2d15cf03dad9b517f1943`.
Its README records completed v2 Medium/Fast collections and a September 16
evaluation audit. The committed `results/` directory at that revision does
not contain the v2 motion-compression summary or per-episode caches. Therefore
this repair uses the README's numerical audit plus source inspection; it does
**not** claim to have independently recomputed those real GPU measurements.

The README reports:

- the formal six-condition gate used MLP, while the quoted Fast position
  `.067` and velocity-y `-.161` values came from Ridge;
- MLP input scaling without target scaling, multiple max-iteration stops,
  and strongly negative primary MLP results;
- `reached_status` is a hit-pose distance event, not physical robot/ball
  contact, allowing collisions and out-of-view tails into `pre_contact`;
- at least 517 of 3,218 Fast retained rows across 26/60 episodes had atypical
  velocity or out-of-range position;
- saved finite differences/velocity were aligned (approximately .99
  correlation and .05s sampling); no axis swap or one-step offset was found;
- restricting a diagnostic to early steps improved position prediction but
  velocity-y remained around .01. This does **not** rescue the motion claim.

## Code root causes confirmed

`state_motion.fit_mlp_probe` passed unstandardized targets to MLPRegressor and
used its frame-random internal early-stopping split. `make_row_mask` used
`reached_status < .5`. The v2 visual masks checked only the current endpoint,
not visibility of the earlier frame. The summarizer's formal gate took MLP
aggregates. These are evaluation/protocol issues, not evidence that an improved
VLA method has been discovered.

## Changes

v2 source/results are preserved. Default execution is redirected to a distinct
v3 pipeline with train-only target scaling and inverse transformation,
whole-episode dev stopping, convergence traces, separate Ridge/MLP decisions,
paired physical-contact/segmentation masks, strict replay checks, source
hashes, and separate outputs. Source trajectories are replayed without loading
the 7B VLA; the 120 existing feature trajectories can be reused only when the
saved state/reward/success trace agrees. The .05s/axis alignment is not
"corrected" without evidence of an error.

The six numerical effect-size thresholds remain unchanged. The primary-probe
and population corrections are explicitly post-audit decisions. They are not
presented as untouched preregistration. No velocity-based exclusion, tuned
steps-30 cutoff, selected-seed replacement, or negative-result deletion is used.

## Decision

The currently supported conclusion is **v2 is provisional; v3 needs running**.
Correcting the estimator/mask may or may not recover motion accessibility.
If corrected two-frame features still fail to expose motion, that remains a
scientific failure of the proposed compression-loss argument. Do not continue
tuning until a desired result appears.

See `docs/experiments/track2_icassp_v3.md` for the complete execution contract.
