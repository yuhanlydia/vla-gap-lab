# LIBERO-Plus language-metadata audit

This audit arose while constructing a vision-only Camera comparison. It is a
benchmark-protocol diagnostic, not a claim that the published category scores
are wrong.

## Static repository audit

LIBERO-Plus `grab_language_from_filename()` uses the entire BDDL filename as
the task instruction whenever `_language_` is absent. Consequently internal
perturbation parameters such as `view 0 0 100 2 354 initstate 0` become part of
the policy prompt.

The checked-in classification contains 10,030 tasks:

| Category | Tasks | Filename-derived instructions with metadata |
|---|---:|---:|
| Background | 1,076 | 1,076 |
| Camera | 1,599 | 1,599 |
| Language | 1,537 | 0 |
| Light | 1,142 | 1,142 |
| Layout | 1,525 | 1,525 |
| Robot initialization | 1,550 | 1,550 |
| Sensor noise | 1,601 | 1,601 |
| **Total** | **10,030** | **8,493** |

Language perturbations take a separate BDDL-parsing branch, so they are not
counted as filename contamination. `scripts/audit_libero_plus_language.py`
reproduces this count without loading a simulator.

## Single-task 2×2 causal pilot

OpenVLA-OFT was evaluated for 10 deterministic initial states with scene BDDL
(clean or Camera task 609) and instruction (natural or filename-suffixed):

| Scene | Natural instruction | Suffixed instruction |
|---|---:|---:|
| Clean BDDL | 9/10 | 3/10 |
| Camera BDDL | 10/10 | 5/10 |

Holding the clean scene fixed, the suffix changes success by **-0.60** (paired
bootstrap 95% interval `[-0.90, -0.30]`; exact two-sided McNemar `p=0.03125`,
six natural-only successes). Holding the Camera scene fixed, it changes success
by **-0.50** (`[-0.80, -0.20]`; `p=0.0625`, five natural-only successes).

In contrast, holding the natural instruction fixed produced 9/10 clean versus
10/10 Camera successes. For this task, the apparent failure in the uncorrected
run was therefore language-metadata sensitivity, not camera sensitivity.

## Scope

This is one task with 10 repeated initial states, 4-bit inference, and a newer
Transformers version than the checkpoint requests. The official benchmark uses
many tasks and one trial per task, so this pilot cannot estimate how much the
metadata changes an aggregate leaderboard score. It does establish that a
vision-only causal analysis must canonicalize non-language instructions and
that the default task object's language is a material confound for at least one
OpenVLA-OFT task.
