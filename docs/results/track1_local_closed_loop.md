# Track 1 instruction-matched closed-loop pilot

This is a 10-seed local pilot, not the canonical LIBERO-Plus result. It uses the
official OpenVLA-OFT combined checkpoint in 4-bit mode and task 609 from the
LIBERO Spatial Camera category.

## Pair construction

Both conditions use the task-609 initial-state tensor and exactly the same
instruction:

> pick up the black bowl between the plate and the ramekin and place it on the plate

The clean condition replaces only the task BDDL with the official unperturbed
base BDDL. The shifted condition keeps task 609's camera-perturbed BDDL. This
matters because the benchmark task object's `language` field includes a
`view_...initstate...` suffix. Passing that field directly to the policy changes
both vision and language and is not a camera-only comparison.

## Results

| Condition | Success | Episode outcomes |
|---|---:|---|
| Counterfactual clean | 9/10 | 1,0,1,1,1,1,1,1,1,1 |
| Camera shift | 10/10 | 1,1,1,1,1,1,1,1,1,1 |

- Control retention: **1.111**
- Paired rate difference (shift - clean): **+0.10**
- Paired bootstrap 95% interval: **[0.00, 0.30]**
- Discordant pairs: clean-only 0, shifted-only 1
- Exact two-sided McNemar p: **1.0**

The camera-only pilot therefore provides no evidence of control collapse on
this variant. As an audit, using the benchmark's suffixed language produced
5/10 shifted successes while the same clean run produced 9/10. That result is
confounded and is deliberately excluded from the Gate decision.

## Gate decision

**Not passed.** The matched local control-retention criterion is not below 0.7,
and the prior 80-state Camera representation retention was only 0.717 rather
than the preregistered value above 0.8. ControlSkip remains untrained. The next
useful test is a larger offline representation sample and additional
instruction-matched Camera variants.

The run also used Transformers 4.51.3 rather than the checkpoint's expected
4.40.1. This dependency mismatch and the small sample prevent an official
benchmark claim.
