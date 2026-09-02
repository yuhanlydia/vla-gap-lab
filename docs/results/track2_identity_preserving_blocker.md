# Track 2 identity-preserving intervention blocker

Date: 2026-09-02

The requested privileged intervention must change the simulator-derived target
slot while preserving the hidden target identity. The current released
mu-VLA checkpoint cannot expose that factorization safely:

1. `memory_module--150000_checkpoint.pt` provides one initial tensor and the
   model returns a single `(batch, 64, hidden_dim)` recurrent state. There is
   no identity slot, slot pointer, or public factorized update API.
2. The primary `ShellGameShuffleTouch-VLA-v0` environment does expose
   `cup_with_ball_number` and `slot_of_mug`, but those are simulator labels;
   they do not define a mapping from a desired slot to a latent memory edit.
3. Replacing the state with a fresh/current candidate is not identity
   preserving. The existing reset-refresh intervention demonstrably erases the
   target identity and is therefore only a destructive baseline.
4. The color-lamp replication was checked at 50 seeds, but its `target_color`
   is revealed by the lamp during manipulation. Its label cannot serve as a
   hidden identity carried through the shuffle.

The safe next experiment requires either a checkpoint with factorized memory
slots or a separately trained, held-out latent editor whose identity
preservation is validated before closed-loop use. Training such an editor from
the current data would change the intervention from a privileged diagnostic to
a new learned method, so it is not silently substituted here.

## Reproducibility evidence

- Primary memory-content probe: `docs/results/track2_memory_content_probe.md`.
- 50-seed color-lamp control and semantic check:
  `docs/results/track2_color_lamp_replication.md`.
- The collector now records `target_identity_semantics` and rejects tracking
  environments without a recognized identity field.

This is a protocol blocker, not a positive Gate-0 result; no
Conflict-Adaptive Refresh training is authorized by the current evidence.
