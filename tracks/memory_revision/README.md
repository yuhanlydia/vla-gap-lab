# Track 2 — Identity–Location Revision Gap

The original Persistence–Revision hypothesis has been narrowed by Gate-0. On
`ShellGameShuffleTouch-VLA-v0`, the recurrent state remains predictive of the
hidden target identity after a swap while its current-slot probe falls to
chance. The next question is causal: does correcting **where** the remembered
target is, while minimally changing **which target** is remembered, rescue the
closed-loop policy?

Do not treat a full memory reset as an identity-preserving oracle. The released
mu-VLA checkpoint exposes one opaque `(batch, 64, hidden_dim)` recurrent state,
so Gate-1 learns local identity and slot subspaces from held-out trajectories
instead of assigning semantic roles to memory tokens by hand.

Execution is frozen in
[`docs/experiments/track2_identity_location_gate1.md`](../../docs/experiments/track2_identity_location_gate1.md):

1. compare the official K=2 and K=8 TBPTT checkpoints;
2. if longer TBPTT does not rescue the primary task, fit the editor using 80
   train + 20 dev episodes from the K=2 checkpoint;
3. run paired `normal`, `random_orthogonal`, `slot_only`, and `ipsi` conditions;
4. authorize a learned memory architecture only if the causal Gate-1 passes.

`ConflictAdaptiveRefresh` remains in the package only for reproducibility of
older pilots. It is not the authorized next method.
