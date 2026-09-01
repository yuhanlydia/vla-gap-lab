# Track 2 memory-content probe on dynamic tracking

This diagnostic uses the released μVLA M64/K2 checkpoint on the official
`ShellGameShuffleTouch-VLA-v0` environment. It replaces the earlier reset
intervention with simulator-label probes that do not erase the hidden target.

## Protocol

- 24 deterministic episodes, seeds 4242424242–4242424265
- 1,440 memory states collected before the current observation update
- Features: concatenated mean, standard deviation, first token, and last token
  over the 64 persistent memory tokens (16,384 dimensions)
- Labels supplied only for analysis:
  - `target_mug`: original mug identity hiding the ball
  - `target_slot`: current slot of that mug after completed swaps
- Four-fold RidgeClassifier evaluation holding out whole episodes
- Alpha sweep: 1, 10, 100, 1000
- Episode-cluster bootstrap intervals below use alpha 100

The policy succeeded in **0/24** episodes. Probe labels are never exposed to the
policy and are not a method or an oracle observation.

## Results

Balanced accuracy has a three-class chance level of 0.333.

| Segment | Samples | Target identity | Current target slot |
|---|---:|---:|---:|
| Cue | 77 | 0.771 [0.655, 0.849] | 0.771 [0.655, 0.849] |
| Shuffle, 0 swaps complete | 234 | 0.893 [0.726, 0.991] | 0.893 [0.726, 0.991] |
| Shuffle, 1 swap complete | 234 | 0.631 [0.445, 0.799] | 0.349 [0.187, 0.549] |
| Shuffle, 2 swaps complete | 120 | 0.306 [0.127, 0.521] | 0.380 [0.193, 0.529] |
| Shuffle, 3 swaps complete | 56 | 0.250 [0.114, 0.333] | 0.180 [0.017, 0.357] |
| Manipulation | 709 | 0.452 [0.322, 0.579] | 0.254 [0.110, 0.412] |

The first completed swap is the cleanest persistence–revision contrast: target
identity remains decodable, while its revised location falls to chance. At the
final manipulation phase, identity remains weakly decodable but current slot is
not. Results were nearly unchanged across the four ridge alphas; for example,
manipulation identity ranged 0.441–0.454 and slot 0.254–0.263.

## Interpretation

This supports a **candidate** persistence–revision failure in the memory
representation, but it does not pass the preregistered method-training Gate:

1. The identity/slot cluster-bootstrap intervals overlap with only 24 episodes.
2. Later shuffle segments have few samples and should not be used alone.
3. Four summary statistics may still miss information distributed across the
   complete token set.
4. The prior reset-from-initial-memory intervention is not an oracle refresh:
   it destroys target identity, and it did not improve success.

Therefore conflict-adaptive refresh remains untrained. A defensible next step
is a larger episode-level replication or a full-token probe, followed by a
causal intervention that preserves identity while changing only the tracked
location state.
