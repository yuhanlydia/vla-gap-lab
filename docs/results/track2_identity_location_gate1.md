# Track 2 Identity–Location Revision Gate-1

The causal intervention was completed on
`ShellGameShuffleTouch-VLA-v0` with 50 paired seeds
`4242524242..4242524291`. The K=2 checkpoint and the editor fitted from the
separate 100-episode trajectory set were used exactly as frozen in the
runbook. Raw JSON artifacts remain local under `artifacts/mikasa/` because
experiment outputs are git-ignored.

| condition | episodes | successes | success rate |
| --- | ---: | ---: | ---: |
| normal | 50 | 0 | 0.00% |
| random_orthogonal | 50 | 0 | 0.00% |
| slot_only | 50 | 1 | 2.00% |
| IPSI | 50 | 1 | 2.00% |

The IPSI-minus-normal paired difference is **+2pp**, with paired-bootstrap
95% CI `[0pp, 6pp]`; the exact McNemar p-value is `1.0`. IPSI-minus-random is
also +2pp with CI `[0pp, 6pp]`, while IPSI-minus-slot-only is 0pp with CI
`[-6pp, 6pp]`. The representation diagnostic passed (slot probe gain
`+67.95pp`, identity accuracy change `0pp`), but the behavioral and random
control criteria failed. Overall Gate-1 is therefore **FAILED**.

Per the frozen decision rule, this stops the Identity–Location Revision
mechanism: do not train Dual-Timescale Memory from this result. The editor
demonstrated latent slot decodability but did not causally recover closed-loop
success on the primary task.

Raw artifacts:

- `ShellGameShuffleTouch-VLA-v0_normal_gate1_n50.json`
- `ShellGameShuffleTouch-VLA-v0_random_orthogonal_gate1_n50.json`
- `ShellGameShuffleTouch-VLA-v0_slot_only_gate1_n50.json`
- `ShellGameShuffleTouch-VLA-v0_ipsi_gate1_n50.json`
- `ShellGameShuffleTouch-VLA-v0_identity_location_gate1.json`
