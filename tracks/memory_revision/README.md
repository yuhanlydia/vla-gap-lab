# Track 2 — Memory Structure in VLA Control

The original Persistence–Revision / Identity–Location line is stopped. Historical absolute
Track-2 scores before the runtime repair are protocol-compromised because the old setup
used a non-memory Transformers fork. See
[`docs/results/track2_runtime_root_cause.md`](../../docs/results/track2_runtime_root_cause.md).

## Active question: Storage–Dynamics Gap

The active phenomenon is deliberately **in distribution** on the released K=2 checkpoint:

```text
Does recurrent memory encode current physical state
without encoding the temporal dynamics needed for predictive control?
```

No new memory method is authorized until released-checkpoint parity passes and the
`InterceptMedium-VLA-v0` representation diagnostic establishes which failure branch is real.

## Current parity status — 2026-09-08

The corrected memory-aware runtime produced:

- `InterceptMedium-VLA-v0`: 9/20 = 45% vs released ~55% — PASS;
- `RememberColor5-VLA-v0`: 18/20 = 90% vs released ~94% — PASS;
- `ShellGamePush-VLA-v0`: 0/20 in NF4 and 0/20 in BF16 — **asset-confounded**.

The ShellGame machine manually extracted the current Hugging Face YCB archive after
ManiSkill 3.0.0b15 rejected it against its historical checksum. ShellGamePush directly uses
YCB `025_mug`; the two passing parity tasks do not. Track 2 now pins the exact historical
archive expected by ManiSkill b15 and refuses ShellGame evaluation without provenance.

See
[`docs/results/track2_ycb_asset_audit_2026-09-08.md`](../../docs/results/track2_ycb_asset_audit_2026-09-08.md).

## Next run — one task only

1. `git pull` and reinstall Track-2 requirements if needed;
2. run `scripts/check_mu_vla_runtime.py`;
3. run `scripts/install_track2_ycb_asset.py`;
4. rerun only `ShellGamePush-VLA-v0`, K2, NF4, the same 20 seeds;
5. if `SR >= 0.76`, Stage-0 parity is restored and the 40-episode
   `InterceptMedium-VLA-v0` dynamics collection may start;
6. if `<0.76`, keep Gate-2 blocked and compare official-vs-local ShellGame rollout state
   step-by-step before changing any scientific hypothesis.

Full commands and the later Storage–Dynamics decision tree are frozen in
[`docs/experiments/track2_predictive_dynamics_gate2.md`](../../docs/experiments/track2_predictive_dynamics_gate2.md).

Historical IPSI and `ConflictAdaptiveRefresh` code remains only for reproducibility.
