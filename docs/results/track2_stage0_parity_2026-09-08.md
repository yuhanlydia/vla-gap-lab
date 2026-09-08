# Track 2 Stage 0 parity — 2026-09-08

> **Root-cause resolution:** the original local evaluator omitted the official
> evaluator's per-step `env.render()` simulator synchronization. ShellGamePush
> mugs fell through the table, invalidating both original 0/20 runs. After
> restoring that behavior, the pinned-asset same-seed NF4 run scored 20/20.
> Archive inspection also showed byte-identical `025_mug` files, rejecting the
> YCB mismatch as the causal explanation. See
> `docs/results/track2_ycb_asset_audit_2026-09-08.md`.

## Final decision

**Stage-0 parity passes.** The valid results are ShellGamePush 20/20,
InterceptMedium 9/20, and RememberColor5 18/20. All are within the frozen
20-percentage-point tolerance. The 40-episode InterceptMedium Gate-2 collection
is authorized; no method claim is authorized until that diagnostic completes.

## Frozen protocol

- repository base: `vla-gap-lab` main at `8f6fdd7b71c7ce432279325021f68f643473afcf`
- evaluator synchronization fix: uncommitted working-tree patch at run time;
  the tracked compact report records the exact result and artifact hash
- MIKASA-Robo submodule: `16634db18bef08128ed79346469c86fc12169aed` (`v1.0.0`)
- checkpoint: official `mu-vla-m64-k2`
- primary precision: NF4 4-bit
- fallback precision: BF16, only for the failed parity task
- episodes: 20 per task
- start seed: `4242424242`
- protocol: `official_224_center_crop_0.9`, `rgb_array`, batch size 1
- parity tolerance: 20 percentage points absolute
- runtime check: passed with `transformers==4.40.1`,
  `tokenizers==0.19.1`, and
  `CognitiveAISystems/transformers-mu-openvla-oft` at
  `9dbc09f574912a45dd0d71354c035e3c37bcce9e`

## Results

| Task | Observed SR | Reference SR | Absolute error | 20pp parity |
|---|---:|---:|---:|---|
| `ShellGamePush-VLA-v0` (NF4, fixed evaluator) | 20/20 = 1.00 | 0.96 | 4pp | PASS |
| `InterceptMedium-VLA-v0` (NF4) | 9/20 = 0.45 | 0.55 | 10pp | PASS |
| `RememberColor5-VLA-v0` (NF4) | 18/20 = 0.90 | 0.94 | 4pp | PASS |

The two historical ShellGamePush 0/20 runs remain invalid because their local
rollout omitted the official per-step render/synchronization side effect. The
fixed evaluator succeeded on every same-seed NF4 episode, typically in 11–22
steps. The YCB mismatch was separately eliminated as a cause because every
`025_mug` file is byte-identical between the two archives.

## Artifacts

The episode JSONs are intentionally under gitignored `artifacts/`:

- `artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_4bit_n20.json`
- `artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_bf16_n20.json`
- `artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_4bit_ycbpinned_n20.json`
- `artifacts/mikasa/parity_InterceptMedium-VLA-v0_k2_4bit_n20.json`
- `artifacts/mikasa/parity_RememberColor5-VLA-v0_k2_4bit_n20.json`
- `artifacts/reports/mu_vla_k2_protocol_parity_4bit_n20.json`
- `results/track2_stage0_parity_fixed_2026_09_08.json` (tracked compact report)

The original analyzer report records the invalid pre-fix failure. The pinned
replacement artifact and `results/gate0_summary.yaml` contain the final
decision.

## Environment notes

The repository environment was refreshed with the frozen lock and
`requirements/track2-extra.txt`. The container initially lacked `uv`, a C
compiler/Python 3.10 headers, OpenGL/Vulkan runtime libraries, and the ManiSkill
YCB asset. These were repaired without changing repository code or experiment
thresholds. The current upstream YCB archive passed Hugging Face download
verification with SHA-256
`1551724fd1ac7bad9807ebcf46dd4a788caed5c9499c1225b9bfa080ffbefcb3`; the
installed ManiSkill package expected SHA-256
`174001ba1003cc0c5adda6453f4433f55ec7e804f0f0da22d015d525d02262fb` and
rejected the current archive, so the current archive was manually extracted to
the standard asset path. This was a provenance mismatch, but the later archive
comparison showed that it did not alter any `025_mug` file.

`vulkaninfo` and a minimal `ShellGamePush-VLA-v0` GPU reset both succeeded
after the runtime repair. The formal evaluator still emits SAPIEN's non-fatal
Vulkan-ICD warning, but all 60 primary episodes and 20 BF16 fallback episodes
completed and wrote atomic reports.

## Next action

This action is complete. See
`docs/results/track2_predictive_dynamics_gate2_2026-09-08.md`.
