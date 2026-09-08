# Track 2 Stage 0 parity — 2026-09-08

> **Post-hoc asset audit:** the `ShellGamePush-VLA-v0` run below used a manually
> extracted current YCB archive (`1551724f...`) after ManiSkill 3.0.0b15 rejected
> it against its pinned checksum (`174001ba...`). Because ShellGamePush uniquely
> depends on the YCB `025_mug` asset among these three parity tasks, its 0/20
> result is now treated as **asset-confounded**, not as a valid released-policy
> parity failure. See `docs/results/track2_ycb_asset_audit_2026-09-08.md`. The
> InterceptMedium and RememberColor5 results remain usable. Gate-2 stays blocked
> until ShellGamePush is rerun on the exact pinned YCB asset.

## Decision at run time

**Parity failed. Stop before Gate-2.** The released K=2 checkpoint passed the
20-episode tolerance on `InterceptMedium-VLA-v0` and `RememberColor5-VLA-v0`,
but failed badly on `ShellGamePush-VLA-v0`. The prescribed BF16 rerun of the
failed task also failed, so the result is not attributable to NF4 quantization
alone. No 40-episode dynamics collection was started and no method claim is
authorized from this run.

## Frozen protocol

- repository: `vla-gap-lab` main at `b3ee3461a2ee774cdb767d33645b571119c14771`
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
| `ShellGamePush-VLA-v0` (NF4) | 0/20 = 0.00 | 0.96 | 96pp | **ASSET-CONFOUNDED** |
| `InterceptMedium-VLA-v0` (NF4) | 9/20 = 0.45 | 0.55 | 10pp | PASS |
| `RememberColor5-VLA-v0` (NF4) | 18/20 = 0.90 | 0.94 | 4pp | PASS |

The failed ShellGamePush task completed all 30 steps for every seed without a
runtime exception. The prescribed BF16 repeat on the same seeds was also
0/20, with all episodes completing 30 steps. Thus the run does not support
calling the failure an NF4-only blocker. The later asset audit found that both
of those ShellGamePush runs shared the same non-pinned YCB asset and therefore
do not isolate model precision or policy competence.

## Artifacts

The episode JSONs are intentionally under gitignored `artifacts/`:

- `artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_4bit_n20.json`
- `artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_bf16_n20.json`
- `artifacts/mikasa/parity_InterceptMedium-VLA-v0_k2_4bit_n20.json`
- `artifacts/mikasa/parity_RememberColor5-VLA-v0_k2_4bit_n20.json`
- `artifacts/reports/mu_vla_k2_protocol_parity_4bit_n20.json`

The original machine-readable analyzer report records `passed: false` because
of ShellGamePush. Do not reinterpret that report as a clean model-parity test
until the pinned-asset ShellGamePush replacement result is available.

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
the standard asset path. That workaround is the protocol mismatch identified
by the later asset audit.

`vulkaninfo` and a minimal `ShellGamePush-VLA-v0` GPU reset both succeeded
after the runtime repair. The formal evaluator still emits SAPIEN's non-fatal
Vulkan-ICD warning, but all 60 primary episodes and 20 BF16 fallback episodes
completed and wrote atomic reports.

## Revised next action

Install the exact pinned YCB archive with `scripts/install_track2_ycb_asset.py`
and rerun only `ShellGamePush-VLA-v0` for the same 20 NF4 seeds. Do not rerun
InterceptMedium or RememberColor5. If pinned-asset ShellGamePush reaches the
original Stage-0 tolerance (`SR >= 0.76`), Stage-0 parity is considered restored
and Gate-2 may start. Otherwise keep Gate-2 blocked and perform a step-by-step
official-vs-local ShellGamePush evaluator trace.
