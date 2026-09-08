# Track 2 Stage 0 parity — 2026-09-08

## Decision

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
| `ShellGamePush-VLA-v0` (NF4) | 0/20 = 0.00 | 0.96 | 96pp | **FAIL** |
| `InterceptMedium-VLA-v0` (NF4) | 9/20 = 0.45 | 0.55 | 10pp | PASS |
| `RememberColor5-VLA-v0` (NF4) | 18/20 = 0.90 | 0.94 | 4pp | PASS |

The failed ShellGamePush task completed all 30 steps for every seed without a
runtime exception. The prescribed BF16 repeat on the same seeds was also
0/20, with all episodes completing 30 steps. Thus the run does not support
calling the failure an NF4-only blocker, and it does not establish a credible
released-checkpoint parity protocol for the full Stage 0 set.

## Artifacts

The episode JSONs are intentionally under gitignored `artifacts/`:

- `artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_4bit_n20.json`
- `artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_bf16_n20.json`
- `artifacts/mikasa/parity_InterceptMedium-VLA-v0_k2_4bit_n20.json`
- `artifacts/mikasa/parity_RememberColor5-VLA-v0_k2_4bit_n20.json`
- `artifacts/reports/mu_vla_k2_protocol_parity_4bit_n20.json`

The machine-readable analyzer report records `passed: false` because of
ShellGamePush. The failed task was not replaced by BF16 in the primary parity
report; the fallback is recorded separately to avoid mixing precisions.

## Environment notes

The repository environment was refreshed with the frozen lock and
`requirements/track2-extra.txt`. The container initially lacked `uv`, a C
compiler/Python 3.10 headers, OpenGL/Vulkan runtime libraries, and the ManiSkill
YCB asset. These were repaired without changing repository code or experiment
thresholds. The current upstream YCB archive passed Hugging Face download
verification with SHA-256
`1551724fd1ac7bad9807ebcf46dd4a788caed5c9499c1225b9bfa080ffbefcb3`; the
installed ManiSkill package contains an older checksum and rejected the current
archive, so the verified archive was extracted to the standard asset path.

`vulkaninfo` and a minimal `ShellGamePush-VLA-v0` GPU reset both succeeded
after the runtime repair. The formal evaluator still emits SAPIEN's non-fatal
Vulkan-ICD warning, but all 60 primary episodes and 20 BF16 fallback episodes
completed and wrote atomic reports.

## Next action

Investigate the ShellGamePush parity discrepancy before rerunning Stage 0 or
starting Gate-2. Do not interpret the 0.45 InterceptMedium result as Gate-2
evidence: the runbook's full Stage 0 parity gate did not pass.
