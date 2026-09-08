# Track 2 ShellGamePush asset audit — 2026-09-08

## Trigger

Stage-0 parity under the corrected mu-VLA runtime produced a strongly task-specific failure:

| Task | Observed | Released reference |
|---|---:|---:|
| `ShellGamePush-VLA-v0` | 0/20 = 0% | ~96% |
| `InterceptMedium-VLA-v0` | 9/20 = 45% | ~55% |
| `RememberColor5-VLA-v0` | 18/20 = 90% | ~94% |

A BF16 rerun of `ShellGamePush-VLA-v0` was also 0/20. This rules out NF4 as the primary explanation and makes a task-specific environment dependency more likely than a general checkpoint/runtime failure.

## Confirmed protocol mismatch

MIKASA-Robo v1.0.0 pins `mani_skill==3.0.0b15`. That ManiSkill release defines the YCB source archive with SHA-256:

```text
174001ba1003cc0c5adda6453f4433f55ec7e804f0f0da22d015d525d02262fb
```

The Stage-0 machine could not pass that checksum because the mutable Hugging Face `main` URL now serves a different archive. To make the environment launch, the run manually extracted the current archive, whose SHA-256 is:

```text
1551724fd1ac7bad9807ebcf46dd4a788caed5c9499c1225b9bfa080ffbefcb3
```

Hugging Face history shows that the YCB archive changed after the version expected by ManiSkill b15. The replacement commit is described as adding/fixing YCB objects with `density+scale`. The parent revision immediately before that replacement is pinned by this repository as:

```text
4c134e2e33b11b1fa751c1d7b6afb2e89231b875
```

and resolves to the b15-expected archive checksum `174001ba...`.

## Why this specifically threatens ShellGamePush

`ShellGamePush-VLA-v0` builds all three shells from the ManiSkill YCB asset `025_mug`. Its environment code then multiplies both collision and visual record scales by `MUG_SCALE=1.3`. The task's success depends on physically pushing the correct mug forward.

Therefore a YCB archive change affecting object scale/density/collision metadata can alter both visual geometry and contact dynamics. Among the three Stage-0 parity tasks, ShellGamePush is the one directly dependent on this YCB mug asset, matching the observed pattern that only this task collapses.

This is the strongest current root-cause candidate, but it is **not yet causally confirmed** because the exact old asset has not yet been rerun on the same seeds.

## Code change

Track 2 now pins the historical YCB archive explicitly:

- dataset: `haosulab/ManiSkill2`
- revision: `4c134e2e33b11b1fa751c1d7b6afb2e89231b875`
- file: `data/mani_skill2_ycb.zip`
- required SHA-256: `174001ba1003cc0c5adda6453f4433f55ec7e804f0f0da22d015d525d02262fb`

Run:

```bash
cd external/MIKASA-Robo
PYTHONPATH=../../src uv run python ../../scripts/install_track2_ycb_asset.py
```

The installer verifies the archive checksum before replacing the active asset directory and writes a provenance marker. `eval_mu_vla_protocol.py` refuses to run `ShellGamePush-VLA-v0` without that exact marker.

## Minimal causal rerun

Do **not** rerun InterceptMedium or RememberColor5. Keep every other variable fixed and rerun only:

```bash
cd external/MIKASA-Robo
PYTHONPATH=../../src uv run python ../../scripts/eval_mu_vla_protocol.py \
  --checkpoint ../../models/mu-vla-m64-k2 \
  --task ShellGamePush-VLA-v0 \
  --precision 4bit \
  --episodes 20 --start-seed 4242424242 \
  --output ../../artifacts/mikasa/parity_ShellGamePush-VLA-v0_k2_4bit_ycbpinned_n20.json
```

Decision:

- `SR >= 0.76`: the original 20pp Stage-0 tolerance is met. Treat the asset mismatch as confirmed and combine this rerun with the already-valid InterceptMedium (45%) and RememberColor5 (90%) parity results. Gate-2 may then start.
- `SR < 0.76`: asset mismatch was not sufficient. Keep Gate-2 blocked and next compare the official mu-VLA evaluator against the local evaluator step-by-step on ShellGamePush (images, proprio, action, memory, curriculum phase, and success state).

Do not use the earlier 0/20 ShellGamePush run as evidence against the scientific Storage-Dynamics hypothesis; its asset provenance did not match the pinned benchmark environment.
