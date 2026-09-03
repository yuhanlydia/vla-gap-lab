# Track 2 runtime root-cause audit

Date: 2026-09-03

## Status

The newest Stage-0 parity JSONs are not committed to GitHub because `artifacts/`
is git-ignored. The newest committed experimental result remains the failed
Identity-Location Gate-1. While auditing the parity path, we found a more
fundamental runtime mismatch that must be fixed before interpreting any new
Track-2 score.

## Confirmed root cause: wrong Transformers fork

The historical Track-2 setup instructions installed:

```text
https://github.com/moojink/transformers-openvla-oft.git
```

The released mu-VLA implementation instead pins the memory-aware fork at the
exact revision:

```text
https://github.com/CognitiveAISystems/transformers-mu-openvla-oft.git
9dbc09f574912a45dd0d71354c035e3c37bcce9e
```

This is not a cosmetic dependency difference. That mu-VLA commit adds
`use_mu_vla_memory_mask` to Llama configuration and changes Llama attention so
a custom 4D additive memory mask is used as-is. The ordinary OpenVLA-OFT fork
does not contain that flag. Using the wrong fork can therefore silently change
the recurrent memory attention contract while the model still loads and runs.

The Track-2 requirements now pin the exact official fork and tokenizers version.
`ProtocolMatchedMuVLAPolicy` checks PEP-610 VCS provenance at startup and refuses
to run parity/Gate-2 if the source URL or resolved commit is wrong.

## Second confirmed mismatch: action clipping

The released MIKASA evaluator applies:

```text
action = clip(action, -1, 1)
```

immediately before the simulator step. The historical local adapter did not.
The protocol-matched policy now performs the same clipping.

## Investigated and ruled out as primary causes

### MIKASA-Robo submodule revision

The lab repo previously referenced MIKASA-Robo commit `509b875...`, while
mu-VLA pins tag `v1.0.0` (commit `16634db...`). Comparing them shows only six
documentation-site commits and no task/wrapper/environment code differences.
The gitlink is nevertheless reset to the exact v1.0.0 commit for unambiguous
provenance.

### Action head and proprio projector

The locally reconstructed L1 action head and proprio projector match the
released mu-VLA architectures: two MLP-ResNet blocks, the same action-token
reshape, and the same two-layer GELU proprio projection. They are not the
leading explanation for parity failure.

### Render overlays

MIKASA task-specific overlay wrappers modify `render()` output rather than the
policy observation `obs["rgb"]`. Running with overlays disabled therefore does
not explain a policy-input mismatch.

## Consequence for historical Track-2 results

Historical Track-2 artifacts were produced before dependency provenance was
recorded and the repository instructions explicitly selected the wrong fork.
They must therefore be treated as **protocol-compromised until replicated**
under the pinned memory-aware runtime. Do not use their absolute SR values as
faithful released-mu-VLA benchmark numbers.

The qualitative findings remain useful for deciding what to retest, but no
method or mechanism claim should rely on them without replication.

## Required rerun order

1. `git pull` and `git submodule update --init --recursive`.
2. Reinstall `requirements/track2-extra.txt` in the MIKASA environment.
3. Run `scripts/check_mu_vla_runtime.py`; it must report the exact official VCS
   URL/commit and versions `transformers==4.40.1`, `tokenizers==0.19.1`.
4. Rerun only the 20-seed train-task parity set:
   `ShellGamePush`, `InterceptMedium`, `RememberColor5`, K=2, 4-bit.
5. If one parity task still fails, repeat only that task in BF16 on a 24 GB GPU.
6. Do not run the 40-episode predictive-dynamics collection until parity is
   credible.

This minimizes additional GPU spend while separating runtime, quantization,
and scientific failure modes.
