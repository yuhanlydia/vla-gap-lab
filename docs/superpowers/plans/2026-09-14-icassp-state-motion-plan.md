# ICASSP State–Motion Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a one-command, resumable ICASSP experiment pipeline that collects full μVLA recurrent memory on InterceptMedium/Fast and runs leakage-safe state-vs-motion probes plus frozen ablations.

**Architecture:** One core helper module handles token selection, masks, probes, aggregation, and the frozen paper kill rule. A dedicated collector stores all 64 post-update memory tokens. A multi-split probe uses train-only two-stage PCA, Ridge/MLP primary probes, and isolated token/contact ablations. A summarizer emits machine-readable results, paper tables, and figures, while a shell runner executes the whole study.

**Tech Stack:** Python 3.10, NumPy, scikit-learn, matplotlib, existing μVLA/MIKASA Track-2 runtime.

**Spec:** `docs/superpowers/specs/2026-09-14-icassp-state-motion-design.md`

## Global Constraints

- Released μVLA K=2 checkpoint only.
- MIKASA-Robo v1.0.0 and exact memory-aware Transformers runtime.
- NF4 4-bit closed-loop rollout, `num_envs=1`, `render_after_step`.
- 60 episodes each for Medium and Fast.
- Five whole-episode split seeds; 36/12/12 train/dev/test.
- Primary representation: full 64 memory tokens, train-only token PCA 4096→32, then sample PCA→256, pre-contact rows, step >= 2.
- Primary probes: Ridge and fixed 2-layer MLP.
- Isolated ablations: stride-8 token coverage and all-step contact filtering, both with Ridge.
- No new VLA training or temporal-operator tuning.

---

### Task 1: Core state-motion helpers

**Files:**
- Create: `src/vla_gap_lab/state_motion.py`
- Test: `tests/test_state_motion.py`

- [x] Write failing tests for token selection, masks, MLP output, aggregation, and paper decision.
- [x] Verify RED before implementation.
- [x] Implement helpers and frozen kill rule.
- [x] Verify 5 targeted tests pass.

### Task 2: Full-memory collector

**Files:**
- Create: `scripts/collect_mu_vla_state_motion_icassp.py`
- Modify: `tests/test_cli_help.py`

- [x] Add CLI-help requirement.
- [x] Implement 64-token float16 episode-atomic collector for Medium/Fast.
- [x] Preserve protocol metadata and `--resume` validation.

### Task 3: Multi-split primary + ablation probe

**Files:**
- Create: `scripts/probe_mu_vla_state_motion_icassp.py`
- Modify: `tests/test_cli_help.py`

- [x] Implement five fixed 36/12/12 whole-episode splits.
- [x] Implement train-only token PCA and sample PCA.
- [x] Implement full64 Ridge/MLP primary cells.
- [x] Implement stride8 Ridge token ablation.
- [x] Implement full64 all-step Ridge contact ablation.
- [x] Report mean/per-axis R² and median/IQR across splits.

### Task 4: Paper-ready summary and plots

**Files:**
- Create: `scripts/summarize_icassp_state_motion.py`

- [x] Implement combined JSON and tracked result summary.
- [x] Implement CSV and LaTeX table generation.
- [x] Implement PNG/PDF grouped-bar figure with IQR error bars.
- [x] Apply the frozen GO/STOP rule programmatically.

### Task 5: One-command GPU handoff

**Files:**
- Create: `scripts/run_track2_icassp.sh`
- Create: `configs/memory_revision/icassp_state_motion.yaml`
- Create: `docs/experiments/track2_icassp_state_motion.md`
- Modify: `README.md`
- Modify: `scripts/README.md`
- Modify: `requirements/track2-extra.txt`

- [x] Pin `scikit-learn==1.5.2`.
- [x] Add frozen YAML config.
- [x] Add resumable one-command runner.
- [x] Document exact agent workflow, output paths, and stop rule.

### Task 6: Verification and integration

- [x] Targeted helper tests: 5 passed in local isolated verification.
- [x] Synthetic 60-episode end-to-end probe + summarizer smoke completed.
- [x] New Python files compile.
- [x] `bash -n scripts/run_track2_icassp.sh` passes.
- [ ] Inspect branch diff against `main` for accidental historical-result changes.
- [ ] Merge to `main`.

GPU benchmark results are intentionally not claimed here; the runner must be executed by the GPU agent after merge.
