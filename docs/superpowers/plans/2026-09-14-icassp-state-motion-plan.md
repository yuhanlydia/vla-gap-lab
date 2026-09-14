# ICASSP State–Motion Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-command, resumable ICASSP experiment pipeline that collects full μVLA recurrent memory on InterceptMedium/Fast and runs leakage-safe state-vs-motion probes plus the three frozen ablations.

**Architecture:** Add one focused core module for token selection, masking, split aggregation, and Ridge/MLP fitting. Add a dedicated collector that stores all 64 post-update memory tokens, a probe CLI that evaluates one task across five whole-episode splits, a summarizer/plotter, and a shell orchestrator that runs both tasks end-to-end. Existing Track-2 historical scripts/results remain unchanged.

**Tech Stack:** Python 3.10, NumPy, scikit-learn, matplotlib, existing μVLA/MIKASA Track-2 runtime.

**Spec:** `docs/superpowers/specs/2026-09-14-icassp-state-motion-design.md`

## Global Constraints

- Released μVLA K=2 checkpoint only.
- MIKASA-Robo v1.0.0 and exact memory-aware Transformers runtime.
- NF4 4-bit closed-loop rollout, `num_envs=1`, `render_after_step`.
- 60 episodes each for Medium and Fast.
- Five whole-episode split seeds; 36/12/12 train/dev/test.
- Primary representation: full 64 memory tokens, PCA-256, pre-contact rows, step >= 2.
- Ablations only: stride-8 token coverage, Ridge vs 2-layer MLP, pre-contact vs all-step.
- No new VLA training or temporal-operator tuning.

---

### Task 1: Core state-motion helpers

**Files:**
- Create: `src/vla_gap_lab/state_motion.py`
- Test: `tests/test_state_motion.py`

**Interfaces:**
- Produces `select_memory_tokens(memory, mode)`, `make_row_mask(step, reached_status, min_step, pre_contact_only)`, `fit_ridge_probe(...)`, `fit_mlp_probe(...)`, and `summarize_split_metrics(...)`.

- [ ] **Step 1: Write failing tests** for full/stride token selection, contact mask behavior, deterministic MLP output shape, and median aggregation.
- [ ] **Step 2: Run** `pytest tests/test_state_motion.py -q` and confirm failure because the module does not exist.
- [ ] **Step 3: Implement minimal helpers** using `StandardScaler`, `Ridge`, and fixed `MLPRegressor(hidden_layer_sizes=(128,64), alpha=1e-4, max_iter=500, early_stopping=True, random_state=seed)`.
- [ ] **Step 4: Re-run** `pytest tests/test_state_motion.py -q` and confirm pass.

### Task 2: Full-memory collector

**Files:**
- Create: `scripts/collect_mu_vla_state_motion_icassp.py`
- Modify: `tests/test_cli_help.py`

**Interfaces:**
- Consumes existing `ProtocolMatchedMuVLAPolicy`, `step_mikasa_env`, and atomic NPZ I/O.
- Produces one NPZ per episode containing full `memory_after` plus state/motion labels and protocol metadata.

- [ ] **Step 1: Add collector CLI to the help smoke list** and confirm current test fails because the script is absent.
- [ ] **Step 2: Implement collector** for `InterceptMedium-VLA-v0` and `InterceptFast-VLA-v0`, requiring 64 memory tokens and writing float16 memory.
- [ ] **Step 3: Run CLI-help and CPU tests**.

### Task 3: Multi-split primary + ablation probe

**Files:**
- Create: `scripts/probe_mu_vla_state_motion_icassp.py`
- Modify: `tests/test_cli_help.py`

**Interfaces:**
- Consumes collector episode directories.
- Produces one JSON report per task with five split seeds and all frozen cells: `{full64,stride8} × {ridge,mlp} × {pre_contact,all_steps}`.

- [ ] **Step 1: Add CLI-help test and verify failure**.
- [ ] **Step 2: Implement 36/12/12 whole-episode splitting**, train-only PCA-256, dev-only Ridge alpha selection, fixed MLP, per-axis and mean R².
- [ ] **Step 3: Encode the frozen GO/STOP diagnostics** without changing thresholds post hoc.
- [ ] **Step 4: Run tests**.

### Task 4: Paper-ready summary and plots

**Files:**
- Create: `scripts/summarize_icassp_state_motion.py`
- Modify: `tests/test_cli_help.py`

**Interfaces:**
- Consumes Medium and Fast JSON probe reports.
- Produces combined JSON, CSV, LaTeX table, PNG and PDF figure; also writes `results/track2_icassp_state_motion_summary.json`.

- [ ] **Step 1: Add CLI-help test and verify failure**.
- [ ] **Step 2: Implement median/IQR aggregation** across five splits for state mean R² and velocity-y R².
- [ ] **Step 3: Implement frozen paper decision** from the spec.
- [ ] **Step 4: Generate a compact grouped-bar figure and table** using default matplotlib styling.
- [ ] **Step 5: Run tests**.

### Task 5: One-command GPU handoff

**Files:**
- Create: `scripts/run_track2_icassp.sh`
- Create: `configs/memory_revision/icassp_state_motion.yaml`
- Create: `docs/experiments/track2_icassp_state_motion.md`
- Modify: `README.md`
- Modify: `scripts/README.md`
- Modify: `requirements/track2-extra.txt`

**Interfaces:**
- `bash scripts/run_track2_icassp.sh [checkpoint]` runs runtime validation, both collectors, both probe reports, and final summarization with resume-safe outputs.

- [ ] **Step 1: Pin `scikit-learn==1.5.2`** in Track-2 extra requirements.
- [ ] **Step 2: Add frozen YAML config** with tasks, seeds, splits, thresholds, PCA, and ablations.
- [ ] **Step 3: Add resumable shell runner** with `set -euo pipefail` and explicit output paths.
- [ ] **Step 4: Document exact agent workflow and stop rule**.
- [ ] **Step 5: Run repository CPU checks and shell syntax validation**.

### Task 6: Final verification and merge

- [ ] Run targeted unit tests.
- [ ] Run `python -m compileall src scripts`.
- [ ] Run `bash -n scripts/run_track2_icassp.sh`.
- [ ] Run repository CPU check if dependencies allow.
- [ ] Inspect diff for accidental changes to historical results.
- [ ] Merge to `main` only after verification.
