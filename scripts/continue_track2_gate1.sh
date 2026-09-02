#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY_MI="$ROOT/external/MIKASA-Robo/.venv/bin/python"
PY_BASE="$ROOT/.venv/bin/python"
K8_PID="${K8_WATCHER_PID:-745743}"

episode_count() {
  "$PY_MI" - "$1" <<'PY'
import json, sys
try:
    print(len(json.load(open(sys.argv[1])).get("episodes", [])))
except Exception:
    print(0)
PY
}

# The existing horizon watcher starts K8 after K2.  Wait for its complete
# artifact, but fail closed if that watcher exits before reaching 30 episodes.
while true; do
  if [[ -f artifacts/mikasa/ShellGameShuffleTouch-Long-VLA-v0_k8_normal_n30.json ]] \
      && [[ "$(episode_count artifacts/mikasa/ShellGameShuffleTouch-Long-VLA-v0_k8_normal_n30.json)" == 30 ]]; then
    break
  fi
  if ! kill -0 "$K8_PID" 2>/dev/null; then
    echo "K8 horizon watcher exited before a complete 30-episode artifact" >&2
    exit 1
  fi
  sleep 30
done

TRAJ="artifacts/mikasa/shell_shuffle_k2_strided_n100.npz"
if [[ ! -f "$TRAJ" ]]; then
  PYTHONPATH=src "$PY_MI" scripts/collect_mu_vla_memory_trajectory.py \
    --checkpoint models/mu-vla-m64-k2 \
    --task ShellGameShuffleTouch-VLA-v0 \
    --episodes 100 --start-seed 4242424242 \
    --pooling strided --checkpoint-every 5 --resume \
    --output "$TRAJ" \
    > artifacts/mikasa/shell_shuffle_k2_strided_n100.log 2>&1
fi

PYTHONPATH=src "$PY_BASE" scripts/fit_mu_vla_identity_slot_editor.py \
  --trajectory "$TRAJ" \
  --train-episodes 80 --dev-episodes 20 --pca-dim 256 \
  --alpha-grid 1,10,100,1000 --scale-grid 0.25,0.5,1,2 \
  --output artifacts/mikasa/shell_shuffle_k2_ipsi_editor.npz \
  --report artifacts/reports/shell_shuffle_k2_ipsi_editor.json \
  > artifacts/mikasa/shell_shuffle_k2_ipsi_editor_fit.log 2>&1

SANITY="$($PY_BASE - <<'PY'
import json
try:
    report = json.load(open("artifacts/reports/shell_shuffle_k2_ipsi_editor.json"))
    chosen = report["chosen_scale"]
    good = (
        report.get("identity_dev_balanced_accuracy", 0.0) > 0.0
        and report.get("slot_dev_balanced_accuracy", 0.0) > 0.0
        and chosen["slot_after"] > chosen["slot_before"]
        and chosen["identity_after"] >= chosen["identity_before"] - 0.05
    )
    print("1" if good else "0")
except Exception:
    print("0")
PY
)"
if [[ "$SANITY" != 1 ]]; then
  echo "editor offline sanity failed; causal Gate-1 was not started" >&2
  exit 2
fi

TASK=ShellGameShuffleTouch-VLA-v0
EDITOR=artifacts/mikasa/shell_shuffle_k2_ipsi_editor.npz
for mode in normal random_orthogonal slot_only ipsi; do
  extra=()
  if [[ "$mode" != normal ]]; then
    extra=(--editor "$EDITOR" --edit-seed 0)
  fi
  PYTHONPATH=src "$PY_MI" scripts/eval_mu_vla_identity_slot_intervention.py \
    --checkpoint models/mu-vla-m64-k2 --task "$TASK" --mode "$mode" \
    --episodes 50 --start-seed 4242524242 --resume \
    "${extra[@]}" \
    --output "artifacts/mikasa/${TASK}_${mode}_gate1_n50.json" \
    > "artifacts/mikasa/${TASK}_${mode}_gate1_n50.log" 2>&1
done

PYTHONPATH=src "$PY_BASE" scripts/analyze_mu_vla_identity_location_gate.py \
  --normal "artifacts/mikasa/${TASK}_normal_gate1_n50.json" \
  --random "artifacts/mikasa/${TASK}_random_orthogonal_gate1_n50.json" \
  --slot-only "artifacts/mikasa/${TASK}_slot_only_gate1_n50.json" \
  --ipsi "artifacts/mikasa/${TASK}_ipsi_gate1_n50.json" \
  --output "artifacts/reports/${TASK}_identity_location_gate1.json" \
  > artifacts/mikasa/${TASK}_identity_location_gate1_analyze.log 2>&1
