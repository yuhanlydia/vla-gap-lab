#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIKASA="$ROOT/external/MIKASA-Robo"
PY="${V3_PYTHON:-$MIKASA/.venv/bin/python}"
MED="${MED_CACHE:-$ROOT/artifacts/mikasa/icassp_motion_compression_medium_n60}"
FAST="${FAST_CACHE:-$ROOT/artifacts/mikasa/icassp_motion_compression_fast_n60}"
OUT="$ROOT/artifacts/reports/icassp_motion_compression_v3"
ANN="$ROOT/artifacts/mikasa/icassp_v3_annotations"
PHASE="${V3_PHASE:-all}"
case "$PHASE" in all|smoke|probe) ;; *) echo 'V3_PHASE must be all, smoke, or probe' >&2; exit 2 ;; esac
[[ -x "$PY" ]] || { echo 'Pinned MIKASA Python missing; see docs/experiments/track2_icassp_v3.md' >&2; exit 2; }
[[ -d "$MED" && -d "$FAST" ]] || { echo 'Existing v2 caches not found. Set MED_CACHE and FAST_CACHE; no new 7B rollout is started.' >&2; exit 2; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "$OUT" "$ANN"
cd "$MIKASA"
# Deliberately do not use `uv run`: automatic sync can undo manually installed extras.
if [[ "$PHASE" != probe ]]; then
  "$PY" "$ROOT/scripts/annotate_icassp_v3.py" --episodes-dir "$MED" --task InterceptMedium-VLA-v0 --output-dir "$ANN/medium" --limit 2 --resume
  "$PY" "$ROOT/scripts/annotate_icassp_v3.py" --episodes-dir "$FAST" --task InterceptFast-VLA-v0 --output-dir "$ANN/fast" --limit 2 --resume
  [[ "$PHASE" == smoke ]] && { echo 'Replay smoke complete; no scientific result has been computed.'; exit 0; }
  "$PY" "$ROOT/scripts/annotate_icassp_v3.py" --episodes-dir "$MED" --task InterceptMedium-VLA-v0 --output-dir "$ANN/medium" --resume
  "$PY" "$ROOT/scripts/annotate_icassp_v3.py" --episodes-dir "$FAST" --task InterceptFast-VLA-v0 --output-dir "$ANN/fast" --resume
fi
"$PY" "$ROOT/scripts/probe_icassp_v3.py" --episodes-dir "$MED" --annotations-dir "$ANN/medium" --task InterceptMedium-VLA-v0 --output "$OUT/medium.json" --resume
"$PY" "$ROOT/scripts/probe_icassp_v3.py" --episodes-dir "$FAST" --annotations-dir "$ANN/fast" --task InterceptFast-VLA-v0 --output "$OUT/fast.json" --resume
"$PY" "$ROOT/scripts/summarize_icassp_v3.py" --medium-report "$OUT/medium.json" --fast-report "$OUT/fast.json" --output-dir "$OUT" --tracked-summary "$ROOT/results/track2_icassp_motion_compression_v3_summary.json"
echo 'Completed corrected v3 analysis. Review decision/status, not just a headline R2.'
echo 'Commit results/track2_icassp_motion_compression_v3_summary.json; retain all v2 files.'
