#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIKASA="$ROOT/external/MIKASA-Robo"
CKPT="${1:-$ROOT/models/mu-vla-m64-k2}"
MED_DIR="$ROOT/artifacts/mikasa/icassp_motion_compression_medium_n60"
FAST_DIR="$ROOT/artifacts/mikasa/icassp_motion_compression_fast_n60"
REPORT_DIR="$ROOT/artifacts/reports/icassp_motion_compression"

mkdir -p "$MED_DIR" "$FAST_DIR" "$REPORT_DIR"

cd "$MIKASA"
uv sync --frozen
uv pip install -r "$ROOT/requirements/track2-extra.txt" --python .venv/bin/python
PYTHONPATH="$ROOT/src" uv run python "$ROOT/scripts/check_mu_vla_runtime.py"

PYTHONPATH="$ROOT/src" uv run python "$ROOT/scripts/collect_mu_vla_state_motion_icassp.py" \
  --checkpoint "$CKPT" --task InterceptMedium-VLA-v0 \
  --episodes 60 --start-seed 4243224242 --precision 4bit --resume \
  --output-dir "$MED_DIR"

PYTHONPATH="$ROOT/src" uv run python "$ROOT/scripts/collect_mu_vla_state_motion_icassp.py" \
  --checkpoint "$CKPT" --task InterceptFast-VLA-v0 \
  --episodes 60 --start-seed 4243324242 --precision 4bit --resume \
  --output-dir "$FAST_DIR"

PYTHONPATH="$ROOT/src" uv run python "$ROOT/scripts/probe_mu_vla_state_motion_icassp.py" \
  --episodes-dir "$MED_DIR" --task InterceptMedium-VLA-v0 \
  --output "$REPORT_DIR/medium.json"

PYTHONPATH="$ROOT/src" uv run python "$ROOT/scripts/probe_mu_vla_state_motion_icassp.py" \
  --episodes-dir "$FAST_DIR" --task InterceptFast-VLA-v0 \
  --output "$REPORT_DIR/fast.json"

cd "$ROOT"
PYTHONPATH="$ROOT/src" "$MIKASA/.venv/bin/python" "$ROOT/scripts/summarize_icassp_state_motion.py" \
  --medium-report "$REPORT_DIR/medium.json" \
  --fast-report "$REPORT_DIR/fast.json" \
  --output-dir "$REPORT_DIR" \
  --tracked-summary "$ROOT/results/track2_icassp_motion_compression_summary.json"

echo "ICASSP motion-compression study complete."
echo "Commit and push results/track2_icassp_motion_compression_summary.json for review."
