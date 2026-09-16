#!/usr/bin/env bash
# v2 is provisional; the default entrypoint now runs the audited v3 correction.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 0 ]]; then
  echo 'v3 replays existing actions without loading a VLA; the old checkpoint argument is unused.' >&2
fi
exec bash "$ROOT/scripts/run_track2_icassp_v3.sh"
