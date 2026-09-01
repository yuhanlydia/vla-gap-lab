#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
audit_root=$(mktemp -d /tmp/vla-gap-release-check.XXXXXX)
trap 'rm -rf -- "$audit_root"' EXIT

cd "$repository_root"
scripts/check.sh
uv build --wheel --out-dir "$audit_root/wheels" .
python3 -m venv "$audit_root/venv"
"$audit_root/venv/bin/pip" install --no-deps "$audit_root"/wheels/*.whl
(
  cd /tmp
  "$audit_root/venv/bin/python" -c \
    'import pathlib, vla_gap_lab; assert "site-packages" in str(pathlib.Path(vla_gap_lab.__file__))'
)
sha256sum "$audit_root"/wheels/*.whl
