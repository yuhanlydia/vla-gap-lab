#!/usr/bin/env bash
set -euo pipefail

ruff check src scripts tests
pytest -q
git diff --check
PYTHONPATH=src python3 scripts/validate_results_manifest.py >/dev/null
