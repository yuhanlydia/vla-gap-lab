#!/usr/bin/env bash
set -euo pipefail

ruff check src scripts tests
pytest -q
git diff --check
