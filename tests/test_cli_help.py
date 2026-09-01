from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "eval_openvla_libero_tasks.py",
    "extract_openvla_pair_hidden.py",
    "run_openvla_causal_utilization.py",
    "extract_xvla_robotwin_pairs.py",
    "collect_mu_vla_memory_trajectory.py",
    "eval_mu_vla_intervention.py",
    "extract_xvla_robotwin_hidden.py",
    "model_preflight.py",
    "smoke_mu_vla_mikasa.py",
    "smoke_openvla_load.py",
    "smoke_xvla_layers.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_heavy_cli_help_does_not_require_track_dependencies(script: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
