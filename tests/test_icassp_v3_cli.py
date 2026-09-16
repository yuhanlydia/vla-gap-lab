import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('name', ['annotate_icassp_v3.py', 'probe_icassp_v3.py', 'summarize_icassp_v3.py'])
def test_v3_cli_help_without_gpu_or_simulator(name):
    path = ROOT / 'scripts' / name
    assert path.exists(), f'missing runnable {name}'
    process = subprocess.run([sys.executable, str(path), '--help'],
                             env={**os.environ, 'PYTHONPATH': str(ROOT/'src')},
                             text=True, capture_output=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert 'usage:' in process.stdout
