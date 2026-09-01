from __future__ import annotations

import json

from vla_gap_lab.artifact_io import write_json_atomic


def test_atomic_json_replaces_complete_document(tmp_path) -> None:
    path = tmp_path / "result.json"
    write_json_atomic(path, {"episodes": [1]})
    write_json_atomic(path, {"episodes": [1, 2]})
    assert json.loads(path.read_text()) == {"episodes": [1, 2]}
    assert not list(tmp_path.glob("*.tmp"))
