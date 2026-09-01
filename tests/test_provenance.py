from __future__ import annotations

import hashlib

import pytest

from vla_gap_lab.provenance import capture_provenance, sha256_file


def test_sha256_and_capture_artifact(tmp_path) -> None:
    artifact = tmp_path / "input.bin"
    artifact.write_bytes(b"diagnostic input")
    assert sha256_file(artifact) == hashlib.sha256(b"diagnostic input").hexdigest()
    report = capture_provenance([], [artifact], ["package-that-cannot-exist-xyz"])
    assert report["artifacts"][0]["bytes"] == 16
    assert report["packages"]["package-that-cannot-exist-xyz"] is None


def test_capture_rejects_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="do not exist"):
        capture_provenance([], [tmp_path / "missing"], [])
