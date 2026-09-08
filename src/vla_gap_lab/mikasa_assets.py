"""Pinned MIKASA/ManiSkill asset provenance for Track 2 parity."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

MU_VLA_YCB_REPO = "haosulab/ManiSkill2"
MU_VLA_YCB_REVISION = "4c134e2e33b11b1fa751c1d7b6afb2e89231b875"
MU_VLA_YCB_FILENAME = "data/mani_skill2_ycb.zip"
MU_VLA_YCB_SHA256 = "174001ba1003cc0c5adda6453f4433f55ec7e804f0f0da22d015d525d02262fb"
MU_VLA_YCB_TARGET_RELATIVE = Path("assets/mani_skill2_ycb")
PROVENANCE_FILENAME = ".vla_gap_asset_provenance.json"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_ycb_provenance() -> dict[str, str]:
    return {
        "repo_id": MU_VLA_YCB_REPO,
        "revision": MU_VLA_YCB_REVISION,
        "filename": MU_VLA_YCB_FILENAME,
        "archive_sha256": MU_VLA_YCB_SHA256,
    }


def ycb_asset_issues(provenance: dict[str, Any] | None) -> list[str]:
    expected = expected_ycb_provenance()
    if not provenance:
        return [
            "missing YCB asset provenance marker; reinstall the pinned Track-2 YCB asset"
        ]
    issues = []
    for key, value in expected.items():
        if provenance.get(key) != value:
            issues.append(
                f"YCB provenance {key} must be {value!r}, got {provenance.get(key)!r}"
            )
    return issues


def ycb_target_path(asset_dir: str | Path) -> Path:
    return Path(asset_dir) / MU_VLA_YCB_TARGET_RELATIVE


def read_ycb_asset_provenance(asset_dir: str | Path) -> dict[str, Any] | None:
    marker = ycb_target_path(asset_dir) / PROVENANCE_FILENAME
    if not marker.is_file():
        return None
    return json.loads(marker.read_text())


def assert_mu_vla_ycb_asset(asset_dir: str | Path) -> dict[str, Any]:
    target = ycb_target_path(asset_dir)
    if not target.is_dir():
        raise RuntimeError(
            f"pinned Track-2 YCB asset is missing at {target}; run scripts/install_track2_ycb_asset.py"
        )
    provenance = read_ycb_asset_provenance(asset_dir)
    issues = ycb_asset_issues(provenance)
    if issues:
        rendered = "\n - ".join(issues)
        raise RuntimeError(
            "incompatible YCB asset for released mu-VLA ShellGamePush parity:\n - "
            + rendered
        )
    assert provenance is not None
    return provenance


def _safe_extract(zip_path: Path, destination: Path) -> Path:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        if not members:
            raise RuntimeError("YCB archive is empty")
        for member in members:
            resolved = (destination / member.filename).resolve()
            if os.path.commonpath([destination, resolved]) != str(destination):
                raise RuntimeError(f"unsafe path in YCB archive: {member.filename}")
        archive.extractall(destination)
        top_levels = {
            Path(member.filename).parts[0]
            for member in members
            if Path(member.filename).parts
        }
    if len(top_levels) != 1:
        raise RuntimeError(
            f"expected one top-level directory in YCB archive, found {sorted(top_levels)}"
        )
    return destination / next(iter(top_levels))


def install_mu_vla_ycb_asset(
    archive_path: str | Path,
    *,
    asset_dir: str | Path,
) -> dict[str, Any]:
    """Install the exact YCB archive expected by ManiSkill 3.0.0b15.

    The caller is responsible for fetching ``archive_path`` from the pinned HF
    dataset revision.  Installation is transactional: extract into a temporary
    directory, then replace the active asset directory only after checksum and
    layout validation succeed.
    """
    archive = Path(archive_path)
    observed_sha = sha256_file(archive)
    if observed_sha != MU_VLA_YCB_SHA256:
        raise RuntimeError(
            "wrong YCB archive checksum: "
            f"expected {MU_VLA_YCB_SHA256}, got {observed_sha}"
        )
    asset_root = Path(asset_dir)
    target = ycb_target_path(asset_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vla-gap-ycb-") as temporary:
        extracted = _safe_extract(archive, Path(temporary))
        if extracted.name != target.name:
            raise RuntimeError(
                f"unexpected YCB archive root {extracted.name!r}; expected {target.name!r}"
            )
        staged = target.parent / f".{target.name}.staged"
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(extracted, staged)
        provenance = {
            **expected_ycb_provenance(),
            "installed_by": "vla-gap-lab",
        }
        (staged / PROVENANCE_FILENAME).write_text(json.dumps(provenance, indent=2) + "\n")
        if target.exists():
            shutil.rmtree(target)
        staged.replace(target)
    return provenance
