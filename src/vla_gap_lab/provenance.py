"""Capture compact, non-secret provenance for reproducible diagnostic runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(path: str | Path) -> dict[str, object]:
    path = Path(path).resolve()

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "path": str(path),
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current") or None,
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(path), "available": False}


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capture_provenance(
    repositories: list[Path], artifacts: list[Path], packages: list[str]
) -> dict[str, object]:
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"artifact files do not exist: {missing}")
    return {
        "schema_version": 1,
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "repositories": [git_state(path) for path in repositories],
        "artifacts": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in artifacts
        ],
        "packages": package_versions(packages),
    }
