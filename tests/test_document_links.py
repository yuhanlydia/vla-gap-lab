from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


@pytest.mark.parametrize("document", [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))])
def test_relative_markdown_links_exist(document: Path) -> None:
    missing = []
    for target in LINK.findall(document.read_text()):
        target = target.split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (document.parent / target).resolve()
        if not resolved.exists():
            missing.append(target)
    assert not missing, f"{document.relative_to(ROOT)} has missing links: {missing}"
