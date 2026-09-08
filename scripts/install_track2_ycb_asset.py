#!/usr/bin/env python3
"""Install the exact ManiSkill b15 YCB archive used by Track-2 ShellGame parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_gap_lab.mikasa_assets import (
    MU_VLA_YCB_FILENAME,
    MU_VLA_YCB_REPO,
    MU_VLA_YCB_REVISION,
    install_mu_vla_ycb_asset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=None,
        help="Override ManiSkill ASSET_DIR (normally auto-detected).",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from mani_skill import ASSET_DIR

    asset_dir = args.asset_dir or Path(ASSET_DIR)
    archive = hf_hub_download(
        repo_id=MU_VLA_YCB_REPO,
        repo_type="dataset",
        filename=MU_VLA_YCB_FILENAME,
        revision=MU_VLA_YCB_REVISION,
    )
    provenance = install_mu_vla_ycb_asset(archive, asset_dir=asset_dir)
    print(
        json.dumps(
            {
                "asset_dir": str(asset_dir),
                "target": str(asset_dir / "assets" / "mani_skill2_ycb"),
                **provenance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
