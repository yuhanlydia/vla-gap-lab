#!/usr/bin/env python3
"""Fail fast unless Track 2 uses the released mu-VLA Transformers fork."""

from __future__ import annotations

import json

from vla_gap_lab.mu_vla_protocol import assert_mu_vla_runtime


def main() -> None:
    print(json.dumps(assert_mu_vla_runtime(), indent=2))


if __name__ == "__main__":
    main()
