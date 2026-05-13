#!/usr/bin/env python3
"""Inspect BONES-SEED metadata without writing to the data directory."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_retarget.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
