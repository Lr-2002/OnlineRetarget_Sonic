#!/usr/bin/env python3
"""CLI wrapper for LR-239 Isaac/SRC replay packet export."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_retarget.isaac_src_replay import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
