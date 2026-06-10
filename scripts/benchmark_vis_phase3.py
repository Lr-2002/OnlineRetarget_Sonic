#!/usr/bin/env python3
"""Phase 3 benchmark harness for VisPacket static/render adapters."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vis_benchmark.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
