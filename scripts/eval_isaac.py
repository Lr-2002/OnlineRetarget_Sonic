#!/usr/bin/env python3
"""Isaac Lab evaluation scaffold for G1 retargeted references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default="isaac_g1_eval")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_root / "isaac_eval" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "index_csv": str(args.index_csv),
        "predictions": str(args.predictions) if args.predictions else "",
        "output_dir": str(output_dir),
        "metrics": [
            "success_rate",
            "fall_rate",
            "episode_length",
            "world_mpjpe",
            "root_relative_mpjpe",
            "foot_slide",
            "foot_float",
            "ground_penetration",
            "self_collision",
        ],
    }

    if args.dry_run:
        payload["status"] = "dry_run"
        _write_status(output_dir, payload)
        return

    try:
        import isaaclab  # noqa: F401
    except ImportError as exc:
        payload["status"] = "blocked"
        payload["blocked"] = "Isaac Lab is required for simulator evaluation"
        _write_status(output_dir, payload)
        raise SystemExit(payload["blocked"]) from exc

    payload["status"] = "blocked"
    payload["blocked"] = (
        "Isaac Lab import succeeded, but G1 replay/tracking task binding is not implemented yet."
    )
    _write_status(output_dir, payload)
    raise SystemExit(payload["blocked"])


def _write_status(output_dir: Path, payload: dict[str, object]) -> None:
    path = output_dir / "isaac_eval_status.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
