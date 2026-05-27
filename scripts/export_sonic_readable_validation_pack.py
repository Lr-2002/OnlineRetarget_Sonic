#!/usr/bin/env python3
"""Render a readable soma-G1 validation pack from persisted SONIC trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from online_retarget.sonic_validation_export import (  # noqa: E402
    DEFAULT_READABLE_CLIP_INDICES,
    DEFAULT_READABLE_HEIGHT,
    DEFAULT_READABLE_WIDTH,
    VARIANT_NAMES,
    export_readable_validation_pack,
    parse_clip_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-group", required=True, help="Formal run group name.")
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path(
            "/mnt/data_cpfs/code/wxh/GR00T-WholeBodyControl-upstream-training/"
            "logs_rl/OnlineRetarget/manager/universal_token/all_modes"
        ),
        help="Root containing SONIC experiment directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Pack output directory. Defaults under OnlineRetarget "
            "outputs/readable_validation_packs."
        ),
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        default=[str(item) for item in DEFAULT_READABLE_CLIP_INDICES],
        help="Clip indices to export, e.g. --clips 0 6.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANT_NAMES),
        help="Variant names to require in the pack.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_READABLE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_READABLE_HEIGHT)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Write a partial manifest instead of exiting nonzero when raw trajectories "
            "are missing."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = REPO_ROOT / "outputs" / "readable_validation_packs" / args.run_group
    clips = parse_clip_indices(",".join(args.clips))
    result = export_readable_validation_pack(
        search_root=args.search_root,
        run_group=args.run_group,
        output_dir=output_dir,
        clips=clips,
        variants=tuple(args.variants),
        width=args.width,
        height=args.height,
        allow_missing=args.allow_missing,
    )
    payload = {
        "status": result.status,
        "output_dir": str(result.output_dir),
        "manifest_path": str(result.manifest_path),
        "videos_ok": result.videos_ok,
        "videos_failed": result.videos_failed,
        "missing": list(result.missing),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if result.status != "ok" and not args.allow_missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
