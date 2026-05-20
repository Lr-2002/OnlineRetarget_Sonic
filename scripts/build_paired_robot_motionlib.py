#!/usr/bin/env python3
"""Build a robot motionlib metadata directory paired to a reference motionlib.

SONIC's G1 robot motionlib and the OnlineRetarget SOMA motionlib must expose
the same motion keys for paired retarget training.  This script filters an
existing robot metadata.pkl to the keys available in a reference directory,
usually ``outputs/sonic_motionlib/soma_filtered_v1``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion-dir", type=Path, required=True)
    parser.add_argument("--reference-motion-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-all-reference-keys-in-robot",
        action="store_true",
        help="Fail if any reference key is not present in the robot metadata.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot_motion_dir = args.robot_motion_dir.expanduser()
    reference_motion_dir = args.reference_motion_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    metadata_path = robot_motion_dir / "metadata.pkl"
    if not metadata_path.exists():
        raise SystemExit(f"robot metadata does not exist: {metadata_path}")
    if not reference_motion_dir.exists():
        raise SystemExit(f"reference motionlib does not exist: {reference_motion_dir}")

    joblib = _import_joblib()
    metadata = joblib.load(metadata_path)
    if not isinstance(metadata, dict):
        raise SystemExit(f"robot metadata must be a dict: {metadata_path}")

    robot_keys = set(str(key) for key in metadata)
    reference_keys = load_motion_keys(reference_motion_dir)
    paired_keys = sorted(robot_keys & reference_keys)
    missing_in_robot = sorted(reference_keys - robot_keys)
    dropped_robot_keys = sorted(robot_keys - reference_keys)
    if args.require_all_reference_keys_in_robot and missing_in_robot:
        raise SystemExit(
            "reference motionlib contains keys missing from robot metadata: "
            + ", ".join(missing_in_robot[:20])
        )
    if not paired_keys:
        raise SystemExit("no paired motion keys found")

    output_dir.mkdir(parents=True, exist_ok=True)
    paired_metadata = {key: metadata[key] for key in paired_keys}
    joblib.dump(paired_metadata, output_dir / "metadata.pkl")
    (output_dir / "keys.txt").write_text("\n".join(paired_keys) + "\n", encoding="utf-8")

    report = {
        "robot_motion_dir": str(robot_motion_dir),
        "reference_motion_dir": str(reference_motion_dir),
        "output_dir": str(output_dir),
        "robot_key_count": len(robot_keys),
        "reference_key_count": len(reference_keys),
        "paired_key_count": len(paired_keys),
        "dropped_robot_key_count": len(dropped_robot_keys),
        "missing_in_robot_count": len(missing_in_robot),
        "dropped_robot_key_examples": dropped_robot_keys[:20],
        "missing_in_robot_examples": missing_in_robot[:20],
    }
    (output_dir / "paired_robot_motionlib_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def load_motion_keys(path: Path) -> set[str]:
    if path.is_dir():
        metadata_path = path / "metadata.pkl"
        if metadata_path.exists():
            metadata = _import_joblib().load(metadata_path)
            if isinstance(metadata, dict):
                return set(str(key) for key in metadata)
        return {item.stem for item in path.glob("*.pkl") if item.name != "metadata.pkl"}
    if path.is_file() and path.suffix == ".pkl":
        payload = _import_joblib().load(path)
        if isinstance(payload, dict):
            return set(str(key) for key in payload)
    return set()


def _import_joblib() -> Any:
    import joblib

    return joblib


if __name__ == "__main__":
    raise SystemExit(main())
