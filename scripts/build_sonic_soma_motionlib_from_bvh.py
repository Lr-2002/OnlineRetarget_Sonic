#!/usr/bin/env python3
"""Build a SONIC-compatible SOMA motionlib from proportional BVH files.

The SONIC runtime expects ``soma_motion_file`` to be either a PKL file or a
directory containing one PKL per robot motion key.  Each PKL stores
``soma_joints``, ``soma_root_quat``, ``soma_transl``, and ``fps`` under the same
motion key as the robot motionlib.

This script intentionally stores SOMA at the BVH source FPS by default.  SONIC's
motion library then performs the same 50Hz resampling path used by the robot
motionlib, which keeps validation and training on the same physical timeline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Sequence


DATE_KEY_RE = re.compile(r"^(?P<date>\d{6})__(?P<name>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-motion-dir", type=Path, required=True)
    parser.add_argument("--source-bvh-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-repo",
        type=Path,
        required=True,
        help="SONIC checkout containing gear_sonic/data_process/extract_soma_joints_from_bvh.py",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument(
        "--write-missing-list",
        action="store_true",
        help="Write missing_bvh_keys.txt into the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot_motion_dir = args.robot_motion_dir.expanduser()
    source_bvh_root = args.source_bvh_root.expanduser()
    output_dir = args.output_dir.expanduser()
    source_repo = args.source_repo.expanduser()

    keys = load_robot_motion_keys(robot_motion_dir)
    if args.limit is not None:
        keys = keys[: args.limit]
    if not keys:
        raise SystemExit(f"no robot motion keys found in {robot_motion_dir}")
    if not source_bvh_root.exists():
        raise SystemExit(f"source BVH root does not exist: {source_bvh_root}")

    extractor_path = (
        source_repo
        / "gear_sonic"
        / "data_process"
        / "extract_soma_joints_from_bvh.py"
    )
    if not extractor_path.exists():
        raise SystemExit(f"missing SONIC SOMA extractor: {extractor_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    bvh_index = build_bvh_index(source_bvh_root, keys)
    jobs = []
    missing = []
    for key in keys:
        bvh_path = resolve_bvh_for_motion_key(key, source_bvh_root, bvh_index)
        if bvh_path is None:
            missing.append(key)
            continue
        out_path = output_dir / f"{key}.pkl"
        if args.skip_existing and out_path.exists():
            jobs.append((key, str(bvh_path), str(out_path), str(extractor_path), True))
        else:
            jobs.append((key, str(bvh_path), str(out_path), str(extractor_path), False))

    t0 = time.time()
    results = []
    workers = max(1, int(args.num_workers))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(convert_one, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    report = build_report(
        robot_motion_dir=robot_motion_dir,
        source_bvh_root=source_bvh_root,
        output_dir=output_dir,
        source_repo=source_repo,
        keys=keys,
        missing=missing,
        results=results,
        elapsed_sec=time.time() - t0,
    )
    write_json(output_dir / "soma_motionlib_report.json", report)
    if args.write_missing_list and missing:
        (output_dir / "missing_bvh_keys.txt").write_text(
            "\n".join(missing) + "\n", encoding="utf-8"
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_all and (missing or report["failed_count"]):
        return 2
    return 0


def load_robot_motion_keys(robot_motion_dir: Path) -> list[str]:
    metadata = robot_motion_dir / "metadata.pkl"
    if metadata.exists():
        joblib = _import_joblib()
        loaded = joblib.load(metadata)
        if isinstance(loaded, dict):
            return sorted(str(key) for key in loaded)
    return sorted(path.stem for path in robot_motion_dir.glob("*.pkl") if path.name != "metadata.pkl")


def build_bvh_index(source_bvh_root: Path, keys: Sequence[str]) -> dict[str, Path]:
    needs_index = any(DATE_KEY_RE.match(key) is None for key in keys)
    if not needs_index:
        return {}
    return {path.stem: path for path in source_bvh_root.rglob("*.bvh")}


def resolve_bvh_for_motion_key(
    key: str,
    source_bvh_root: Path,
    bvh_index: dict[str, Path],
) -> Path | None:
    for candidate in bvh_candidates_for_motion_key(key, source_bvh_root):
        if candidate.exists():
            return candidate
    return bvh_index.get(key)


def bvh_candidates_for_motion_key(key: str, source_bvh_root: Path) -> tuple[Path, ...]:
    match = DATE_KEY_RE.match(key)
    if match:
        return (source_bvh_root / match.group("date") / f"{match.group('name')}.bvh",)
    return (source_bvh_root / f"{key}.bvh",)


def convert_one(job: tuple[str, str, str, str, bool]) -> dict[str, Any]:
    key, bvh_path_text, out_path_text, extractor_path_text, skipped = job
    out_path = Path(out_path_text)
    if skipped:
        return {"key": key, "status": "skipped", "output_path": str(out_path)}
    try:
        extractor = _load_sonic_soma_extractor(Path(extractor_path_text))
        joblib = _import_joblib()
        np = _import_numpy()

        bvh_path = Path(bvh_path_text)
        joints, channel_order, motion_data, _n_frames, frame_time = extractor.parse_bvh(bvh_path)
        fps_source = round(1.0 / frame_time)
        positions, root_quats = extractor.compute_fk_selected(
            joints,
            channel_order,
            motion_data,
            extractor.SOMA_JOINTS,
        )

        positions_m = positions / 100.0
        hips_idx = 0
        transl = positions_m[:, hips_idx, :].copy()
        positions_m = positions_m - transl[:, None, :]
        positions_zup = positions_m.copy()
        positions_zup[..., 1] = -positions_m[..., 2]
        positions_zup[..., 2] = positions_m[..., 1]
        root_quats = root_quats[:, [3, 0, 1, 2]]

        payload = {
            key: {
                "soma_joints": positions_zup.astype(np.float32),
                "soma_root_quat": root_quats.astype(np.float32),
                "soma_transl": transl.astype(np.float32),
                "fps": int(fps_source),
                "joint_names": list(extractor.SOMA_JOINTS),
                "source_bvh": str(bvh_path),
            }
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, out_path)
        return {
            "key": key,
            "status": "converted",
            "output_path": str(out_path),
            "source_bvh": str(bvh_path),
            "fps": int(fps_source),
            "frames": int(positions_zup.shape[0]),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "key": key,
            "status": "failed",
            "output_path": str(out_path),
            "source_bvh": bvh_path_text,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_report(
    *,
    robot_motion_dir: Path,
    source_bvh_root: Path,
    output_dir: Path,
    source_repo: Path,
    keys: Sequence[str],
    missing: Sequence[str],
    results: Sequence[dict[str, Any]],
    elapsed_sec: float,
) -> dict[str, Any]:
    status_counts = Counter(str(result.get("status")) for result in results)
    fps_counts = Counter(str(result.get("fps")) for result in results if result.get("fps"))
    failed = [result for result in results if result.get("status") == "failed"]
    return {
        "robot_motion_dir": str(robot_motion_dir),
        "source_bvh_root": str(source_bvh_root),
        "output_dir": str(output_dir),
        "source_repo": str(source_repo),
        "requested_count": len(keys),
        "converted_count": int(status_counts.get("converted", 0)),
        "skipped_count": int(status_counts.get("skipped", 0)),
        "failed_count": int(status_counts.get("failed", 0)),
        "missing_bvh_count": len(missing),
        "status_counts": dict(sorted(status_counts.items())),
        "source_fps_counts": dict(sorted(fps_counts.items())),
        "missing_bvh_examples": list(missing[:20]),
        "failed_examples": failed[:20],
        "elapsed_sec": elapsed_sec,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_sonic_soma_extractor(path: Path):
    spec = importlib.util.spec_from_file_location("sonic_soma_extractor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load extractor module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _import_joblib():
    import joblib

    return joblib


def _import_numpy():
    import numpy

    return numpy


if __name__ == "__main__":
    raise SystemExit(main())
