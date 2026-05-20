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
    results = []
    missing = []
    for key in keys:
        bvh_path = resolve_bvh_for_motion_key(key, source_bvh_root, bvh_index)
        if bvh_path is None:
            missing.append(key)
            continue
        out_path = output_dir / f"{key}.pkl"
        if args.skip_existing and out_path.exists():
            results.append(
                {
                    "key": key,
                    "status": "skipped",
                    "output_path": str(out_path),
                    "source_bvh": str(bvh_path),
                }
            )
        else:
            jobs.append((key, str(bvh_path), str(out_path), str(extractor_path), False))

    t0 = time.time()
    workers = max(1, int(args.num_workers))
    if jobs:
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
        joints, channel_order, motion_data, _n_frames, frame_time, parse_stats = parse_bvh_sanitized(
            bvh_path,
            np,
        )
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
                "parse_stats": parse_stats,
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
            "parse_stats": parse_stats,
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


def parse_bvh_sanitized(filepath: Path, np: Any):
    """Parse BVH while tolerating known source-file corruption.

    A small number of proportional BVH files contain NUL bytes in motion rows
    or advertise one more frame than the file actually stores.  We normalize
    those issues before handing the arrays to SONIC's FK routine so the
    generated motionlib still follows the SONIC representation contract.
    """

    raw = filepath.read_bytes()
    nul_bytes = raw.count(b"\x00")
    text = raw.decode("utf-8", errors="replace")
    if nul_bytes:
        text = text.replace("\x00", " ")
    lines = text.splitlines()

    joints = []
    joint_stack = []
    channel_order = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "MOTION":
            i += 1
            break

        match = re.match(r"(ROOT|JOINT)\s+(\S+)", line)
        if match:
            name = match.group(2)
            parent_idx = joint_stack[-1] if joint_stack else -1
            joints.append({"name": name, "offset": None, "channels": [], "parent_idx": parent_idx})
            joint_stack.append(len(joints) - 1)
        elif line.startswith("OFFSET") and joint_stack:
            vals = [_safe_float(token) for token in line.split()[1:4]]
            while len(vals) < 3:
                vals.append(0.0)
            joints[joint_stack[-1]]["offset"] = np.asarray(vals, dtype=np.float64)
        elif line.startswith("CHANNELS") and joint_stack:
            parts = line.split()
            n_ch = int(parts[1])
            ch_names = parts[2 : 2 + n_ch]
            joints[joint_stack[-1]]["channels"] = ch_names
            for ch in ch_names:
                channel_order.append((joint_stack[-1], ch))
        elif line == "}":
            if joint_stack:
                joint_stack.pop()
        i += 1

    if i >= len(lines):
        raise ValueError(f"BVH motion section is missing in {filepath}")

    frames_line = lines[i].strip()
    if not frames_line.startswith("Frames:"):
        raise ValueError(f"BVH frames line is malformed in {filepath}: {frames_line!r}")
    declared_frames = int(frames_line.split(":", 1)[1])
    i += 1

    if i >= len(lines):
        raise ValueError(f"BVH frame-time line is missing in {filepath}")
    frame_time_line = lines[i].strip()
    if not frame_time_line.startswith("Frame Time:"):
        raise ValueError(f"BVH frame-time line is malformed in {filepath}: {frame_time_line!r}")
    frame_time = float(frame_time_line.split(":", 1)[1])
    i += 1

    channel_count = len(channel_order)
    rows = []
    bad_float_count = 0
    padded_rows = 0
    truncated_rows = 0
    available_lines = lines[i : i + declared_frames]
    previous_row = [0.0] * channel_count
    for line in available_lines:
        tokens = line.strip().split()
        if len(tokens) < channel_count:
            padded_rows += 1
        elif len(tokens) > channel_count:
            truncated_rows += 1
        row = []
        for col in range(channel_count):
            if col < len(tokens):
                value = _safe_float(tokens[col])
                if value != value:  # NaN without importing math in worker hot path.
                    value = previous_row[col]
                    bad_float_count += 1
            else:
                value = previous_row[col]
                bad_float_count += 1
            row.append(value)
        previous_row = row
        rows.append(row)

    if not rows:
        raise ValueError(f"BVH has no parseable motion rows in {filepath}")

    motion_data = np.asarray(rows, dtype=np.float64)
    nonfinite_mask = ~np.isfinite(motion_data)
    nonfinite_count = int(nonfinite_mask.sum())
    if nonfinite_count:
        motion_data = np.nan_to_num(motion_data, nan=0.0, posinf=0.0, neginf=0.0)

    parse_stats = {
        "declared_frames": int(declared_frames),
        "parsed_frames": int(motion_data.shape[0]),
        "channel_count": int(channel_count),
        "nul_bytes": int(nul_bytes),
        "bad_float_count": int(bad_float_count),
        "nonfinite_count": nonfinite_count,
        "padded_rows": int(padded_rows),
        "truncated_rows": int(truncated_rows),
    }
    return joints, channel_order, motion_data, int(motion_data.shape[0]), frame_time, parse_stats


def _safe_float(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        return float("nan")


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
