#!/usr/bin/env python3
"""Execute one LR-272 BONES-SEED SOMA ablation candidate stage.

The runner consumes a candidate config, a stage CSV, and an output directory.
It writes candidate G1 CSVs, metric CSV/JSON, and lightweight visual artifacts.
The heavy Isaac mesh renderer is intentionally reported as a separate blocker
when this smoke runner is used without a renderer template.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import tarfile
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from online_retarget.data.bones_seed import G1_CSV_COLUMNS, G1_JOINT_COLUMNS
except Exception:  # noqa: BLE001
    G1_CSV_COLUMNS = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
        "left_hip_pitch_joint_dof",
        "left_hip_roll_joint_dof",
        "left_hip_yaw_joint_dof",
        "left_knee_joint_dof",
        "left_ankle_pitch_joint_dof",
        "left_ankle_roll_joint_dof",
        "right_hip_pitch_joint_dof",
        "right_hip_roll_joint_dof",
        "right_hip_yaw_joint_dof",
        "right_knee_joint_dof",
        "right_ankle_pitch_joint_dof",
        "right_ankle_roll_joint_dof",
        "waist_yaw_joint_dof",
        "waist_roll_joint_dof",
        "waist_pitch_joint_dof",
        "left_shoulder_pitch_joint_dof",
        "left_shoulder_roll_joint_dof",
        "left_shoulder_yaw_joint_dof",
        "left_elbow_joint_dof",
        "left_wrist_roll_joint_dof",
        "left_wrist_pitch_joint_dof",
        "left_wrist_yaw_joint_dof",
        "right_shoulder_pitch_joint_dof",
        "right_shoulder_roll_joint_dof",
        "right_shoulder_yaw_joint_dof",
        "right_elbow_joint_dof",
        "right_wrist_roll_joint_dof",
        "right_wrist_pitch_joint_dof",
        "right_wrist_yaw_joint_dof",
    ]
    G1_JOINT_COLUMNS = G1_CSV_COLUMNS[7:]


ROOT_POSITION_SCALE = 0.01
ANGLE_SCALE = math.pi / 180.0


@dataclass
class Motion:
    fps: float
    root_pos: np.ndarray
    root_euler: np.ndarray
    dof: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(min(len(self.root_pos), len(self.root_euler), len(self.dof)))

    def slice(self, n: int) -> "Motion":
        n = min(int(n), self.frame_count)
        return Motion(
            fps=self.fps,
            root_pos=self.root_pos[:n].copy(),
            root_euler=self.root_euler[:n].copy(),
            dof=self.dof[:n].copy(),
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("retarget", "metric", "visual", "all"), default="all")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = read_json(args.config)
    rows = read_stage_csv(args.stage_csv)
    if not rows:
        raise SystemExit(f"stage CSV has no rows: {args.stage_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "config": str(args.config),
        "stage_csv": str(args.stage_csv),
        "output_dir": str(args.output_dir),
        "mode": args.mode,
        "candidate_id": config["candidate"]["candidate_id"],
        "route": config["candidate"]["route"],
        "row_count": len(rows),
    }
    if args.mode in ("retarget", "all"):
        manifest["retarget"] = run_retarget(config, rows, args.output_dir, max_frames=args.max_frames)
    if args.mode in ("metric", "all"):
        manifest["metric"] = run_metric(config, rows, args.output_dir, max_frames=args.max_frames)
    if args.mode in ("visual", "all"):
        manifest["visual"] = run_visual(config, rows, args.output_dir, max_frames=args.max_frames)
    write_json(args.output_dir / "runner_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def run_retarget(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    *,
    max_frames: int,
) -> list[dict[str, Any]]:
    candidate = config["candidate"]
    out_dir = output_dir / "retarget_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for row in rows:
        key = row_key(row)
        official = load_official_motion(row, config)
        base = load_base_soma_motion(row, official.frame_count, official.fps)
        n = min(official.frame_count, base.frame_count)
        if max_frames > 0:
            n = min(n, max_frames)
        official = official.slice(n)
        base = base.slice(n)
        transformed, diagnostics = apply_candidate(base, official, candidate)
        csv_path = candidate_csv_path(output_dir, key)
        write_g1_csv(csv_path, transformed)
        results.append(
            {
                "key": key,
                "candidate_id": candidate["candidate_id"],
                "route": candidate["route"],
                "source_soma_online_npy": row.get("soma_online_npy", ""),
                "official_bones_g1_csv_member": row.get("official_bones_g1_csv_member", ""),
                "candidate_csv": str(csv_path),
                "frames": transformed.frame_count,
                "fps": transformed.fps,
                "diagnostics": diagnostics,
            }
        )
    write_json(output_dir / "retarget_manifest.json", {"rows": results})
    return results


def run_metric(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    *,
    max_frames: int,
) -> dict[str, Any]:
    metric_dir = output_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        key = row_key(row)
        csv_path = candidate_csv_path(output_dir, key)
        if not csv_path.exists():
            run_retarget(config, [row], output_dir, max_frames=max_frames)
        official = load_official_motion(row, config)
        candidate_motion = load_candidate_csv(csv_path, float(row.get("source_bvh_fps") or official.fps))
        n = min(official.frame_count, candidate_motion.frame_count)
        if max_frames > 0:
            n = min(n, max_frames)
        official = official.slice(n)
        candidate_motion = candidate_motion.slice(n)
        rec = {
            "key": key,
            "candidate_id": config["candidate"]["candidate_id"],
            "route": config["candidate"]["route"],
            **motion_metrics(candidate_motion, official),
        }
        records.append(rec)
    csv_path = metric_dir / "candidate_metrics.csv"
    write_metric_csv(csv_path, records)
    summary = summarize_records(records)
    payload = {
        "candidate_id": config["candidate"]["candidate_id"],
        "route": config["candidate"]["route"],
        "metric_csv": str(csv_path),
        "rows": records,
        "summary": summary,
        "metric_limitations": [
            "Smoke runner computes root and DoF metrics only; FK/contact acceptance remains with the full evaluator.",
            "Angles are compared in radians after CSV degrees-to-radians conversion.",
        ],
    }
    write_json(metric_dir / "candidate_metrics.json", payload)
    return payload


def run_visual(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    *,
    max_frames: int,
) -> dict[str, Any]:
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    visuals = []
    for row in rows:
        key = row_key(row)
        csv_path = candidate_csv_path(output_dir, key)
        if not csv_path.exists():
            run_retarget(config, [row], output_dir, max_frames=max_frames)
        official = load_official_motion(row, config)
        candidate_motion = load_candidate_csv(csv_path, float(row.get("source_bvh_fps") or official.fps))
        n = min(official.frame_count, candidate_motion.frame_count)
        if max_frames > 0:
            n = min(n, max_frames)
        official = official.slice(n)
        candidate_motion = candidate_motion.slice(n)
        svg = visual_dir / f"{safe_name(key)}_root_xy.svg"
        write_root_xy_svg(svg, official, candidate_motion, title=f"{key} official vs {config['candidate']['candidate_id']}")
        visuals.append({"key": key, "root_xy_svg": str(svg)})
    blocker = {
        "status": "blocked",
        "renderer": "IsaacLab mesh side-by-side",
        "reason": "This smoke runner produces lightweight SVG diagnostics. It does not launch IsaacLab/renderer templates.",
        "lightweight_visuals": visuals,
    }
    write_json(visual_dir / "isaac_mesh_renderer_blocker.json", blocker)
    payload = {
        "candidate_id": config["candidate"]["candidate_id"],
        "route": config["candidate"]["route"],
        "visuals": visuals,
        "renderer_blocker": str(visual_dir / "isaac_mesh_renderer_blocker.json"),
    }
    write_json(visual_dir / "visual_manifest.json", payload)
    return payload


def load_official_motion(row: Mapping[str, str], config: Mapping[str, Any]) -> Motion:
    tar_path = Path(row.get("official_bones_g1_tar") or config["provenance"]["inputs"]["g1_tar"])
    member_name = row.get("official_bones_g1_csv_member", "")
    if not member_name:
        raise FileNotFoundError("stage row missing official_bones_g1_csv_member")
    fps = float(row.get("source_bvh_fps") or 120.0048)
    with tarfile.open(tar_path, "r:*") as tar:
        member = tar.extractfile(member_name)
        if member is None:
            raise FileNotFoundError(f"{tar_path}::{member_name}")
        text_rows = list(csv.DictReader(line.decode("utf-8") for line in member))
    root_pos = np.asarray(
        [
            [
                float(item["root_translateX"]) * ROOT_POSITION_SCALE,
                float(item["root_translateY"]) * ROOT_POSITION_SCALE,
                float(item["root_translateZ"]) * ROOT_POSITION_SCALE,
            ]
            for item in text_rows
        ],
        dtype=np.float64,
    )
    root_euler = np.asarray(
        [
            [
                float(item["root_rotateX"]) * ANGLE_SCALE,
                float(item["root_rotateY"]) * ANGLE_SCALE,
                float(item["root_rotateZ"]) * ANGLE_SCALE,
            ]
            for item in text_rows
        ],
        dtype=np.float64,
    )
    dof = np.asarray([[float(item[column]) * ANGLE_SCALE for column in G1_JOINT_COLUMNS] for item in text_rows], dtype=np.float64)
    return Motion(fps=fps, root_pos=root_pos, root_euler=root_euler, dof=dof)


def load_base_soma_motion(row: Mapping[str, str], frames: int, fps: float) -> Motion:
    path = Path(row.get("soma_online_npy", ""))
    if not path.exists():
        raise FileNotFoundError(f"missing soma_online_npy for {row_key(row)}: {path}")
    arr = np.load(path).astype(np.float64)
    arr = arr[:frames]
    if arr.ndim != 2 or arr.shape[1] < 36:
        raise ValueError(f"expected qpos array with at least 36 columns: {path} got {arr.shape}")
    root_pos = arr[:, 0:3]
    root_euler = quat_xyzw_to_euler_xyz(normalize_quat(arr[:, 3:7]))
    dof = arr[:, 7:36]
    return Motion(fps=fps, root_pos=root_pos, root_euler=root_euler, dof=dof)


def load_candidate_csv(path: Path, fps: float) -> Motion:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    root_pos = np.asarray(
        [
            [
                float(item["root_translateX"]) * ROOT_POSITION_SCALE,
                float(item["root_translateY"]) * ROOT_POSITION_SCALE,
                float(item["root_translateZ"]) * ROOT_POSITION_SCALE,
            ]
            for item in rows
        ],
        dtype=np.float64,
    )
    root_euler = np.asarray(
        [
            [
                float(item["root_rotateX"]) * ANGLE_SCALE,
                float(item["root_rotateY"]) * ANGLE_SCALE,
                float(item["root_rotateZ"]) * ANGLE_SCALE,
            ]
            for item in rows
        ],
        dtype=np.float64,
    )
    dof = np.asarray([[float(item[column]) * ANGLE_SCALE for column in G1_JOINT_COLUMNS] for item in rows], dtype=np.float64)
    return Motion(fps=fps, root_pos=root_pos, root_euler=root_euler, dof=dof)


def apply_candidate(base: Motion, official: Motion, candidate: Mapping[str, Any]) -> tuple[Motion, dict[str, Any]]:
    out = Motion(
        fps=base.fps,
        root_pos=base.root_pos.copy(),
        root_euler=base.root_euler.copy(),
        dof=base.dof.copy(),
    )
    diagnostics: dict[str, Any] = {"candidate_id": candidate["candidate_id"], "route": candidate["route"]}
    root_world = candidate.get("root_world", {})
    mode = root_world.get("xy_scale_mode")
    if mode == "global":
        scale = float(root_world.get("xy_scale", 1.0))
        out.root_pos[:, :2] = out.root_pos[:1, :2] + (out.root_pos[:, :2] - out.root_pos[:1, :2]) * scale
        diagnostics["xy_scale_applied"] = scale
    elif mode == "per_clip_bestfit_xy":
        pred = out.root_pos[:, :2] - out.root_pos[:1, :2]
        ref = official.root_pos[:, :2] - official.root_pos[:1, :2]
        denom = float(np.sum(pred * pred))
        scale = float(np.sum(pred * ref) / denom) if denom > 1e-12 else 1.0
        out.root_pos[:, :2] = out.root_pos[:1, :2] + pred * scale
        diagnostics["xy_scale_applied"] = scale
        diagnostics["diagnostic_only"] = True

    yaw_alignment = root_world.get("yaw_alignment")
    if yaw_alignment == "first_frame_forward_heading":
        delta = float(angle_diff_scalar(official.root_euler[0, 2], out.root_euler[0, 2]))
        rotate_root_xy(out, delta)
        out.root_euler[:, 2] += delta
        diagnostics["yaw_delta_rad"] = delta
    elif yaw_alignment == "early_velocity_heading":
        delta = velocity_heading_delta(out.root_pos, official.root_pos)
        rotate_root_xy(out, delta)
        out.root_euler[:, 2] += delta
        diagnostics["yaw_delta_rad"] = delta

    dof_cfg = candidate.get("dof_convention", {})
    signs = dof_cfg.get("sign_overrides") or {}
    for joint, sign in signs.items():
        idx = joint_index(joint)
        if idx is not None:
            out.dof[:, idx] *= float(sign)
    if signs:
        diagnostics["sign_overrides_applied"] = signs

    swaps = dof_cfg.get("axis_swaps") or {}
    seen: set[tuple[int, int]] = set()
    for left, right in swaps.items():
        li = joint_index(left)
        ri = joint_index(right)
        if li is None or ri is None:
            continue
        pair = tuple(sorted((li, ri)))
        if pair in seen:
            continue
        out.dof[:, [li, ri]] = out.dof[:, [ri, li]]
        seen.add(pair)
    if seen:
        diagnostics["axis_swaps_applied"] = [list(item) for item in sorted(seen)]

    if candidate["route"] == "B_summarizer_preprocess":
        diagnostics["preprocess_contract_checks"] = preprocess_contract_checks(base, official, candidate)
    return out, diagnostics


def preprocess_contract_checks(base: Motion, official: Motion, candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fps_delta": abs(base.fps - official.fps),
        "frame_count_delta": abs(base.frame_count - official.frame_count),
        "raw_action_contract": candidate.get("summarizer", {}).get("raw_action_contract", ""),
        "per_clip_skeleton_requested": bool(candidate.get("summarizer", {}).get("per_clip_skeleton")),
        "pre_roll_frames": candidate.get("summarizer", {}).get("pre_roll_frames", 0),
        "stabilization_frames": candidate.get("summarizer", {}).get("stabilization_frames", 0),
    }


def motion_metrics(pred: Motion, ref: Motion) -> dict[str, float | int | str]:
    n = min(pred.frame_count, ref.frame_count)
    root_pred = pred.root_pos[:n]
    root_ref = ref.root_pos[:n]
    dof_delta = angle_diff(pred.dof[:n], ref.dof[:n])
    root_rot_delta = angle_diff(pred.root_euler[:n], ref.root_euler[:n])
    root_pred_rel = root_pred - root_pred[:1]
    root_ref_rel = root_ref - root_ref[:1]
    return {
        "aligned_frames": int(n),
        "duration_delta_sec": abs((pred.frame_count / pred.fps) - (ref.frame_count / ref.fps)),
        "root_abs_initial_delta_m": float(np.linalg.norm(root_pred[0] - root_ref[0])),
        "root_rel_rmse_m": float(np.sqrt(np.mean((root_pred_rel - root_ref_rel) ** 2))),
        "root_rel_max_m": float(np.max(np.linalg.norm(root_pred_rel - root_ref_rel, axis=1))),
        "root_z_rmse_m": float(np.sqrt(np.mean((root_pred[:, 2] - root_ref[:, 2]) ** 2))),
        "root_rot_rmse_rad": float(np.sqrt(np.mean(root_rot_delta**2))),
        "root_rot_max_rad": float(np.max(np.linalg.norm(root_rot_delta, axis=1))),
        "dof_rmse_rad": float(np.sqrt(np.mean(dof_delta**2))),
        "dof_mae_rad": float(np.mean(np.abs(dof_delta))),
        "dof_max_abs_rad": float(np.max(np.abs(dof_delta))),
        "root_xy_span_pred_m": root_xy_span(root_pred),
        "root_xy_span_ref_m": root_xy_span(root_ref),
        "root_xy_span_ratio": root_xy_span(root_pred) / max(root_xy_span(root_ref), 1e-9),
        "fk_contact_status": "not_computed_by_smoke_runner",
    }


def write_g1_csv(path: Path, motion: Motion) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=G1_CSV_COLUMNS)
        writer.writeheader()
        for idx in range(motion.frame_count):
            row: dict[str, float | int] = {
                "Frame": idx,
                "root_translateX": float(motion.root_pos[idx, 0] / ROOT_POSITION_SCALE),
                "root_translateY": float(motion.root_pos[idx, 1] / ROOT_POSITION_SCALE),
                "root_translateZ": float(motion.root_pos[idx, 2] / ROOT_POSITION_SCALE),
                "root_rotateX": float(motion.root_euler[idx, 0] / ANGLE_SCALE),
                "root_rotateY": float(motion.root_euler[idx, 1] / ANGLE_SCALE),
                "root_rotateZ": float(motion.root_euler[idx, 2] / ANGLE_SCALE),
            }
            for j, column in enumerate(G1_JOINT_COLUMNS):
                row[column] = float(motion.dof[idx, j] / ANGLE_SCALE)
            writer.writerow(row)


def write_metric_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [
        "root_rel_rmse_m",
        "root_rot_rmse_rad",
        "dof_rmse_rad",
        "dof_max_abs_rad",
        "root_xy_span_ratio",
    ]
    out: dict[str, Any] = {}
    for metric in metrics:
        vals = [float(row[metric]) for row in records if row.get(metric) not in (None, "")]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[metric] = {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "max": float(np.max(arr)),
            }
    return out


def write_root_xy_svg(path: Path, ref: Motion, pred: Motion, *, title: str) -> None:
    width, height, pad = 720, 520, 36
    ref_xy = ref.root_pos[:, :2] - ref.root_pos[:1, :2]
    pred_xy = pred.root_pos[:, :2] - pred.root_pos[:1, :2]
    points = np.concatenate([ref_xy, pred_xy], axis=0)
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)

    def project(arr: np.ndarray) -> list[tuple[float, float]]:
        x = pad + (arr[:, 0] - min_xy[0]) / span[0] * (width - 2 * pad)
        y = height - pad - (arr[:, 1] - min_xy[1]) / span[1] * (height - 2 * pad)
        return [(float(a), float(b)) for a, b in zip(x, y)]

    def polyline(coords: list[tuple[float, float]]) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)

    ref_line = polyline(project(ref_xy))
    pred_line = polyline(project(pred_xy))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                f'<text x="{pad}" y="24" font-family="monospace" font-size="14">{escape_xml(title)}</text>',
                f'<polyline points="{ref_line}" fill="none" stroke="#2354a6" stroke-width="2"/>',
                f'<polyline points="{pred_line}" fill="none" stroke="#d05a2a" stroke-width="2"/>',
                f'<circle cx="{project(ref_xy[:1])[0][0]:.2f}" cy="{project(ref_xy[:1])[0][1]:.2f}" r="4" fill="#2354a6"/>',
                f'<circle cx="{project(pred_xy[:1])[0][0]:.2f}" cy="{project(pred_xy[:1])[0][1]:.2f}" r="4" fill="#d05a2a"/>',
                f'<text x="{pad}" y="{height - 14}" font-family="monospace" font-size="12" fill="#2354a6">official BONES G1 root XY</text>',
                f'<text x="{pad + 260}" y="{height - 14}" font-family="monospace" font-size="12" fill="#d05a2a">candidate root XY</text>',
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def candidate_csv_path(output_dir: Path, key: str) -> Path:
    return output_dir / "retarget_csv" / f"{safe_name(key)}.csv"


def row_key(row: Mapping[str, str]) -> str:
    for column in ("lr271_key", "motion_key", "key", "clip_key", "bones_rel_key"):
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return Path(str(row.get("source_bvh", "clip"))).stem


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "clip"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_stage_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(n == 0.0, 1.0, n)


def quat_xyzw_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    q = normalize_quat(q)
    x, y, z, w = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return np.arctan2(np.sin(d), np.cos(d))


def angle_diff_scalar(a: float, b: float) -> float:
    return float(math.atan2(math.sin(a - b), math.cos(a - b)))


def rotate_root_xy(motion: Motion, yaw_delta: float) -> None:
    rel = motion.root_pos[:, :2] - motion.root_pos[:1, :2]
    c, s = math.cos(yaw_delta), math.sin(yaw_delta)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    motion.root_pos[:, :2] = motion.root_pos[:1, :2] + rel @ rot.T


def velocity_heading_delta(pred_pos: np.ndarray, ref_pos: np.ndarray) -> float:
    n = min(len(pred_pos), len(ref_pos), 30)
    if n <= 2:
        return 0.0
    pred_vec = pred_pos[n - 1, :2] - pred_pos[0, :2]
    ref_vec = ref_pos[n - 1, :2] - ref_pos[0, :2]
    pred_yaw = math.atan2(float(pred_vec[1]), float(pred_vec[0]))
    ref_yaw = math.atan2(float(ref_vec[1]), float(ref_vec[0]))
    return angle_diff_scalar(ref_yaw, pred_yaw)


def joint_index(joint_name: str) -> int | None:
    target = joint_name[:-4] if joint_name.endswith("_dof") else joint_name
    for idx, column in enumerate(G1_JOINT_COLUMNS):
        name = column[:-4] if column.endswith("_dof") else column
        if name == target:
            return idx
    return None


def root_xy_span(root_pos: np.ndarray) -> float:
    rel = root_pos[:, :2] - root_pos[:1, :2]
    return float(np.max(np.linalg.norm(rel, axis=1))) if len(rel) else 0.0


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
