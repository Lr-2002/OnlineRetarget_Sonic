#!/usr/bin/env python3
"""Execute one LR-272 BONES-SEED SOMA ablation candidate stage.

The runner consumes a candidate config, a stage CSV, and an output directory.
It writes candidate G1 CSVs, full evaluator metric CSV/JSON, and visual
artifacts or a concrete Isaac renderer blocker.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
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

try:
    from online_retarget.data.g1_quality import (
        DEFAULT_FOOT_BODIES,
        G1QualityConfig,
        g1_fk_body_positions,
        load_g1_kinematic_model,
    )
    from online_retarget.data import g1_quality as g1_quality_module
except Exception as exc:  # noqa: BLE001
    DEFAULT_FOOT_BODIES = (
        "left_ankle_roll_link",
        "left_toe_link",
        "right_ankle_roll_link",
        "right_toe_link",
    )
    G1QualityConfig = None  # type: ignore[assignment]
    g1_fk_body_positions = None  # type: ignore[assignment]
    load_g1_kinematic_model = None  # type: ignore[assignment]
    g1_quality_module = None  # type: ignore[assignment]
    G1_QUALITY_IMPORT_ERROR = repr(exc)
else:
    G1_QUALITY_IMPORT_ERROR = ""


ROOT_POSITION_SCALE = 0.01
ANGLE_SCALE = math.pi / 180.0
DEFAULT_G1_MODEL_XMLS = (
    Path("/home/user/repos/GR00T-WholeBodyControl-upstream-training/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"),
    Path("/home/user/project/GR00T-WholeBodyControl-upstream-training/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"),
    Path("/home/user/project/OnlineRetarget/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"),
)
DEFAULT_G1_USD = Path("/home/user/project/OnlineRetarget/runs/isaaclab_urdf_cache/g1_main/main.usd")
ACCEPTANCE_THRESHOLDS = {
    "root_rel_rmse_m": 0.05,
    "root_rot_geodesic_rmse_rad": 0.15,
    "fk_rootrel_mpjpe_m": 0.05,
    "dof_rmse_rad": 0.15,
    "contact_frame_ratio_delta": 0.15,
    "contact_slide_delta_mps": 0.50,
}


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
    parser.add_argument("--model-xml", type=Path, default=None, help="G1 MJCF XML for FK/contact metrics.")
    parser.add_argument("--robot-usd", type=Path, default=None, help="G1 USD for IsaacLab mesh rendering.")
    parser.add_argument("--render-isaac", action="store_true", help="Attempt IsaacLab mesh rendering during visual mode.")
    parser.add_argument("--render-timeout-sec", type=int, default=900)
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
        manifest["metric"] = run_metric(config, rows, args.output_dir, max_frames=args.max_frames, model_xml=args.model_xml)
    if args.mode in ("visual", "all"):
        manifest["visual"] = run_visual(
            config,
            rows,
            args.output_dir,
            max_frames=args.max_frames,
            robot_usd=args.robot_usd,
            render_isaac=args.render_isaac,
            render_timeout_sec=args.render_timeout_sec,
        )
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
    calibration = resolve_candidate_calibration(config, rows, output_dir, max_frames=max_frames)
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
        transformed, diagnostics = apply_candidate(base, official, candidate, calibration=calibration, row=row)
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
                "eval_split": split_value(row),
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
    model_xml: Path | None = None,
) -> dict[str, Any]:
    metric_dir = output_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    records = []
    evaluator = load_full_evaluator(model_xml)
    if evaluator.get("status") != "ok":
        write_json(metric_dir / "full_evaluator_blocker.json", public_evaluator_report(evaluator))
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
        full_metrics = full_evaluator_metrics(candidate_motion, official, evaluator)
        rec = {
            "key": key,
            "candidate_id": config["candidate"]["candidate_id"],
            "route": config["candidate"]["route"],
            "eval_split": split_value(row),
            **motion_metrics(candidate_motion, official),
            **full_metrics,
        }
        rec["metric_threshold_pass"] = metric_threshold_pass(rec) if rec.get("full_evaluator_status") == "ok" else False
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
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "full_evaluator": public_evaluator_report(evaluator),
        "metric_notes": [
            "Eval rows are tagged with their pairing split; train-split calibration candidates must not fit eval rows.",
            "Angles are compared in radians after CSV degrees-to-radians conversion.",
            "FK/contact fields are computed from the configured G1 MJCF when available; otherwise full_evaluator_blocker.json explains the missing dependency.",
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
    robot_usd: Path | None = None,
    render_isaac: bool = False,
    render_timeout_sec: int = 900,
) -> dict[str, Any]:
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    resolved_robot_usd = robot_usd or DEFAULT_G1_USD
    render_script = REPO_ROOT / "scripts" / "render_g1_isaac_pair.py"
    visuals = []
    isaac_reports = []
    command_lines = []
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
        mp4 = visual_dir / f"{safe_name(key)}_isaac_g1_mesh.mp4"
        command = build_isaac_render_command(
            row,
            candidate_csv=csv_path,
            output_mp4=mp4,
            robot_usd=resolved_robot_usd,
            max_frames=max_frames,
        )
        command_lines.append(shlex.join(command))
        report = {
            "key": key,
            "root_xy_svg": str(svg),
            "isaac_mp4": str(mp4),
            "isaac_command": shlex.join(command),
            "isaac_render_requested": bool(render_isaac),
        }
        if render_isaac:
            report.update(run_isaac_render_command(command, mp4, timeout_sec=render_timeout_sec))
        visuals.append(report)
        isaac_reports.append(report)

    commands_path = visual_dir / "isaac_mesh_commands.sh"
    commands_path.write_text(
        "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", *command_lines, ""]) + "\n",
        encoding="utf-8",
    )
    commands_path.chmod(0o755)
    rendered = [item for item in isaac_reports if item.get("isaac_status") == "ok"]
    failures = [item for item in isaac_reports if item.get("isaac_status") not in ("ok", None)]
    blocker = {
        "status": "ok" if rendered and not failures else "blocked",
        "renderer": "IsaacLab G1 USD mesh playback",
        "robot_usd": str(resolved_robot_usd),
        "robot_usd_exists": resolved_robot_usd.exists(),
        "render_script": str(render_script),
        "render_script_exists": render_script.exists(),
        "render_isaac_requested": bool(render_isaac),
        "reason": renderer_blocker_reason(
            robot_usd=resolved_robot_usd,
            render_script=render_script,
            render_isaac=render_isaac,
            rendered_count=len(rendered),
            failure_count=len(failures),
        ),
        "isaac_command_manifest": str(commands_path),
        "lightweight_visuals": visuals,
    }
    write_json(visual_dir / "isaac_mesh_renderer_blocker.json", blocker)
    payload = {
        "candidate_id": config["candidate"]["candidate_id"],
        "route": config["candidate"]["route"],
        "visuals": visuals,
        "renderer_blocker": str(visual_dir / "isaac_mesh_renderer_blocker.json"),
        "isaac_command_manifest": str(commands_path),
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


def apply_candidate(
    base: Motion,
    official: Motion,
    candidate: Mapping[str, Any],
    *,
    calibration: Mapping[str, Any] | None = None,
    row: Mapping[str, str] | None = None,
) -> tuple[Motion, dict[str, Any]]:
    out = Motion(
        fps=base.fps,
        root_pos=base.root_pos.copy(),
        root_euler=base.root_euler.copy(),
        dof=base.dof.copy(),
    )
    diagnostics: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "route": candidate["route"],
        "eval_key": row_key(row or {}) if row is not None else "",
        "eval_split": split_value(row or {}),
    }
    root_world = candidate.get("root_world", {})
    mode = root_world.get("xy_scale_mode")
    train_split_calibration_applied = False
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
        diagnostics["target_leakage_on_eval"] = True
    elif mode == "train_split_calibrated":
        train_split_calibration_applied = apply_train_split_root_front_calibration(out, calibration)
        diagnostics["train_split_calibration"] = calibration or {"status": "missing"}
        diagnostics["target_leakage_on_eval"] = False
        diagnostics["xy_scale_applied"] = float((calibration or {}).get("xy_scale", 1.0))
        diagnostics["yaw_delta_rad"] = float((calibration or {}).get("yaw_delta_rad", 0.0))

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
    elif yaw_alignment == "train_split_calibrated" and not train_split_calibration_applied:
        delta = float((calibration or {}).get("yaw_delta_rad", 0.0))
        rotate_root_xy(out, delta)
        out.root_euler[:, 2] += delta
        diagnostics["yaw_delta_rad"] = delta
        diagnostics["target_leakage_on_eval"] = False

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


def resolve_candidate_calibration(
    config: Mapping[str, Any],
    eval_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    *,
    max_frames: int,
) -> dict[str, Any] | None:
    candidate = config["candidate"]
    root_world = candidate.get("root_world", {})
    needs_calibration = (
        root_world.get("xy_scale_mode") == "train_split_calibrated"
        or root_world.get("yaw_alignment") == "train_split_calibrated"
    )
    if not needs_calibration:
        return None
    calibration = learn_train_split_root_front_calibration(config, candidate, eval_rows, max_frames=max_frames)
    path = output_dir / "calibration" / "train_split_root_front_calibration.json"
    calibration["calibration_json"] = str(path)
    write_json(path, calibration)
    return calibration


def learn_train_split_root_front_calibration(
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    eval_rows: Sequence[Mapping[str, str]],
    *,
    max_frames: int,
) -> dict[str, Any]:
    root_world = candidate.get("root_world", {})
    pairing_csv = Path(config.get("provenance", {}).get("inputs", {}).get("pairing_csv", ""))
    eval_keys = {row_key(row) for row in eval_rows}
    if not pairing_csv.exists():
        return calibration_blocker(
            "pairing_csv_missing",
            pairing_csv=str(pairing_csv),
            eval_keys=sorted(eval_keys),
        )
    try:
        all_rows = read_stage_csv(pairing_csv)
    except Exception as exc:  # noqa: BLE001
        return calibration_blocker(
            "pairing_csv_unreadable",
            pairing_csv=str(pairing_csv),
            error=repr(exc),
            eval_keys=sorted(eval_keys),
        )

    split_columns = split_columns_present(all_rows)
    train_rows = [
        row
        for row in all_rows
        if split_value(row).lower() == "train" and row_key(row) not in eval_keys
    ]
    max_rows = int(root_world.get("calibration_max_rows", 32) or 32)
    calibration_max_frames = int(root_world.get("calibration_max_frames", 240) or 240)
    if max_frames > 0:
        calibration_max_frames = min(calibration_max_frames, max_frames)

    yaw_deltas: list[float] = []
    scale_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    root_yaw_deltas: list[float] = []
    used_keys: list[str] = []
    skipped: list[dict[str, str]] = []
    for row in train_rows:
        if len(used_keys) >= max_rows:
            break
        key = row_key(row)
        try:
            official = load_official_motion(row, config)
            base = load_base_soma_motion(row, official.frame_count, official.fps)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"key": key, "reason": repr(exc)})
            continue
        n = min(official.frame_count, base.frame_count, calibration_max_frames)
        if n < 3:
            skipped.append({"key": key, "reason": "too_few_frames"})
            continue
        official = official.slice(n)
        base = base.slice(n)
        base_rel = base.root_pos[:, :2] - base.root_pos[:1, :2]
        ref_rel = official.root_pos[:, :2] - official.root_pos[:1, :2]
        heading_delta = robust_heading_delta(base_rel, ref_rel)
        if heading_delta is not None:
            yaw_deltas.append(heading_delta)
        root_yaw_delta = circular_mean(angle_diff(official.root_euler[:, 2], base.root_euler[:, 2]))
        if root_yaw_delta is not None:
            root_yaw_deltas.append(root_yaw_delta)
        scale_pairs.append((base_rel, ref_rel))
        used_keys.append(key)

    if not scale_pairs:
        return calibration_blocker(
            "no_usable_train_split_rows",
            pairing_csv=str(pairing_csv),
            split_columns=split_columns,
            train_row_count=len(train_rows),
            eval_keys=sorted(eval_keys),
            skipped_rows=skipped[:20],
        )

    yaw_delta = circular_mean(yaw_deltas)
    if yaw_delta is None:
        yaw_delta = circular_mean(root_yaw_deltas)
    if yaw_delta is None:
        yaw_delta = 0.0

    numerator = 0.0
    denominator = 0.0
    for base_rel, ref_rel in scale_pairs:
        rotated = rotate_xy_array(base_rel, yaw_delta)
        numerator += float(np.sum(rotated * ref_rel))
        denominator += float(np.sum(rotated * rotated))
    scale = numerator / denominator if denominator > 1e-12 else 1.0
    scale = float(np.clip(scale, float(root_world.get("calibration_min_scale", 0.25)), float(root_world.get("calibration_max_scale", 4.0))))
    return {
        "status": "ok",
        "calibration_type": "frozen_train_split_root_front",
        "target_leakage_on_eval": False,
        "pairing_csv": str(pairing_csv),
        "split_columns": split_columns,
        "calibration_split": "train",
        "eval_keys_excluded": sorted(eval_keys),
        "train_rows_available": len(train_rows),
        "train_rows_used": len(used_keys),
        "train_keys_used": used_keys,
        "skipped_rows": skipped[:20],
        "xy_scale": scale,
        "yaw_delta_rad": float(yaw_delta),
        "yaw_delta_deg": float(yaw_delta / ANGLE_SCALE),
        "root_yaw_delta_train_rad": float(circular_mean(root_yaw_deltas) or 0.0),
        "trajectory_yaw_samples": len(yaw_deltas),
        "max_train_rows": max_rows,
        "max_train_frames_per_row": calibration_max_frames,
    }


def calibration_blocker(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "status": "blocked",
        "calibration_type": "frozen_train_split_root_front",
        "target_leakage_on_eval": False,
        "reason": reason,
        "xy_scale": 1.0,
        "yaw_delta_rad": 0.0,
        **details,
    }


def apply_train_split_root_front_calibration(motion: Motion, calibration: Mapping[str, Any] | None) -> bool:
    if not calibration or calibration.get("status") != "ok":
        return False
    yaw_delta = float(calibration.get("yaw_delta_rad", 0.0))
    scale = float(calibration.get("xy_scale", 1.0))
    rel = motion.root_pos[:, :2] - motion.root_pos[:1, :2]
    motion.root_pos[:, :2] = motion.root_pos[:1, :2] + rotate_xy_array(rel, yaw_delta) * scale
    motion.root_euler[:, 2] += yaw_delta
    return True


def full_evaluator_metrics(pred: Motion, ref: Motion, evaluator: Mapping[str, Any]) -> dict[str, Any]:
    if evaluator.get("status") != "ok":
        return {
            "full_evaluator_status": "blocked",
            "full_evaluator_blocker": evaluator.get("reason", "unavailable"),
            "metric_threshold_pass": False,
        }
    model = evaluator["model"]
    config = evaluator["config"]
    n = min(pred.frame_count, ref.frame_count)
    pred_parsed = motion_to_parsed_frames(pred.slice(n))
    ref_parsed = motion_to_parsed_frames(ref.slice(n))
    pred_fk = [
        g1_fk_body_positions(model, joints, root, root_euler)
        for joints, root, root_euler, _ in pred_parsed
    ]
    ref_fk = [
        g1_fk_body_positions(model, joints, root, root_euler)
        for joints, root, root_euler, _ in ref_parsed
    ]
    fk_metrics = fk_mpjpe_metrics(pred_fk, ref_fk, pred.root_pos[:n], ref.root_pos[:n])
    pred_contact = g1_quality_module._contact_stats(pred_parsed, model, config, pred.fps)  # type: ignore[union-attr]
    ref_contact = g1_quality_module._contact_stats(ref_parsed, model, config, ref.fps)  # type: ignore[union-attr]
    pred_lr = foot_lr_contact_asymmetry(pred_fk, model, config)
    ref_lr = foot_lr_contact_asymmetry(ref_fk, model, config)
    contact_metrics = {
        "contact_frame_ratio_pred": float(pred_contact.get("contact_frame_ratio", 0.0)),
        "contact_frame_ratio_ref": float(ref_contact.get("contact_frame_ratio", 0.0)),
        "contact_frame_ratio_delta": abs(float(pred_contact.get("contact_frame_ratio", 0.0)) - float(ref_contact.get("contact_frame_ratio", 0.0))),
        "contact_slide_mps_pred": float(pred_contact.get("max_contact_slide_speed", 0.0)),
        "contact_slide_mps_ref": float(ref_contact.get("max_contact_slide_speed", 0.0)),
        "contact_slide_delta_mps": abs(float(pred_contact.get("max_contact_slide_speed", 0.0)) - float(ref_contact.get("max_contact_slide_speed", 0.0))),
        "foot_penetration_depth_pred_m": float(pred_contact.get("penetration_depth", 0.0)),
        "foot_penetration_depth_ref_m": float(ref_contact.get("penetration_depth", 0.0)),
        "mean_foot_clearance_pred_m": float(pred_contact.get("mean_foot_clearance", 0.0)),
        "mean_foot_clearance_ref_m": float(ref_contact.get("mean_foot_clearance", 0.0)),
        "contact_lr_asymmetry_pred": float(pred_lr.get("lr_contact_ratio_asymmetry", 0.0)),
        "contact_lr_asymmetry_ref": float(ref_lr.get("lr_contact_ratio_asymmetry", 0.0)),
        "contact_lr_asymmetry_delta": abs(float(pred_lr.get("lr_contact_ratio_asymmetry", 0.0)) - float(ref_lr.get("lr_contact_ratio_asymmetry", 0.0))),
    }
    metrics = {
        "full_evaluator_status": "ok",
        "model_xml": str(evaluator["model_xml"]),
        "foot_bodies": "|".join(model.foot_body_names),
        **fk_metrics,
        **contact_metrics,
        "contact_pred_json": json.dumps(round_float_mapping(pred_contact), sort_keys=True),
        "contact_ref_json": json.dumps(round_float_mapping(ref_contact), sort_keys=True),
        "contact_lr_json": json.dumps({"pred": pred_lr, "ref": ref_lr}, sort_keys=True),
    }
    return metrics


def load_full_evaluator(model_xml: Path | None = None) -> dict[str, Any]:
    if G1_QUALITY_IMPORT_ERROR:
        return {
            "status": "blocked",
            "reason": "g1_quality_import_failed",
            "error": G1_QUALITY_IMPORT_ERROR,
        }
    selected = model_xml if model_xml is not None else first_existing_path(DEFAULT_G1_MODEL_XMLS)
    checked = [str(path) for path in ([model_xml] if model_xml is not None else DEFAULT_G1_MODEL_XMLS) if path is not None]
    if selected is None or not selected.exists():
        return {
            "status": "blocked",
            "reason": "g1_mjcf_model_xml_missing",
            "checked_model_xml_paths": checked,
        }
    try:
        config = G1QualityConfig(model_xml=selected)  # type: ignore[operator]
        model = load_g1_kinematic_model(selected, DEFAULT_FOOT_BODIES)  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "blocked",
            "reason": "g1_mjcf_model_load_failed",
            "model_xml": str(selected),
            "error": repr(exc),
        }
    return {
        "status": "ok",
        "model_xml": selected,
        "model": model,
        "config": config,
        "body_count": len(model.bodies),
        "foot_bodies": list(model.foot_body_names),
    }


def motion_metrics(pred: Motion, ref: Motion) -> dict[str, float | int | str]:
    n = min(pred.frame_count, ref.frame_count)
    root_pred = pred.root_pos[:n]
    root_ref = ref.root_pos[:n]
    dof_delta = angle_diff(pred.dof[:n], ref.dof[:n])
    root_rot_delta = angle_diff(pred.root_euler[:n], ref.root_euler[:n])
    root_rot_geodesic = rotation_geodesic_errors(pred.root_euler[:n], ref.root_euler[:n])
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
        "root_rot_geodesic_rmse_rad": float(np.sqrt(np.mean(root_rot_geodesic**2))),
        "root_rot_max_rad": float(np.max(np.linalg.norm(root_rot_delta, axis=1))),
        "root_rot_geodesic_max_rad": float(np.max(root_rot_geodesic)),
        "dof_rmse_rad": float(np.sqrt(np.mean(dof_delta**2))),
        "dof_mae_rad": float(np.mean(np.abs(dof_delta))),
        "dof_max_abs_rad": float(np.max(np.abs(dof_delta))),
        "root_xy_span_pred_m": root_xy_span(root_pred),
        "root_xy_span_ref_m": root_xy_span(root_ref),
        "root_xy_span_ratio": root_xy_span(root_pred) / max(root_xy_span(root_ref), 1e-9),
    }


def motion_to_parsed_frames(motion: Motion) -> list[tuple[list[float], list[float], list[float], int]]:
    return [
        (
            [float(value) for value in motion.dof[idx]],
            [float(value) for value in motion.root_pos[idx]],
            [float(value) for value in motion.root_euler[idx]],
            idx,
        )
        for idx in range(motion.frame_count)
    ]


def fk_mpjpe_metrics(
    pred_fk: Sequence[Mapping[str, Sequence[tuple[float, float, float]]]],
    ref_fk: Sequence[Mapping[str, Sequence[tuple[float, float, float]]]],
    pred_root: np.ndarray,
    ref_root: np.ndarray,
) -> dict[str, Any]:
    body_world_errors: dict[str, list[float]] = {}
    body_rootrel_errors: dict[str, list[float]] = {}
    world_errors: list[float] = []
    rootrel_errors: list[float] = []
    for frame_idx, (pred_frame, ref_frame) in enumerate(zip(pred_fk, ref_fk)):
        for body in sorted(set(pred_frame) & set(ref_frame)):
            pred_point = body_representative(pred_frame.get(body, ()))
            ref_point = body_representative(ref_frame.get(body, ()))
            if pred_point is None or ref_point is None:
                continue
            world_error = float(np.linalg.norm(pred_point - ref_point))
            rootrel_error = float(
                np.linalg.norm((pred_point - pred_root[frame_idx]) - (ref_point - ref_root[frame_idx]))
            )
            body_world_errors.setdefault(body, []).append(world_error)
            body_rootrel_errors.setdefault(body, []).append(rootrel_error)
            world_errors.append(world_error)
            rootrel_errors.append(rootrel_error)

    per_body = {
        body: {
            "world_mean_m": float(np.mean(values)),
            "world_p95_m": float(np.percentile(values, 95)),
            "world_max_m": float(np.max(values)),
            "rootrel_mean_m": float(np.mean(body_rootrel_errors.get(body, [0.0]))),
            "rootrel_p95_m": float(np.percentile(body_rootrel_errors.get(body, [0.0]), 95)),
            "rootrel_max_m": float(np.max(body_rootrel_errors.get(body, [0.0]))),
        }
        for body, values in body_world_errors.items()
    }
    worst_body = ""
    if per_body:
        worst_body = max(per_body.items(), key=lambda item: item[1]["rootrel_mean_m"])[0]
    return {
        "fk_body_count": len(per_body),
        "fk_world_mpjpe_m": finite_stat_mean(world_errors),
        "fk_world_p50_m": finite_stat_percentile(world_errors, 50),
        "fk_world_p95_m": finite_stat_percentile(world_errors, 95),
        "fk_world_max_m": finite_stat_max(world_errors),
        "fk_rootrel_mpjpe_m": finite_stat_mean(rootrel_errors),
        "fk_rootrel_p50_m": finite_stat_percentile(rootrel_errors, 50),
        "fk_rootrel_p95_m": finite_stat_percentile(rootrel_errors, 95),
        "fk_rootrel_max_m": finite_stat_max(rootrel_errors),
        "fk_worst_body": worst_body,
        "fk_per_body_json": json.dumps(round_nested(per_body), sort_keys=True),
    }


def foot_lr_contact_asymmetry(
    fk_frames: Sequence[Mapping[str, Sequence[tuple[float, float, float]]]],
    model: Any,
    config: Any,
) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for side in ("left", "right"):
        bodies = [body for body in model.foot_body_names if body.startswith(side)]
        if not bodies:
            ratios[f"{side}_contact_ratio"] = 0.0
            continue
        contact_count = 0
        frame_count = 0
        for frame in fk_frames:
            points = [point for body in bodies for point in frame.get(body, ())]
            if not points:
                continue
            frame_count += 1
            low = min(points, key=lambda point: point[2])
            if low[2] - config.ground_height <= config.contact_height_threshold:
                contact_count += 1
        ratios[f"{side}_contact_ratio"] = contact_count / frame_count if frame_count else 0.0
    ratios["lr_contact_ratio_asymmetry"] = abs(ratios["left_contact_ratio"] - ratios["right_contact_ratio"])
    return ratios


def body_representative(points: Sequence[tuple[float, float, float]]) -> np.ndarray | None:
    if not points:
        return None
    return np.asarray(points, dtype=np.float64).mean(axis=0)


def metric_threshold_pass(metrics: Mapping[str, Any]) -> bool:
    checks = []
    for key, threshold in ACCEPTANCE_THRESHOLDS.items():
        value = metrics.get(key)
        if value in (None, ""):
            return False
        try:
            checks.append(float(value) <= float(threshold))
        except (TypeError, ValueError):
            return False
    return all(checks)


def public_evaluator_report(evaluator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in evaluator.items()
        if key not in {"model", "config"}
    }


def finite_stat_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def finite_stat_percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def finite_stat_max(values: Sequence[float]) -> float:
    return float(np.max(values)) if values else 0.0


def round_float_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (round(float(value), 6) if isinstance(value, (int, float)) else value)
        for key, value in values.items()
    }


def round_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    if isinstance(value, tuple):
        return [round_nested(item) for item in value]
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


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
            writer.writerow({key: csv_cell(value) for key, value in record.items()})


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [
        "root_rel_rmse_m",
        "root_rot_geodesic_rmse_rad",
        "root_rot_rmse_rad",
        "dof_rmse_rad",
        "dof_max_abs_rad",
        "root_xy_span_ratio",
        "fk_world_mpjpe_m",
        "fk_rootrel_mpjpe_m",
        "contact_frame_ratio_delta",
        "contact_slide_delta_mps",
        "foot_penetration_depth_pred_m",
        "mean_foot_clearance_pred_m",
    ]
    out: dict[str, Any] = {}
    for metric in metrics:
        vals = [float(row[metric]) for row in records if row.get(metric) not in (None, "")]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            keyed = [
                (float(row[metric]), str(row.get("key", "")))
                for row in records
                if row.get(metric) not in (None, "")
            ]
            worst_val, worst_key = max(keyed, key=lambda item: item[0])
            out[metric] = {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "max": float(np.max(arr)),
                "worst_key": worst_key,
                "worst_value": worst_val,
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


def build_isaac_render_command(
    row: Mapping[str, str],
    *,
    candidate_csv: Path,
    output_mp4: Path,
    robot_usd: Path,
    max_frames: int,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_g1_isaac_pair.py"),
        "--g1-motion",
        str(candidate_csv),
        "--format",
        "csv",
        "--output",
        str(output_mp4),
        "--robot-usd",
        str(robot_usd),
        "--target-fps",
        str(float(row.get("source_bvh_fps") or 120.0048)),
        "--preserve-world-root",
        "--draw-orientation-labels",
        "--fast-exit-after-report",
    ]
    if max_frames > 0:
        command.extend(["--max-frames", str(max_frames)])
    source_bvh = source_bvh_path(row)
    if source_bvh is not None and source_bvh.exists():
        command.extend(["--bvh", str(source_bvh), "--source-renderer", "somamesh"])
    return command


def run_isaac_render_command(command: Sequence[str], output_mp4: Path, *, timeout_sec: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "isaac_status": "blocked",
            "isaac_blocker": "render_timeout",
            "timeout_sec": timeout_sec,
            "stdout_tail": tail_text(exc.stdout or ""),
            "stderr_tail": tail_text(exc.stderr or ""),
            "output_exists": output_mp4.exists(),
            "output_bytes": output_mp4.stat().st_size if output_mp4.exists() else 0,
        }
    output_bytes = output_mp4.stat().st_size if output_mp4.exists() else 0
    status = "ok" if result.returncode == 0 and output_bytes > 0 else "blocked"
    report = {
        "isaac_status": status,
        "isaac_returncode": result.returncode,
        "output_exists": output_mp4.exists(),
        "output_bytes": output_bytes,
        "stdout_tail": tail_text(result.stdout),
        "stderr_tail": tail_text(result.stderr),
    }
    sidecar = output_mp4.with_suffix(".json")
    if sidecar.exists():
        try:
            report["isaac_sidecar_json"] = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            report["isaac_sidecar_parse_error"] = repr(exc)
    if status != "ok":
        report["isaac_blocker"] = "render_command_failed_or_missing_output"
    return report


def renderer_blocker_reason(
    *,
    robot_usd: Path,
    render_script: Path,
    render_isaac: bool,
    rendered_count: int,
    failure_count: int,
) -> str:
    if not render_script.exists():
        return "render_script_missing"
    if not robot_usd.exists():
        return "robot_usd_missing"
    if rendered_count and not failure_count:
        return ""
    if render_isaac and failure_count:
        return "isaac_render_failed; see per-row stdout/stderr and sidecar JSON"
    if not render_isaac:
        return "isaac_render_not_executed; commands are materialized in isaac_mesh_commands.sh"
    return "isaac_render_missing_output"


def source_bvh_path(row: Mapping[str, str]) -> Path | None:
    for column in ("source_bvh", "soma_bvh", "move_soma_proportional_path"):
        value = str(row.get(column, "")).strip()
        if value and not value.endswith(".tar"):
            path = Path(value)
            if path.suffix.lower() == ".bvh":
                return path
    return None


def tail_text(value: str, *, limit: int = 4000) -> str:
    return value[-limit:] if len(value) > limit else value


def candidate_csv_path(output_dir: Path, key: str) -> Path:
    return output_dir / "retarget_csv" / f"{safe_name(key)}.csv"


def row_key(row: Mapping[str, str]) -> str:
    for column in ("lr271_key", "motion_key", "key", "clip_key", "bones_rel_key"):
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return Path(str(row.get("source_bvh", "clip"))).stem


def split_value(row: Mapping[str, str]) -> str:
    for column in ("split", "dataset_split", "source_split", "bones_split"):
        value = str(row.get(column, "")).strip()
        if value:
            return value
    return ""


def split_columns_present(rows: Sequence[Mapping[str, str]]) -> list[str]:
    columns = []
    for column in ("split", "dataset_split", "source_split", "bones_split"):
        if any(str(row.get(column, "")).strip() for row in rows):
            columns.append(column)
    return columns


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


def circular_mean(values: Sequence[float] | np.ndarray) -> float | None:
    if len(values) == 0:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(math.atan2(float(np.mean(np.sin(arr))), float(np.mean(np.cos(arr)))))


def robust_heading_delta(base_rel: np.ndarray, ref_rel: np.ndarray) -> float | None:
    deltas: list[float] = []
    for base_vec, ref_vec in zip(base_rel[1:], ref_rel[1:]):
        base_norm = float(np.linalg.norm(base_vec))
        ref_norm = float(np.linalg.norm(ref_vec))
        if base_norm < 0.03 or ref_norm < 0.03:
            continue
        base_yaw = math.atan2(float(base_vec[1]), float(base_vec[0]))
        ref_yaw = math.atan2(float(ref_vec[1]), float(ref_vec[0]))
        deltas.append(angle_diff_scalar(ref_yaw, base_yaw))
    return circular_mean(deltas)


def rotate_root_xy(motion: Motion, yaw_delta: float) -> None:
    rel = motion.root_pos[:, :2] - motion.root_pos[:1, :2]
    motion.root_pos[:, :2] = motion.root_pos[:1, :2] + rotate_xy_array(rel, yaw_delta)


def rotate_xy_array(values: np.ndarray, yaw_delta: float) -> np.ndarray:
    c, s = math.cos(yaw_delta), math.sin(yaw_delta)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(values, dtype=np.float64) @ rot.T


def rotation_geodesic_errors(pred_euler: np.ndarray, ref_euler: np.ndarray) -> np.ndarray:
    errors = []
    for pred, ref in zip(pred_euler, ref_euler):
        pred_matrix = euler_xyz_to_matrix(pred)
        ref_matrix = euler_xyz_to_matrix(ref)
        delta = pred_matrix.T @ ref_matrix
        trace = float(np.trace(delta))
        errors.append(math.acos(float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))))
    return np.asarray(errors, dtype=np.float64)


def euler_xyz_to_matrix(euler_xyz: Sequence[float]) -> np.ndarray:
    rx, ry, rz = [float(value) for value in euler_xyz[:3]]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mat_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    mat_y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    mat_z = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mat_x @ mat_y @ mat_z


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


def first_existing_path(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, Path):
        return str(value)
    return value


def escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
