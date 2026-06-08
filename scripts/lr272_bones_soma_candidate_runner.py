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
import itertools
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

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
DEFAULT_CONTACT_METRIC_BODY_FLOOR_AUDIT_REPORT = (
    Path("/home/user/project/OnlineRetarget")
    / "outputs"
    / "lr272_contact_metric_body_floor_audit_20260608T193304Z"
    / "contact_metric_body_floor_audit_report.json"
)
CORRECTED_CONTACT_EVALUATOR_ID = "provisional_collision_sphere_p05_v1"
CORRECTED_CONTACT_HEIGHT_DEFINITION = "foot_collision_sphere_bottom_min_z"
CORRECTED_CONTACT_FLOOR_RULE = "train_official_p05_height"
CORRECTED_CONTACT_GROUND_HEIGHT_M = -0.03624434809693291
CORRECTED_CONTACT_THRESHOLD_M = 0.04
CORRECTED_CONTACT_EXPECTED_SPHERE_COUNT = 8
ACCEPTANCE_THRESHOLDS = {
    "root_rel_rmse_m": 0.05,
    "root_rot_geodesic_rmse_rad": 0.15,
    "fk_rootrel_mpjpe_m": 0.05,
    "dof_rmse_rad": 0.15,
    "contact_frame_ratio_delta": 0.15,
    "contact_slide_delta_mps": 0.50,
}

LOWER_BODY_FK_SIGNATURE_GROUPS = {
    "left_hip": {
        "joints": ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"),
        "bodies": (
            "left_hip_pitch_link",
            "left_hip_roll_link",
            "left_hip_yaw_link",
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "left_toe_link",
        ),
    },
    "right_hip": {
        "joints": ("right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint"),
        "bodies": (
            "right_hip_pitch_link",
            "right_hip_roll_link",
            "right_hip_yaw_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
            "right_toe_link",
        ),
    },
    "left_knee": {
        "joints": ("left_knee_joint",),
        "bodies": ("left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link", "left_toe_link"),
    },
    "right_knee": {
        "joints": ("right_knee_joint",),
        "bodies": ("right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link", "right_toe_link"),
    },
    "left_ankle": {
        "joints": ("left_ankle_pitch_joint", "left_ankle_roll_joint"),
        "bodies": ("left_ankle_pitch_link", "left_ankle_roll_link", "left_toe_link"),
    },
    "right_ankle": {
        "joints": ("right_ankle_pitch_joint", "right_ankle_roll_joint"),
        "bodies": ("right_ankle_pitch_link", "right_ankle_roll_link", "right_toe_link"),
    },
    "waist": {
        "joints": ("waist_yaw_joint", "waist_roll_joint"),
        "bodies": ("waist_yaw_link", "waist_roll_link", "torso_link"),
    },
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


@dataclass(frozen=True)
class FootCollisionSphere:
    body_name: str
    side: str
    local_pos: np.ndarray
    radius: float
    attrs: Mapping[str, str]


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
    run_start_gates = verify_run_start_gates(config, args.output_dir, model_xml=args.model_xml)
    manifest: dict[str, Any] = {
        "config": str(args.config),
        "stage_csv": str(args.stage_csv),
        "output_dir": str(args.output_dir),
        "mode": args.mode,
        "candidate_id": config["candidate"]["candidate_id"],
        "route": config["candidate"]["route"],
        "row_count": len(rows),
        "run_start_gates": run_start_gates,
    }
    if args.mode in ("retarget", "all"):
        manifest["retarget"] = run_retarget(config, rows, args.output_dir, max_frames=args.max_frames, model_xml=args.model_xml)
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
            model_xml=args.model_xml,
        )
    write_json(args.output_dir / "runner_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def verify_run_start_gates(config: Mapping[str, Any], output_dir: Path, *, model_xml: Path | None = None) -> dict[str, Any]:
    candidate = config.get("candidate", {})
    validation = candidate.get("validation", {})
    gates = validation.get("run_start_gates", {})
    frame_gate = gates.get("frame_consistency_report", {})
    if not frame_gate:
        gate = (
            candidate.get("dof_convention", {})
            .get("train_split_fk_signature_map", {})
            .get("frame_consistency_report", {})
        )
        frame_gate = gate
    contact_gate = gates.get("contact_metric_body_floor_audit", {})
    if not frame_gate and not contact_gate:
        return {"status": "not_required"}

    results: dict[str, Any] = {"status": "passed", "checks": {}}
    if frame_gate and bool(frame_gate.get("required", False)):
        frame_result = verify_frame_consistency_gate(config, frame_gate)
        results["frame_consistency_report"] = frame_result.get("frame_consistency_report", "")
        results["expected_status"] = frame_result.get("expected_status", "")
        results["observed_status"] = frame_result.get("observed_status", "")
        results["pass_checks"] = frame_result.get("pass_checks", {})
        results["proof_output_dir"] = frame_result.get("proof_output_dir", "")
        results["frame_consistency_gate"] = frame_result
        results["checks"]["frame_consistency_report"] = frame_result.get("status") == "passed"
        if frame_result.get("status") != "passed":
            results["status"] = "blocked"
            results["reason"] = frame_result.get("reason", "frame_consistency_report_failed")
    if contact_gate and bool(contact_gate.get("required", False)):
        contact_result = verify_contact_metric_body_floor_gate(config, contact_gate, model_xml=model_xml)
        results["contact_metric_body_floor_audit_gate"] = contact_result
        results["checks"]["contact_metric_body_floor_audit"] = contact_result.get("status") == "passed"
        if contact_result.get("status") != "passed":
            results["status"] = "blocked"
            results["reason"] = contact_result.get("reason", "contact_metric_body_floor_audit_failed")
    write_json(output_dir / "run_start_gates.json", results)
    if results.get("status") != "passed":
        raise SystemExit(f"run_start_gate_failed:{results.get('reason', 'unknown')}")
    return results


def verify_frame_consistency_gate(config: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    proof_path = str(gate.get("path") or config.get("provenance", {}).get("inputs", {}).get("frame_consistency_report_json", ""))
    report_path = Path(proof_path) if proof_path else Path()
    if proof_path and not report_path.is_absolute():
        report_path = REPO_ROOT / report_path
    result: dict[str, Any] = {
        "status": "blocked",
        "frame_consistency_report": str(report_path) if proof_path else "",
        "required": True,
        "expected_status": str(gate.get("expected_status", "passed")),
    }
    if not proof_path:
        result["reason"] = "frame_consistency_report_path_missing"
    elif not report_path.exists():
        result["reason"] = "frame_consistency_report_missing"
    else:
        try:
            proof = read_json(report_path)
        except Exception as exc:  # noqa: BLE001
            result["reason"] = "frame_consistency_report_unreadable"
            result["error"] = repr(exc)
        else:
            pass_checks = proof.get("pass_checks", {})
            result.update(
                {
                    "observed_status": proof.get("status", ""),
                    "pass_checks": pass_checks,
                    "proof_output_dir": proof.get("output_dir", ""),
                }
            )
            expected_status = result["expected_status"]
            if proof.get("status") == expected_status and all(bool(value) for value in pass_checks.values()):
                result["status"] = "passed"
                result["reason"] = ""
    return result


def verify_contact_metric_body_floor_gate(
    config: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    model_xml: Path | None,
) -> dict[str, Any]:
    expected_definition = str(gate.get("expected_definition", CORRECTED_CONTACT_HEIGHT_DEFINITION))
    expected_floor_rule = str(gate.get("expected_floor_rule", CORRECTED_CONTACT_FLOOR_RULE))
    expected_ground = float(gate.get("expected_ground_height_m", CORRECTED_CONTACT_GROUND_HEIGHT_M))
    expected_threshold = float(gate.get("expected_contact_threshold_m", CORRECTED_CONTACT_THRESHOLD_M))
    expected_spheres = int(gate.get("expected_collision_sphere_count", CORRECTED_CONTACT_EXPECTED_SPHERE_COUNT))
    expected_mesh_blocker = bool(gate.get("expected_mesh_extent_blocker", True))
    audit_path = contact_audit_report_path(config, gate)
    result: dict[str, Any] = {
        "status": "blocked",
        "required": True,
        "audit_report": str(audit_path),
        "expected_definition": expected_definition,
        "expected_floor_rule": expected_floor_rule,
        "expected_ground_height_m": expected_ground,
        "expected_contact_threshold_m": expected_threshold,
        "expected_collision_sphere_count": expected_spheres,
        "expected_mesh_extent_blocker": expected_mesh_blocker,
        "checks": {},
    }
    if not audit_path.exists():
        result["reason"] = "contact_metric_body_floor_audit_report_missing"
        return result
    try:
        audit = read_json(audit_path)
    except Exception as exc:  # noqa: BLE001
        result["reason"] = "contact_metric_body_floor_audit_report_unreadable"
        result["error"] = repr(exc)
        return result
    selected = audit.get("selected_floor_convention", {})
    inventory = audit.get("mjcf_feature_inventory", {})
    observed_definition = selected.get("definition", "")
    observed_floor_rule = selected.get("floor_rule", "")
    observed_ground = float(selected.get("ground_height_m", float("nan")))
    observed_threshold = float(selected.get("contact_threshold_m", expected_threshold))
    observed_mesh_blocker = bool(selected.get("mesh_extent_blocker", False))
    observed_spheres = int(inventory.get("foot_collision_sphere_count", 0))
    selected_model = model_xml
    if selected_model is None:
        model_value = str(gate.get("model_xml", "")).strip()
        selected_model = Path(model_value) if model_value else None
    if selected_model is None:
        selected_model = first_existing_path(DEFAULT_G1_MODEL_XMLS)
    model_spheres = load_foot_collision_spheres(selected_model, DEFAULT_FOOT_BODIES) if selected_model and selected_model.exists() else []
    result.update(
        {
            "observed_definition": observed_definition,
            "observed_floor_rule": observed_floor_rule,
            "observed_ground_height_m": observed_ground,
            "observed_contact_threshold_m": observed_threshold,
            "observed_collision_sphere_count": observed_spheres,
            "observed_model_collision_sphere_count": len(model_spheres),
            "observed_mesh_extent_blocker": observed_mesh_blocker,
            "mesh_extent_blocker_status": "git_lfs_pointer_no_mesh_vertices" if observed_mesh_blocker else "not_blocked",
            "audit_status": audit.get("status", ""),
        }
    )
    checks = {
        "audit_status_complete": audit.get("status") == "complete",
        "definition_match": observed_definition == expected_definition,
        "floor_rule_match": observed_floor_rule == expected_floor_rule,
        "ground_height_match": abs(observed_ground - expected_ground) <= 1e-12,
        "contact_threshold_match": abs(observed_threshold - expected_threshold) <= 1e-12,
        "audit_collision_sphere_count_match": observed_spheres == expected_spheres,
        "model_collision_sphere_count_match": len(model_spheres) == expected_spheres,
        "mesh_extent_blocker_match": observed_mesh_blocker is expected_mesh_blocker,
    }
    result["checks"] = checks
    if all(checks.values()):
        result["status"] = "passed"
        result["reason"] = ""
    else:
        failed = [name for name, ok in checks.items() if not ok]
        result["reason"] = "contact_metric_body_floor_audit_gate_failed:" + ",".join(failed)
    return result


def run_retarget(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    *,
    max_frames: int,
    model_xml: Path | None = None,
) -> list[dict[str, Any]]:
    candidate = config["candidate"]
    out_dir = output_dir / "retarget_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration = resolve_candidate_calibration(config, rows, output_dir, max_frames=max_frames, model_xml=model_xml)
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
            run_retarget(config, [row], output_dir, max_frames=max_frames, model_xml=model_xml)
        official = load_official_motion(row, config)
        candidate_motion = load_candidate_csv(csv_path, float(row.get("source_bvh_fps") or official.fps))
        n = min(official.frame_count, candidate_motion.frame_count)
        if max_frames > 0:
            n = min(n, max_frames)
        official = official.slice(n)
        candidate_motion = candidate_motion.slice(n)
        full_metrics = full_evaluator_metrics(candidate_motion, official, evaluator, config=config)
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
        "contact_evaluator": public_corrected_contact_context(corrected_contact_context(config, evaluator)) if evaluator.get("status") == "ok" else {},
        "metric_notes": [
            "Eval rows are tagged with their pairing split; train-split calibration candidates must not fit eval rows.",
            "Angles are compared in radians after CSV degrees-to-radians conversion.",
            "FK fields are computed from the configured G1 MJCF when available; otherwise full_evaluator_blocker.json explains the missing dependency.",
            "Primary contact fields use the provisional collision-sphere/p05 floor evaluator; legacy_* contact fields preserve the old ankle-point/ground-zero evaluator for before/after comparison.",
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
    model_xml: Path | None = None,
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
            run_retarget(config, [row], output_dir, max_frames=max_frames, model_xml=model_xml)
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
    if dof_cfg.get("train_split_fk_signature_map", {}).get("enabled", False):
        applied = apply_source_to_target_dof_map(out, (calibration or {}).get("source_to_target_dof_map", {}))
        diagnostics["lower_body_fk_signature_dof_map"] = calibration or {"status": "missing"}
        diagnostics["lower_body_fk_signature_dof_map_applied"] = applied
        diagnostics["target_leakage_on_eval"] = False

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
    model_xml: Path | None = None,
) -> dict[str, Any] | None:
    candidate = config["candidate"]
    root_world = candidate.get("root_world", {})
    dof_cfg = candidate.get("dof_convention", {})
    needs_calibration = (
        root_world.get("xy_scale_mode") == "train_split_calibrated"
        or root_world.get("yaw_alignment") == "train_split_calibrated"
    )
    needs_dof_calibration = bool(dof_cfg.get("train_split_fk_signature_map", {}).get("enabled", False))
    if needs_dof_calibration:
        calibration = learn_train_split_lower_body_dof_map(
            config,
            candidate,
            eval_rows,
            max_frames=max_frames,
            model_xml=model_xml,
        )
        path = output_dir / "calibration" / "lower_body_fk_signature_dof_map_train_split_v1.json"
        calibration["calibration_json"] = str(path)
        write_json(path, calibration)
        if calibration.get("status") != "ok" and bool(
            dof_cfg.get("train_split_fk_signature_map", {}).get("required", True)
        ):
            raise SystemExit(f"lower_body_dof_calibration_failed:{calibration.get('reason', 'unknown')}")
        return calibration
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


def learn_train_split_lower_body_dof_map(
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    eval_rows: Sequence[Mapping[str, str]],
    *,
    max_frames: int,
    model_xml: Path | None,
) -> dict[str, Any]:
    dof_cfg = candidate.get("dof_convention", {}).get("train_split_fk_signature_map", {})
    evaluator = load_full_evaluator(model_xml)
    if evaluator.get("status") != "ok":
        return lower_body_dof_map_blocker(
            "full_evaluator_unavailable",
            evaluator=public_evaluator_report(evaluator),
        )
    pairing_csv = Path(config.get("provenance", {}).get("inputs", {}).get("pairing_csv", ""))
    eval_keys = {row_key(row) for row in eval_rows}
    if not pairing_csv.exists():
        return lower_body_dof_map_blocker(
            "pairing_csv_missing",
            pairing_csv=str(pairing_csv),
            eval_keys=sorted(eval_keys),
        )
    try:
        all_rows = read_stage_csv(pairing_csv)
    except Exception as exc:  # noqa: BLE001
        return lower_body_dof_map_blocker(
            "pairing_csv_unreadable",
            pairing_csv=str(pairing_csv),
            error=repr(exc),
            eval_keys=sorted(eval_keys),
        )

    split_columns = split_columns_present(all_rows)
    train_rows = [
        row
        for row in all_rows
        if split_value(row).lower() == str(dof_cfg.get("calibration_split", "train")).lower()
        and row_key(row) not in eval_keys
    ]
    max_rows = int(dof_cfg.get("calibration_max_rows", 16) or 16)
    calibration_max_frames = int(dof_cfg.get("calibration_max_frames", 160) or 160)
    if max_frames > 0:
        calibration_max_frames = min(calibration_max_frames, max_frames)
    examples: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for row in train_rows:
        if len(examples) >= max_rows:
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
        examples.append(
            {
                "key": key,
                "base": base.slice(n),
                "official": official,
                "official_fk": fk_frames_for_motion(official, evaluator["model"]),
            }
        )

    if not examples:
        return lower_body_dof_map_blocker(
            "no_usable_train_split_rows",
            pairing_csv=str(pairing_csv),
            split_columns=split_columns,
            train_row_count=len(train_rows),
            eval_keys=sorted(eval_keys),
            skipped_rows=skipped[:20],
        )

    groups = selected_fk_signature_groups(dof_cfg)
    min_relative_improvement = float(dof_cfg.get("min_relative_improvement", 0.0) or 0.0)
    selected_map: dict[str, dict[str, Any]] = {}
    group_reports: list[dict[str, Any]] = []
    for group_name in groups:
        group = LOWER_BODY_FK_SIGNATURE_GROUPS[group_name]
        target_joints = tuple(group["joints"])
        bodies = tuple(group["bodies"])
        identity_map = {
            joint: {"source_joint": joint, "sign": 1.0, "group": group_name}
            for joint in target_joints
        }
        baseline_map = {**selected_map, **identity_map}
        baseline_score = score_dof_mapping_on_train_examples(examples, evaluator["model"], baseline_map, bodies)
        best_score = baseline_score
        best_map = identity_map
        best_reason = "identity_best_or_no_improvement"
        for trial_map in candidate_group_dof_maps(group_name, target_joints):
            score = score_dof_mapping_on_train_examples(examples, evaluator["model"], {**selected_map, **trial_map}, bodies)
            if score < best_score:
                best_score = score
                best_map = trial_map
                best_reason = "train_root_aligned_fk_score"
        required_score = baseline_score * (1.0 - min_relative_improvement)
        if best_score > required_score:
            best_score = baseline_score
            best_map = identity_map
            best_reason = "identity_retained_below_min_relative_improvement"
        selected_map.update(best_map)
        group_reports.append(
            {
                "group": group_name,
                "target_joints": list(target_joints),
                "bodies": list(bodies),
                "candidate_count": len(candidate_group_dof_maps(group_name, target_joints)),
                "baseline_rootrel_fk_mean_m": baseline_score,
                "selected_rootrel_fk_mean_m": best_score,
                "selected_reason": best_reason,
                "selected_map": best_map,
            }
        )

    signatures = single_axis_signature_report(
        evaluator["model"],
        groups,
        float(dof_cfg.get("single_axis_delta_rad", 0.174533) or 0.174533),
    )
    return {
        "status": "ok",
        "calibration_type": "lower_body_fk_signature_dof_map_train_split_v1",
        "target_leakage_on_eval": False,
        "pairing_csv": str(pairing_csv),
        "split_columns": split_columns,
        "calibration_split": str(dof_cfg.get("calibration_split", "train")),
        "eval_keys_excluded": sorted(eval_keys),
        "train_rows_available": len(train_rows),
        "train_rows_used": len(examples),
        "train_keys_used": [str(example["key"]) for example in examples],
        "skipped_rows": skipped[:20],
        "max_train_rows": max_rows,
        "max_train_frames_per_row": calibration_max_frames,
        "model_xml": str(evaluator["model_xml"]),
        "groups": group_reports,
        "single_axis_signatures": signatures,
        "source_to_target_dof_map": selected_map,
        "root_world_policy": candidate.get("root_world", {}),
        "summarizer_policy": candidate.get("summarizer", {}),
        "scope": {
            "lower_body_only": True,
            "allow_waist": bool(dof_cfg.get("allow_waist", False)),
            "allow_shoulder_elbow": False,
            "no_per_clip_eval_target_fitting": True,
        },
    }


def lower_body_dof_map_blocker(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "status": "blocked",
        "calibration_type": "lower_body_fk_signature_dof_map_train_split_v1",
        "target_leakage_on_eval": False,
        "reason": reason,
        "source_to_target_dof_map": {},
        **details,
    }


def selected_fk_signature_groups(dof_cfg: Mapping[str, Any]) -> list[str]:
    requested = [str(item) for item in dof_cfg.get("lower_body_groups", ()) if str(item)]
    groups = [name for name in requested if name in LOWER_BODY_FK_SIGNATURE_GROUPS]
    if bool(dof_cfg.get("allow_waist", False)) and "waist" not in groups:
        groups.append("waist")
    return groups or [
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]


def candidate_group_dof_maps(group_name: str, target_joints: Sequence[str]) -> list[dict[str, dict[str, Any]]]:
    maps: list[dict[str, dict[str, Any]]] = []
    for source_order in itertools.permutations(target_joints, len(target_joints)):
        for signs in itertools.product((-1.0, 1.0), repeat=len(target_joints)):
            maps.append(
                {
                    target_joint: {
                        "source_joint": source_joint,
                        "sign": float(sign),
                        "group": group_name,
                    }
                    for target_joint, source_joint, sign in zip(target_joints, source_order, signs)
                }
            )
    return maps


def score_dof_mapping_on_train_examples(
    examples: Sequence[Mapping[str, Any]],
    model: Any,
    source_to_target_map: Mapping[str, Mapping[str, Any]],
    body_names: Sequence[str],
) -> float:
    errors: list[float] = []
    for example in examples:
        base: Motion = example["base"]
        official: Motion = example["official"]
        pred = Motion(
            fps=base.fps,
            root_pos=base.root_pos.copy(),
            root_euler=base.root_euler.copy(),
            dof=base.dof.copy(),
        )
        apply_source_to_target_dof_map(pred, source_to_target_map)
        pred_fk = fk_frames_for_motion(pred, model)
        ref_fk = example["official_fk"]
        errors.extend(rootrel_fk_body_errors(pred, official, pred_fk, ref_fk, body_names))
    return finite_stat_mean(errors)


def apply_source_to_target_dof_map(
    motion: Motion,
    source_to_target_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    original = motion.dof.copy()
    applied: list[dict[str, Any]] = []
    for target_joint, spec in sorted(source_to_target_map.items()):
        target_idx = joint_index(target_joint)
        source_joint = str(spec.get("source_joint", target_joint))
        source_idx = joint_index(source_joint)
        if target_idx is None or source_idx is None:
            continue
        sign = float(spec.get("sign", 1.0))
        motion.dof[:, target_idx] = original[:, source_idx] * sign
        applied.append(
            {
                "target_joint": target_joint,
                "source_joint": source_joint,
                "sign": sign,
                "group": spec.get("group", ""),
            }
        )
    return applied


def fk_frames_for_motion(motion: Motion, model: Any) -> list[Mapping[str, Sequence[tuple[float, float, float]]]]:
    return [
        g1_fk_body_positions(model, joints, root, root_euler)
        for joints, root, root_euler, _ in motion_to_parsed_frames(motion)
    ]


def rootrel_fk_body_errors(
    pred: Motion,
    ref: Motion,
    pred_fk: Sequence[Mapping[str, Sequence[tuple[float, float, float]]]],
    ref_fk: Sequence[Mapping[str, Sequence[tuple[float, float, float]]]],
    body_names: Sequence[str],
) -> list[float]:
    errors: list[float] = []
    wanted = tuple(body_names)
    for frame_idx, (pred_frame, ref_frame) in enumerate(zip(pred_fk, ref_fk)):
        for body in wanted:
            pred_point = body_representative(pred_frame.get(body, ()))
            ref_point = body_representative(ref_frame.get(body, ()))
            if pred_point is None or ref_point is None:
                continue
            pred_root_point = root_aligned_point(pred_point, pred.root_pos[frame_idx], pred.root_euler[frame_idx])
            ref_root_point = root_aligned_point(ref_point, ref.root_pos[frame_idx], ref.root_euler[frame_idx])
            errors.append(float(np.linalg.norm(pred_root_point - ref_root_point)))
    return errors


def single_axis_signature_report(model: Any, groups: Sequence[str], delta_rad: float) -> dict[str, Any]:
    report: dict[str, Any] = {}
    neutral = Motion(
        fps=120.0,
        root_pos=np.zeros((1, 3), dtype=np.float64),
        root_euler=np.zeros((1, 3), dtype=np.float64),
        dof=np.zeros((1, len(G1_JOINT_COLUMNS)), dtype=np.float64),
    )
    base_fk = fk_frames_for_motion(neutral, model)
    for group_name in groups:
        group = LOWER_BODY_FK_SIGNATURE_GROUPS[group_name]
        entries: dict[str, Any] = {}
        for joint in group["joints"]:
            idx = joint_index(joint)
            if idx is None:
                continue
            perturbed = neutral.slice(1)
            perturbed.dof[:, idx] = delta_rad
            plus_fk = fk_frames_for_motion(perturbed, model)
            displacements: list[float] = []
            for body in group["bodies"]:
                base_point = body_representative(base_fk[0].get(body, ()))
                plus_point = body_representative(plus_fk[0].get(body, ()))
                if base_point is None or plus_point is None:
                    continue
                base_root = root_aligned_point(base_point, neutral.root_pos[0], neutral.root_euler[0])
                plus_root = root_aligned_point(plus_point, perturbed.root_pos[0], perturbed.root_euler[0])
                displacements.append(float(np.linalg.norm(plus_root - base_root)))
            entries[joint] = {
                "delta_rad": delta_rad,
                "body_count": len(displacements),
                "mean_rootrel_displacement_m": finite_stat_mean(displacements),
                "max_rootrel_displacement_m": finite_stat_max(displacements),
            }
        report[group_name] = entries
    return report


def full_evaluator_metrics(
    pred: Motion,
    ref: Motion,
    evaluator: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if evaluator.get("status") != "ok":
        return {
            "full_evaluator_status": "blocked",
            "full_evaluator_blocker": evaluator.get("reason", "unavailable"),
            "metric_threshold_pass": False,
        }
    model = evaluator["model"]
    quality_config = evaluator["config"]
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
    fk_metrics = fk_mpjpe_metrics(
        pred_fk,
        ref_fk,
        pred.root_pos[:n],
        ref.root_pos[:n],
        pred.root_euler[:n],
        ref.root_euler[:n],
    )
    legacy_pred_contact = g1_quality_module._contact_stats(pred_parsed, model, quality_config, pred.fps)  # type: ignore[union-attr]
    legacy_ref_contact = g1_quality_module._contact_stats(ref_parsed, model, quality_config, ref.fps)  # type: ignore[union-attr]
    legacy_pred_lr = foot_lr_contact_asymmetry(pred_fk, model, quality_config)
    legacy_ref_lr = foot_lr_contact_asymmetry(ref_fk, model, quality_config)
    corrected_context = corrected_contact_context(config or {}, evaluator)
    if corrected_context.get("status") == "ok":
        pred_contact = corrected_contact_stats(pred.slice(n), model, corrected_context, pred.fps)
        ref_contact = corrected_contact_stats(ref.slice(n), model, corrected_context, ref.fps)
        pred_lr = corrected_contact_lr_asymmetry(pred_contact)
        ref_lr = corrected_contact_lr_asymmetry(ref_contact)
    else:
        pred_contact = legacy_pred_contact
        ref_contact = legacy_ref_contact
        pred_lr = legacy_pred_lr
        ref_lr = legacy_ref_lr
    contact_metrics = {
        "contact_evaluator_id": str(corrected_context.get("evaluator_id", "legacy_ankle_body_points_ground0")),
        "contact_evaluator_status": str(corrected_context.get("status", "")),
        "contact_height_definition": str(corrected_context.get("height_definition", "legacy_ankle_body_points_min_z")),
        "contact_ground_height_m": float(corrected_context.get("ground_height_m", quality_config.ground_height)),
        "contact_threshold_m": float(corrected_context.get("contact_threshold_m", quality_config.contact_height_threshold)),
        "contact_floor_rule": str(corrected_context.get("floor_rule", "ground_zero")),
        "contact_floor_provenance_json": json.dumps(corrected_context.get("floor_provenance", {}), sort_keys=True),
        "contact_mesh_extent_blocker": bool(corrected_context.get("mesh_extent_blocker", False)),
        "contact_mesh_extent_status": str(corrected_context.get("mesh_extent_status", "")),
        "contact_collision_sphere_count": int(corrected_context.get("collision_sphere_count", 0)),
        "contact_audit_report_json": str(corrected_context.get("audit_report", "")),
        "contact_frame_ratio_pred": float(pred_contact.get("contact_frame_ratio", 0.0)),
        "contact_frame_ratio_ref": float(ref_contact.get("contact_frame_ratio", 0.0)),
        "contact_frame_ratio_delta": abs(float(pred_contact.get("contact_frame_ratio", 0.0)) - float(ref_contact.get("contact_frame_ratio", 0.0))),
        "contact_slide_mps_pred": float(pred_contact.get("max_contact_slide_speed", 0.0)),
        "contact_slide_mps_ref": float(ref_contact.get("max_contact_slide_speed", 0.0)),
        "contact_slide_delta_mps": abs(float(pred_contact.get("max_contact_slide_speed", 0.0)) - float(ref_contact.get("max_contact_slide_speed", 0.0))),
        "foot_penetration_depth_pred_m": float(pred_contact.get("penetration_depth", 0.0)),
        "foot_penetration_depth_ref_m": float(ref_contact.get("penetration_depth", 0.0)),
        "foot_p95_penetration_pred_m": float(pred_contact.get("p95_penetration", 0.0)),
        "foot_p95_penetration_ref_m": float(ref_contact.get("p95_penetration", 0.0)),
        "foot_below_floor_ratio_pred": float(pred_contact.get("below_floor_ratio", 0.0)),
        "foot_below_floor_ratio_ref": float(ref_contact.get("below_floor_ratio", 0.0)),
        "mean_foot_clearance_pred_m": float(pred_contact.get("mean_foot_clearance", 0.0)),
        "mean_foot_clearance_ref_m": float(ref_contact.get("mean_foot_clearance", 0.0)),
        "contact_lr_asymmetry_pred": float(pred_lr.get("lr_contact_ratio_asymmetry", 0.0)),
        "contact_lr_asymmetry_ref": float(ref_lr.get("lr_contact_ratio_asymmetry", 0.0)),
        "contact_lr_asymmetry_delta": abs(float(pred_lr.get("lr_contact_ratio_asymmetry", 0.0)) - float(ref_lr.get("lr_contact_ratio_asymmetry", 0.0))),
        "corrected_contact_frame_ratio_pred": float(pred_contact.get("contact_frame_ratio", 0.0)),
        "corrected_contact_frame_ratio_ref": float(ref_contact.get("contact_frame_ratio", 0.0)),
        "corrected_contact_slide_mps_pred": float(pred_contact.get("max_contact_slide_speed", 0.0)),
        "corrected_contact_slide_mps_ref": float(ref_contact.get("max_contact_slide_speed", 0.0)),
        "corrected_foot_penetration_depth_pred_m": float(pred_contact.get("penetration_depth", 0.0)),
        "corrected_foot_penetration_depth_ref_m": float(ref_contact.get("penetration_depth", 0.0)),
        "corrected_foot_p95_penetration_pred_m": float(pred_contact.get("p95_penetration", 0.0)),
        "corrected_foot_p95_penetration_ref_m": float(ref_contact.get("p95_penetration", 0.0)),
        "corrected_mean_foot_clearance_pred_m": float(pred_contact.get("mean_foot_clearance", 0.0)),
        "corrected_mean_foot_clearance_ref_m": float(ref_contact.get("mean_foot_clearance", 0.0)),
        "legacy_contact_frame_ratio_pred": float(legacy_pred_contact.get("contact_frame_ratio", 0.0)),
        "legacy_contact_frame_ratio_ref": float(legacy_ref_contact.get("contact_frame_ratio", 0.0)),
        "legacy_contact_frame_ratio_delta": abs(float(legacy_pred_contact.get("contact_frame_ratio", 0.0)) - float(legacy_ref_contact.get("contact_frame_ratio", 0.0))),
        "legacy_contact_slide_mps_pred": float(legacy_pred_contact.get("max_contact_slide_speed", 0.0)),
        "legacy_contact_slide_mps_ref": float(legacy_ref_contact.get("max_contact_slide_speed", 0.0)),
        "legacy_contact_slide_delta_mps": abs(float(legacy_pred_contact.get("max_contact_slide_speed", 0.0)) - float(legacy_ref_contact.get("max_contact_slide_speed", 0.0))),
        "legacy_foot_penetration_depth_pred_m": float(legacy_pred_contact.get("penetration_depth", 0.0)),
        "legacy_foot_penetration_depth_ref_m": float(legacy_ref_contact.get("penetration_depth", 0.0)),
        "legacy_mean_foot_clearance_pred_m": float(legacy_pred_contact.get("mean_foot_clearance", 0.0)),
        "legacy_mean_foot_clearance_ref_m": float(legacy_ref_contact.get("mean_foot_clearance", 0.0)),
        "legacy_contact_lr_asymmetry_pred": float(legacy_pred_lr.get("lr_contact_ratio_asymmetry", 0.0)),
        "legacy_contact_lr_asymmetry_ref": float(legacy_ref_lr.get("lr_contact_ratio_asymmetry", 0.0)),
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
        "legacy_contact_pred_json": json.dumps(round_float_mapping(legacy_pred_contact), sort_keys=True),
        "legacy_contact_ref_json": json.dumps(round_float_mapping(legacy_ref_contact), sort_keys=True),
        "legacy_contact_lr_json": json.dumps({"pred": legacy_pred_lr, "ref": legacy_ref_lr}, sort_keys=True),
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


def corrected_contact_context(config: Mapping[str, Any], evaluator: Mapping[str, Any]) -> dict[str, Any]:
    if evaluator.get("status") != "ok":
        return {"status": "blocked", "reason": "full_evaluator_unavailable"}
    model = evaluator["model"]
    model_xml = Path(str(evaluator["model_xml"]))
    audit_path = contact_audit_report_path(config, {})
    audit: dict[str, Any] = {}
    if audit_path.exists():
        try:
            audit = read_json(audit_path)
        except Exception:  # noqa: BLE001
            audit = {}
    selected = audit.get("selected_floor_convention", {})
    inventory = audit.get("mjcf_feature_inventory", {})
    spheres = load_foot_collision_spheres(model_xml, model.foot_body_names)
    ground_height = float(selected.get("ground_height_m", CORRECTED_CONTACT_GROUND_HEIGHT_M))
    contact_threshold = float(selected.get("contact_threshold_m", CORRECTED_CONTACT_THRESHOLD_M))
    definition = str(selected.get("definition", CORRECTED_CONTACT_HEIGHT_DEFINITION))
    floor_rule = str(selected.get("floor_rule", CORRECTED_CONTACT_FLOOR_RULE))
    mesh_extent_blocker = bool(selected.get("mesh_extent_blocker", True))
    status = "ok"
    reason = ""
    if definition != CORRECTED_CONTACT_HEIGHT_DEFINITION:
        status = "blocked"
        reason = "contact_height_definition_mismatch"
    elif len(spheres) != CORRECTED_CONTACT_EXPECTED_SPHERE_COUNT:
        status = "blocked"
        reason = "collision_sphere_count_mismatch"
    return {
        "status": status,
        "reason": reason,
        "evaluator_id": CORRECTED_CONTACT_EVALUATOR_ID,
        "height_definition": CORRECTED_CONTACT_HEIGHT_DEFINITION,
        "floor_rule": floor_rule,
        "ground_height_m": ground_height,
        "contact_threshold_m": contact_threshold,
        "audit_report": str(audit_path),
        "floor_provenance": {
            "source": "E_contact_metric_body_floor_audit",
            "audit_report": str(audit_path),
            "audit_status": audit.get("status", ""),
            "selected_floor_convention": selected,
        },
        "mesh_extent_blocker": mesh_extent_blocker,
        "mesh_extent_status": "git_lfs_pointer_no_mesh_vertices" if mesh_extent_blocker else "not_blocked",
        "collision_sphere_count": len(spheres),
        "audit_collision_sphere_count": int(inventory.get("foot_collision_sphere_count", 0) or 0),
        "spheres": spheres,
    }


def contact_audit_report_path(config: Mapping[str, Any], gate: Mapping[str, Any]) -> Path:
    value = (
        str(gate.get("path", "")).strip()
        or str(config.get("provenance", {}).get("inputs", {}).get("contact_metric_body_floor_audit_report_json", "")).strip()
        or str(DEFAULT_CONTACT_METRIC_BODY_FLOOR_AUDIT_REPORT)
    )
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def corrected_contact_stats(motion: Motion, model: Any, context: Mapping[str, Any], fps: float) -> dict[str, Any]:
    spheres = list(context.get("spheres", ()))
    ground_height = float(context.get("ground_height_m", CORRECTED_CONTACT_GROUND_HEIGHT_M))
    threshold = float(context.get("contact_threshold_m", CORRECTED_CONTACT_THRESHOLD_M))
    frame_min_heights: list[float] = []
    left_heights: list[float] = []
    right_heights: list[float] = []
    left_xy: list[tuple[float, float]] = []
    right_xy: list[tuple[float, float]] = []
    for frame_idx in range(motion.frame_count):
        transforms = g1_body_transforms(model, motion, frame_idx)
        side_features: dict[str, list[tuple[float, tuple[float, float]]]] = {"left": [], "right": []}
        for sphere in spheres:
            if sphere.body_name not in transforms:
                continue
            rot, body_pos = transforms[sphere.body_name]
            center = body_pos + rot @ sphere.local_pos
            bottom_z = float(center[2] - sphere.radius)
            side_features.setdefault(sphere.side, []).append((bottom_z, (float(center[0]), float(center[1]))))
        lows = [item for values in side_features.values() for item in values]
        frame_min_heights.append(min((item[0] for item in lows), default=float("nan")))
        for side, heights, xys in (("left", left_heights, left_xy), ("right", right_heights, right_xy)):
            values = side_features.get(side, [])
            if values:
                z, xy = min(values, key=lambda item: item[0])
                heights.append(z)
                xys.append(xy)
            else:
                heights.append(float("nan"))
                xys.append((float("nan"), float("nan")))

    clearances = finite_array([height - ground_height for height in frame_min_heights])
    penetration = np.maximum(0.0, -clearances)
    contact_flags = clearances <= threshold
    slide_speeds = corrected_contact_slide_speeds(
        left_heights,
        right_heights,
        left_xy,
        right_xy,
        ground_height=ground_height,
        threshold=threshold,
        fps=fps,
    )
    left_clearances = finite_array([height - ground_height for height in left_heights])
    right_clearances = finite_array([height - ground_height for height in right_heights])
    return {
        "definition": CORRECTED_CONTACT_HEIGHT_DEFINITION,
        "ground_height": ground_height,
        "contact_height_threshold": threshold,
        "min_foot_height": finite_stat_min(frame_min_heights),
        "mean_foot_clearance": finite_stat_mean(clearances.tolist()),
        "max_foot_clearance": finite_stat_max(clearances.tolist()),
        "below_floor_ratio": float(np.mean(clearances < 0.0)) if clearances.size else 0.0,
        "p95_penetration": float(np.percentile(penetration, 95)) if penetration.size else 0.0,
        "penetration_depth": finite_stat_max(penetration.tolist()),
        "contact_frame_ratio": float(np.mean(contact_flags)) if contact_flags.size else 0.0,
        "max_contact_slide_speed": finite_stat_max(slide_speeds.tolist()),
        "contact_slide_rate": float(np.mean(slide_speeds > 0.50)) if slide_speeds.size else 0.0,
        "left_contact_ratio": float(np.mean(left_clearances <= threshold)) if left_clearances.size else 0.0,
        "right_contact_ratio": float(np.mean(right_clearances <= threshold)) if right_clearances.size else 0.0,
        "lr_contact_ratio_asymmetry": abs(
            (float(np.mean(left_clearances <= threshold)) if left_clearances.size else 0.0)
            - (float(np.mean(right_clearances <= threshold)) if right_clearances.size else 0.0)
        ),
        "collision_sphere_count": len(spheres),
        "mesh_extent_blocker": bool(context.get("mesh_extent_blocker", False)),
    }


def corrected_contact_lr_asymmetry(stats: Mapping[str, Any]) -> dict[str, float]:
    return {
        "left_contact_ratio": float(stats.get("left_contact_ratio", 0.0)),
        "right_contact_ratio": float(stats.get("right_contact_ratio", 0.0)),
        "lr_contact_ratio_asymmetry": float(stats.get("lr_contact_ratio_asymmetry", 0.0)),
    }


def public_corrected_contact_context(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in context.items()
        if key != "spheres"
    } | {"collision_sphere_count": int(context.get("collision_sphere_count", 0))}


def corrected_contact_slide_speeds(
    left_heights: Sequence[float],
    right_heights: Sequence[float],
    left_xy: Sequence[tuple[float, float]],
    right_xy: Sequence[tuple[float, float]],
    *,
    ground_height: float,
    threshold: float,
    fps: float,
) -> np.ndarray:
    speeds: list[float] = []
    for heights, xys in ((left_heights, left_xy), (right_heights, right_xy)):
        for idx in range(1, len(heights)):
            prev_z = float(heights[idx - 1])
            cur_z = float(heights[idx])
            if not (math.isfinite(prev_z) and math.isfinite(cur_z)):
                continue
            if prev_z - ground_height > threshold or cur_z - ground_height > threshold:
                continue
            prev_xy = np.asarray(xys[idx - 1], dtype=np.float64)
            cur_xy = np.asarray(xys[idx], dtype=np.float64)
            if np.all(np.isfinite(prev_xy)) and np.all(np.isfinite(cur_xy)):
                speeds.append(float(np.linalg.norm(cur_xy - prev_xy) * fps))
    return np.asarray(speeds, dtype=np.float64)


def load_foot_collision_spheres(model_xml: Path, foot_body_names: Sequence[str]) -> list[FootCollisionSphere]:
    if not model_xml or not model_xml.exists():
        return []
    root = ET.parse(model_xml).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return []
    wanted = set(foot_body_names)
    spheres: list[FootCollisionSphere] = []

    def visit(element: ET.Element) -> None:
        if element.tag != "body":
            return
        body_name = element.attrib.get("name", "")
        if body_name in wanted:
            for geom in element.findall("geom"):
                geom_type = geom.attrib.get("type", "sphere")
                size = parse_float_tuple(geom.attrib.get("size", ""))
                if geom_type == "sphere" and size and not is_visual_geom(geom.attrib):
                    spheres.append(
                        FootCollisionSphere(
                            body_name=body_name,
                            side=side_from_name(body_name),
                            local_pos=np.asarray(parse_vec3(geom.attrib.get("pos", "0 0 0")), dtype=np.float64),
                            radius=float(size[0]),
                            attrs=dict(geom.attrib),
                        )
                    )
        for child in element:
            if child.tag == "body":
                visit(child)

    for child in worldbody:
        if child.tag == "body":
            visit(child)
    return spheres


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
    pred_root_euler: np.ndarray,
    ref_root_euler: np.ndarray,
) -> dict[str, Any]:
    """Compare FK points in world coordinates and in each motion's root-aligned frame.

    ``fk_world_*`` is the direct Euclidean distance between representative body
    points in the shared world frame.  ``fk_rootrel_*`` first maps each body
    point into that motion's own root frame with R_root.T @ (p_world - root).
    A known rigid root transform of an unchanged local pose should therefore
    change world FK but leave root-relative FK near zero.
    """

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
            pred_root_point = root_aligned_point(pred_point, pred_root[frame_idx], pred_root_euler[frame_idx])
            ref_root_point = root_aligned_point(ref_point, ref_root[frame_idx], ref_root_euler[frame_idx])
            rootrel_error = float(
                np.linalg.norm(pred_root_point - ref_root_point)
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


def root_aligned_point(point: np.ndarray, root_pos: np.ndarray, root_euler: np.ndarray) -> np.ndarray:
    rotation = euler_xyz_to_matrix(root_euler)
    return rotation.T @ (np.asarray(point, dtype=np.float64) - np.asarray(root_pos, dtype=np.float64))


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


def finite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def finite_stat_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def finite_stat_min(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.min(finite)) if finite else 0.0


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


def g1_body_transforms(model: Any, motion: Motion, frame_idx: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    joint_values = {
        column[:-4] if column.endswith("_dof") else column: float(value)
        for column, value in zip(G1_JOINT_COLUMNS, motion.dof[frame_idx])
    }
    transforms_list: list[tuple[np.ndarray, np.ndarray]] = []
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body in model.bodies:
        local_rotation = quat_wxyz_to_matrix(body.quat)
        local_position = np.asarray(body.pos, dtype=np.float64)
        if body.has_freejoint:
            local_rotation = euler_xyz_to_matrix(motion.root_euler[frame_idx])
            local_position = np.asarray(motion.root_pos[frame_idx], dtype=np.float64)
        if body.joint_name and body.joint_axis is not None:
            local_rotation = local_rotation @ axis_angle_matrix(
                np.asarray(body.joint_axis, dtype=np.float64),
                joint_values.get(body.joint_name, 0.0),
            )
        if body.parent is None:
            world_rotation = local_rotation
            world_position = local_position
        else:
            parent_rotation, parent_position = transforms_list[body.parent]
            world_rotation = parent_rotation @ local_rotation
            world_position = parent_position + parent_rotation @ local_position
        transforms_list.append((world_rotation, world_position))
        transforms[body.name] = (world_rotation, world_position)
    return transforms


def quat_wxyz_to_matrix(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(value) for value in quat[:4]]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.asarray(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def euler_xyz_to_matrix(euler_xyz: Sequence[float]) -> np.ndarray:
    rx, ry, rz = [float(value) for value in euler_xyz[:3]]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mat_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    mat_y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    mat_z = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mat_x @ mat_y @ mat_z


def parse_vec3(value: str) -> tuple[float, float, float]:
    parsed = parse_float_tuple(value)
    padded = (*parsed, 0.0, 0.0, 0.0)
    return (padded[0], padded[1], padded[2])


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part) for part in value.split()) if value else ()


def is_visual_geom(attrs: Mapping[str, str]) -> bool:
    return (
        attrs.get("contype") == "0"
        and attrs.get("conaffinity") == "0"
    ) or (
        attrs.get("group") == "1"
        and attrs.get("density") == "0"
    )


def side_from_name(*values: str) -> str:
    text = " ".join(values).lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return "center"


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
