#!/usr/bin/env python3
"""Prove LR-272 G1 FK evaluator frame consistency on paired BONES/SOMA clips.

This script does not create or test a new retargeting candidate.  It validates
the evaluator contract with identity/null and known rigid root transforms, then
writes a mixed10 per-body sanity table for selected G1 body groups.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lr272_bones_soma_candidate_runner import (  # noqa: E402
    Motion,
    body_representative,
    fk_mpjpe_metrics,
    g1_fk_body_positions,
    load_candidate_csv,
    load_full_evaluator,
    load_official_motion,
    motion_metrics,
    motion_to_parsed_frames,
    read_json,
    read_stage_csv,
    root_aligned_point,
    rotate_xy_array,
    row_key,
    safe_name,
    write_json,
)


FRAME_DEFINITIONS = {
    "fk_world": (
        "Euclidean distance between representative FK body points in the shared world frame. "
        "This metric is expected to change under a known rigid root translation/yaw."
    ),
    "fk_rootrel": (
        "Euclidean distance after each FK body point is mapped into that motion's own root-aligned frame: "
        "p_root = R_root.T @ (p_world - root_world). This metric should remain invariant under a known "
        "rigid root transform when local joint pose is unchanged."
    ),
    "body_representative": "Mean of the FK points emitted for a body by the G1 MJCF helper.",
}

BODY_GROUPS = {
    "pelvis": ("pelvis",),
    "waist": ("waist_yaw_link", "waist_roll_link", "torso_link"),
    "feet": (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_toe_link",
        "right_toe_link",
    ),
    "hands": (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
    ),
}

INVARIANCE_TOL = 1e-8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-xml", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--rigid-yaw-rad", type=float, default=0.7)
    parser.add_argument("--rigid-translation-m", type=float, nargs=3, default=(1.25, -0.45, 0.08))
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=RETARGET_CSV_DIR",
        help="Optional actual candidate directory for per-body mixed10 sanity rows.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = read_json(args.config)
    rows = read_stage_csv(args.stage_csv)
    if not rows:
        raise SystemExit(f"stage CSV has no rows: {args.stage_csv}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = load_full_evaluator(args.model_xml)
    report: dict[str, Any] = {
        "schema_version": 1,
        "config": str(args.config),
        "stage_csv": str(args.stage_csv),
        "output_dir": str(args.output_dir),
        "frame_definitions": FRAME_DEFINITIONS,
        "body_groups": {key: list(value) for key, value in BODY_GROUPS.items()},
        "rigid_transform": {
            "yaw_rad": args.rigid_yaw_rad,
            "translation_m": list(args.rigid_translation_m),
        },
        "evaluator": public_evaluator_report(evaluator),
        "row_count": len(rows),
    }
    if evaluator.get("status") != "ok":
        report["status"] = "blocked"
        report["reason"] = evaluator.get("reason", "evaluator_unavailable")
        write_json(args.output_dir / "frame_consistency_report.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    candidate_dirs = parse_candidate_dirs(args.candidate)
    identity_records: list[dict[str, Any]] = []
    rigid_records: list[dict[str, Any]] = []
    per_body_rows: list[dict[str, Any]] = []

    for row in rows:
        key = row_key(row)
        official = load_official_motion(row, config)
        if args.max_frames > 0:
            official = official.slice(args.max_frames)

        identity = evaluator_record("identity_null", key, official, official, evaluator)
        identity_records.append(identity)
        per_body_rows.extend(per_body_error_rows("identity_null", key, official, official, evaluator))

        rigid_motion = rigid_root_transform(
            official,
            yaw_rad=args.rigid_yaw_rad,
            translation=np.asarray(args.rigid_translation_m, dtype=np.float64),
        )
        rigid = evaluator_record("known_rigid_root_transform", key, rigid_motion, official, evaluator)
        rigid_records.append(rigid)
        per_body_rows.extend(per_body_error_rows("known_rigid_root_transform", key, rigid_motion, official, evaluator))

        for label, csv_dir in candidate_dirs.items():
            candidate_csv = csv_dir / f"{safe_name(key)}.csv"
            if not candidate_csv.exists():
                per_body_rows.append(
                    {
                        "scenario": f"candidate:{label}",
                        "key": key,
                        "group": "",
                        "body": "",
                        "status": "missing_candidate_csv",
                        "candidate_csv": str(candidate_csv),
                    }
                )
                continue
            candidate_motion = load_candidate_csv(candidate_csv, official.fps).slice(official.frame_count)
            per_body_rows.extend(per_body_error_rows(f"candidate:{label}", key, candidate_motion, official, evaluator))

    identity_summary = summarize_records(identity_records)
    rigid_summary = summarize_records(rigid_records)
    pass_checks = {
        "identity_world_max_le_tol": identity_summary.get("fk_world_max_m", 0.0) <= INVARIANCE_TOL,
        "identity_rootrel_max_le_tol": identity_summary.get("fk_rootrel_max_m", 0.0) <= INVARIANCE_TOL,
        "identity_dof_max_le_tol": identity_summary.get("dof_max_abs_rad", 0.0) <= INVARIANCE_TOL,
        "rigid_rootrel_max_le_tol": rigid_summary.get("fk_rootrel_max_m", 0.0) <= INVARIANCE_TOL,
        "rigid_dof_max_le_tol": rigid_summary.get("dof_max_abs_rad", 0.0) <= INVARIANCE_TOL,
        "rigid_world_changed": rigid_summary.get("fk_world_mpjpe_m", 0.0) > INVARIANCE_TOL,
    }
    report.update(
        {
            "status": "passed" if all(pass_checks.values()) else "failed",
            "identity_summary": identity_summary,
            "known_rigid_root_transform_summary": rigid_summary,
            "pass_checks": pass_checks,
            "identity_csv": str(args.output_dir / "identity_null_metrics.csv"),
            "known_rigid_csv": str(args.output_dir / "known_rigid_root_transform_metrics.csv"),
            "per_body_mixed10_csv": str(args.output_dir / "per_body_mixed10_sanity_table.csv"),
        }
    )

    write_metric_csv(args.output_dir / "identity_null_metrics.csv", identity_records)
    write_metric_csv(args.output_dir / "known_rigid_root_transform_metrics.csv", rigid_records)
    write_metric_csv(args.output_dir / "per_body_mixed10_sanity_table.csv", per_body_rows)
    write_json(args.output_dir / "frame_consistency_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


def evaluator_record(
    scenario: str,
    key: str,
    pred: Motion,
    ref: Motion,
    evaluator: Mapping[str, Any],
) -> dict[str, Any]:
    basic = motion_metrics(pred, ref)
    fk = fk_metrics_only(pred, ref, evaluator)
    return {
        "scenario": scenario,
        "key": key,
        **basic,
        **fk,
    }


def fk_metrics_only(pred: Motion, ref: Motion, evaluator: Mapping[str, Any]) -> dict[str, Any]:
    model = evaluator["model"]
    n = min(pred.frame_count, ref.frame_count)
    pred = pred.slice(n)
    ref = ref.slice(n)
    pred_fk = fk_frames(pred, model)
    ref_fk = fk_frames(ref, model)
    return fk_mpjpe_metrics(pred_fk, ref_fk, pred.root_pos, ref.root_pos, pred.root_euler, ref.root_euler)


def fk_frames(motion: Motion, model: Any) -> list[Mapping[str, Sequence[tuple[float, float, float]]]]:
    return [
        g1_fk_body_positions(model, joints, root, root_euler)
        for joints, root, root_euler, _ in motion_to_parsed_frames(motion)
    ]


def per_body_error_rows(
    scenario: str,
    key: str,
    pred: Motion,
    ref: Motion,
    evaluator: Mapping[str, Any],
) -> list[dict[str, Any]]:
    model = evaluator["model"]
    n = min(pred.frame_count, ref.frame_count)
    pred = pred.slice(n)
    ref = ref.slice(n)
    pred_fk = fk_frames(pred, model)
    ref_fk = fk_frames(ref, model)
    rows: list[dict[str, Any]] = []
    for group, body_names in BODY_GROUPS.items():
        for body in body_names:
            world_errors: list[float] = []
            rootrel_errors: list[float] = []
            pred_root_norms: list[float] = []
            ref_root_norms: list[float] = []
            for frame_idx, (pred_frame, ref_frame) in enumerate(zip(pred_fk, ref_fk)):
                pred_point = body_representative(pred_frame.get(body, ()))
                ref_point = body_representative(ref_frame.get(body, ()))
                if pred_point is None or ref_point is None:
                    continue
                pred_root_point = root_aligned_point(pred_point, pred.root_pos[frame_idx], pred.root_euler[frame_idx])
                ref_root_point = root_aligned_point(ref_point, ref.root_pos[frame_idx], ref.root_euler[frame_idx])
                world_errors.append(float(np.linalg.norm(pred_point - ref_point)))
                rootrel_errors.append(float(np.linalg.norm(pred_root_point - ref_root_point)))
                pred_root_norms.append(float(np.linalg.norm(pred_root_point)))
                ref_root_norms.append(float(np.linalg.norm(ref_root_point)))
            if not world_errors:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "key": key,
                    "group": group,
                    "body": body,
                    "status": "ok",
                    "frames": len(world_errors),
                    "world_mean_m": float(np.mean(world_errors)),
                    "world_p95_m": float(np.percentile(world_errors, 95)),
                    "world_max_m": float(np.max(world_errors)),
                    "rootrel_mean_m": float(np.mean(rootrel_errors)),
                    "rootrel_p95_m": float(np.percentile(rootrel_errors, 95)),
                    "rootrel_max_m": float(np.max(rootrel_errors)),
                    "pred_rootrel_norm_mean_m": float(np.mean(pred_root_norms)),
                    "ref_rootrel_norm_mean_m": float(np.mean(ref_root_norms)),
                }
            )
    return rows


def rigid_root_transform(motion: Motion, *, yaw_rad: float, translation: np.ndarray) -> Motion:
    root_pos = motion.root_pos.copy()
    origin_xy = root_pos[:1, :2]
    root_pos[:, :2] = origin_xy + rotate_xy_array(root_pos[:, :2] - origin_xy, yaw_rad)
    root_pos += translation.reshape(1, 3)
    root_euler = motion.root_euler.copy()
    root_euler[:, 2] += yaw_rad
    return Motion(fps=motion.fps, root_pos=root_pos, root_euler=root_euler, dof=motion.dof.copy())


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    keys = (
        "root_rel_rmse_m",
        "root_rot_geodesic_rmse_rad",
        "dof_rmse_rad",
        "dof_max_abs_rad",
        "fk_world_mpjpe_m",
        "fk_world_max_m",
        "fk_rootrel_mpjpe_m",
        "fk_rootrel_max_m",
    )
    summary: dict[str, float] = {}
    for key in keys:
        values = [float(record[key]) for record in records if key in record]
        if values:
            summary[key] = float(np.max(values) if key.endswith("_max_m") or key.endswith("_max_abs_rad") else np.mean(values))
    return summary


def parse_candidate_dirs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--candidate must be LABEL=RETARGET_CSV_DIR, got {value!r}")
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"--candidate label is empty in {value!r}")
        result[label] = Path(path)
    return result


def public_evaluator_report(evaluator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in evaluator.items()
        if key not in {"model", "config"}
    }


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


if __name__ == "__main__":
    raise SystemExit(main())
