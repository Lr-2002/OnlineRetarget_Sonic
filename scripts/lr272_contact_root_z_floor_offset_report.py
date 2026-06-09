#!/usr/bin/env python3
"""Offline LR-272 train-split contact/root-Z calibration report.

This script does not generate a retarget candidate.  It reads the official G1
CSV target and the current SOMA-retarget qpos for train rows, compares FK foot
clearance/contact distributions, and estimates one frozen additive root-Z
offset from train data only.
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
    fk_frames_for_motion,
    load_base_soma_motion,
    load_full_evaluator,
    load_official_motion,
    motion_metrics,
    read_stage_csv,
    row_key,
    rotation_geodesic_errors,
    round_nested,
    split_value,
    write_json,
)


DEFAULT_PAIRING_CSV = Path(
    "/home/user/project/OnlineRetarget/outputs/"
    "lr272_bones_seed_pairing_probe_20260608T133932Z/"
    "bones_seed_walk100_pairing.csv"
)
DEFAULT_G1_TAR = Path("/home/user/data/motion_data/g1.tar")
CONTACT_THRESHOLD_M = 0.04


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing-csv", type=Path, default=DEFAULT_PAIRING_CSV)
    parser.add_argument("--g1-tar", type=Path, default=DEFAULT_G1_TAR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-xml", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=0, help="Debug cap only; 0 means all train rows.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_stage_csv(args.pairing_csv)
    train_rows = [row for row in rows if split_value(row).lower() == "train"]
    if args.max_rows > 0:
        train_rows = train_rows[: args.max_rows]

    evaluator = load_full_evaluator(args.model_xml)
    if evaluator.get("status") != "ok":
        payload = {
            "status": "blocked",
            "reason": "full_evaluator_unavailable",
            "evaluator": public_evaluator(evaluator),
            "pairing_csv": str(args.pairing_csv),
            "train_rows_selected": len(train_rows),
        }
        write_json(args.output_dir / "contact_root_z_floor_offset_train_report.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    config = {"provenance": {"inputs": {"g1_tar": str(args.g1_tar)}}}
    model = evaluator["model"]
    quality_config = evaluator["config"]
    contact_threshold = float(getattr(quality_config, "contact_height_threshold", CONTACT_THRESHOLD_M))
    ground_height = float(getattr(quality_config, "ground_height", 0.0))

    frame_store = FrameStore()
    clip_records: list[dict[str, Any]] = []
    row_errors: list[dict[str, str]] = []
    for idx, row in enumerate(train_rows, start=1):
        key = row_key(row)
        try:
            official = load_official_motion(row, config)
            current = load_base_soma_motion(row, official.frame_count, official.fps)
            n = min(official.frame_count, current.frame_count)
            official = official.slice(n)
            current = current.slice(n)
            clip = summarize_clip(
                row,
                official,
                current,
                model,
                ground_height=ground_height,
                contact_threshold=contact_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            row_errors.append({"key": key, "error": repr(exc)})
            continue
        clip_records.append(clip)
        frame_store.extend(clip)
        if idx % 10 == 0:
            print(f"processed {idx}/{len(train_rows)} train rows", flush=True)

    if frame_store.frame_count == 0:
        payload = {
            "status": "blocked",
            "reason": "no_train_frames_measured",
            "pairing_csv": str(args.pairing_csv),
            "train_rows_selected": len(train_rows),
            "row_errors": row_errors,
        }
        write_json(args.output_dir / "contact_root_z_floor_offset_train_report.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    estimates = offset_estimates(frame_store, clip_records, contact_threshold)
    recommended = float(estimates["recommended_root_z_offset_m"])
    risk_rows = [
        offset_risk_row(name, float(value), frame_store, contact_threshold)
        for name, value in estimates["candidate_offsets_m"].items()
    ]
    risk_by_name = {str(row["offset_name"]): row for row in risk_rows}
    final = interpret_result(frame_store, clip_records, risk_by_name, recommended, contact_threshold)

    report = {
        "status": "complete" if not row_errors and len(clip_records) == len(train_rows) else "incomplete",
        "scope": "offline_train_split_report_only_no_candidate_no_smoke",
        "pairing_csv": str(args.pairing_csv),
        "g1_tar": str(args.g1_tar),
        "model_xml": str(evaluator["model_xml"]),
        "foot_bodies": list(model.foot_body_names),
        "ground_height_m": ground_height,
        "contact_height_threshold_m": contact_threshold,
        "rows_total": len(rows),
        "train_rows_selected": len(train_rows),
        "train_rows_measured": len(clip_records),
        "train_frames_measured": frame_store.frame_count,
        "row_errors": row_errors,
        "distribution_summary": distribution_summary(frame_store, contact_threshold),
        "offset_estimates": estimates,
        "offset_risk_table": risk_rows,
        "root_xy_rot_contract": {
            "root_z_offset_only": True,
            "root_xy_changed_by_offset": False,
            "root_rotation_changed_by_offset": False,
            "root_rel_xy_rmse_current_m": finite_percentile(frame_store.root_xy_rel_error_m, 50, default=0.0),
            "root_rel_xy_p95_current_m": finite_percentile(frame_store.root_xy_rel_error_m, 95, default=0.0),
            "root_rot_geodesic_rmse_current_rad": float(
                np.sqrt(np.mean(np.asarray(frame_store.root_rot_geodesic_rad, dtype=np.float64) ** 2))
            ),
            "root_rot_geodesic_p95_current_rad": finite_percentile(
                frame_store.root_rot_geodesic_rad, 95, default=0.0
            ),
        },
        "judgment": final,
        "artifacts": {
            "json": str(args.output_dir / "contact_root_z_floor_offset_train_report.json"),
            "clip_csv": str(args.output_dir / "train_clip_contact_root_z_summary.csv"),
            "risk_csv": str(args.output_dir / "offset_risk_table.csv"),
            "readme": str(args.output_dir / "README.md"),
        },
    }

    write_json(args.output_dir / "contact_root_z_floor_offset_train_report.json", report)
    write_csv(args.output_dir / "train_clip_contact_root_z_summary.csv", clip_records)
    write_csv(args.output_dir / "offset_risk_table.csv", risk_rows)
    write_readme(args.output_dir / "README.md", report)
    print(json.dumps(round_nested(report), indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


class FrameStore:
    def __init__(self) -> None:
        self.official_foot_clearance_m: list[float] = []
        self.current_soma_foot_clearance_m: list[float] = []
        self.official_body_clearance_m: list[float] = []
        self.current_soma_body_clearance_m: list[float] = []
        self.ref_minus_current_foot_clearance_m: list[float] = []
        self.current_minus_ref_foot_clearance_m: list[float] = []
        self.current_minus_ref_root_z_m: list[float] = []
        self.root_xy_rel_error_m: list[float] = []
        self.root_rot_geodesic_rad: list[float] = []

    @property
    def frame_count(self) -> int:
        return len(self.current_soma_foot_clearance_m)

    def extend(self, clip: Mapping[str, Any]) -> None:
        for key, target in (
            ("official_foot_clearance_m", self.official_foot_clearance_m),
            ("current_soma_foot_clearance_m", self.current_soma_foot_clearance_m),
            ("official_body_clearance_m", self.official_body_clearance_m),
            ("current_soma_body_clearance_m", self.current_soma_body_clearance_m),
            ("ref_minus_current_foot_clearance_m", self.ref_minus_current_foot_clearance_m),
            ("current_minus_ref_foot_clearance_m", self.current_minus_ref_foot_clearance_m),
            ("current_minus_ref_root_z_m", self.current_minus_ref_root_z_m),
            ("root_xy_rel_error_m", self.root_xy_rel_error_m),
            ("root_rot_geodesic_rad", self.root_rot_geodesic_rad),
        ):
            target.extend(float(value) for value in clip["_arrays"][key])


def summarize_clip(
    row: Mapping[str, str],
    official: Motion,
    current: Motion,
    model: Any,
    *,
    ground_height: float,
    contact_threshold: float,
) -> dict[str, Any]:
    official_fk = fk_frames_for_motion(official, model)
    current_fk = fk_frames_for_motion(current, model)
    official_foot = np.asarray(
        [lowest_clearance(frame, model.foot_body_names, ground_height) for frame in official_fk],
        dtype=np.float64,
    )
    current_foot = np.asarray(
        [lowest_clearance(frame, model.foot_body_names, ground_height) for frame in current_fk],
        dtype=np.float64,
    )
    all_bodies = tuple(body.name for body in model.bodies)
    official_body = np.asarray(
        [lowest_clearance(frame, all_bodies, ground_height) for frame in official_fk],
        dtype=np.float64,
    )
    current_body = np.asarray(
        [lowest_clearance(frame, all_bodies, ground_height) for frame in current_fk],
        dtype=np.float64,
    )
    ref_minus_current = official_foot - current_foot
    root_xy_ref = official.root_pos[:, :2] - official.root_pos[:1, :2]
    root_xy_current = current.root_pos[:, :2] - current.root_pos[:1, :2]
    root_xy_rel_error = np.linalg.norm(root_xy_current - root_xy_ref, axis=1)
    rot_geo = rotation_geodesic_errors(current.root_euler, official.root_euler)
    metrics = motion_metrics(current, official)
    return {
        "key": row_key(row),
        "split": split_value(row),
        "frames": official.frame_count,
        "fps": official.fps,
        "official_contact_ratio": contact_ratio(official_foot, contact_threshold),
        "current_contact_ratio": contact_ratio(current_foot, contact_threshold),
        "official_foot_clearance_mean_m": finite_mean(official_foot),
        "official_foot_clearance_p50_m": finite_percentile(official_foot, 50),
        "official_foot_clearance_p95_m": finite_percentile(official_foot, 95),
        "current_foot_clearance_mean_m": finite_mean(current_foot),
        "current_foot_clearance_p50_m": finite_percentile(current_foot, 50),
        "current_foot_clearance_p95_m": finite_percentile(current_foot, 95),
        "current_minus_official_foot_clearance_mean_m": finite_mean(current_foot - official_foot),
        "current_minus_official_foot_clearance_p50_m": finite_percentile(current_foot - official_foot, 50),
        "estimated_root_z_offset_median_m": finite_percentile(ref_minus_current, 50),
        "estimated_root_z_offset_mean_m": finite_mean(ref_minus_current),
        "current_minus_official_root_z_mean_m": finite_mean(current.root_pos[:, 2] - official.root_pos[:, 2]),
        "current_minus_official_root_z_p50_m": finite_percentile(current.root_pos[:, 2] - official.root_pos[:, 2], 50),
        "root_rel_rmse_m": float(metrics["root_rel_rmse_m"]),
        "root_rel_xy_error_p50_m": finite_percentile(root_xy_rel_error, 50),
        "root_rel_xy_error_p95_m": finite_percentile(root_xy_rel_error, 95),
        "root_rot_geodesic_rmse_rad": float(metrics["root_rot_geodesic_rmse_rad"]),
        "root_rot_geodesic_p95_rad": finite_percentile(rot_geo, 95),
        "_arrays": {
            "official_foot_clearance_m": official_foot.tolist(),
            "current_soma_foot_clearance_m": current_foot.tolist(),
            "official_body_clearance_m": official_body.tolist(),
            "current_soma_body_clearance_m": current_body.tolist(),
            "ref_minus_current_foot_clearance_m": ref_minus_current.tolist(),
            "current_minus_ref_foot_clearance_m": (current_foot - official_foot).tolist(),
            "current_minus_ref_root_z_m": (current.root_pos[:, 2] - official.root_pos[:, 2]).tolist(),
            "root_xy_rel_error_m": root_xy_rel_error.tolist(),
            "root_rot_geodesic_rad": rot_geo.tolist(),
        },
    }


def lowest_clearance(
    frame: Mapping[str, Sequence[tuple[float, float, float]]],
    body_names: Sequence[str],
    ground_height: float,
) -> float:
    values = [
        float(point[2]) - ground_height
        for body in body_names
        for point in frame.get(body, ())
    ]
    return min(values) if values else float("nan")


def offset_estimates(
    store: FrameStore,
    clip_records: Sequence[Mapping[str, Any]],
    contact_threshold: float,
) -> dict[str, Any]:
    ref = finite_array(store.official_foot_clearance_m)
    current = finite_array(store.current_soma_foot_clearance_m)
    ref_minus_current = finite_array(store.ref_minus_current_foot_clearance_m)
    current_minus_ref = finite_array(store.current_minus_ref_foot_clearance_m)
    ref_contact_ratio = contact_ratio(ref, contact_threshold)
    clip_medians = [
        float(row["estimated_root_z_offset_median_m"])
        for row in clip_records
        if np.isfinite(float(row["estimated_root_z_offset_median_m"]))
    ]
    contact_quantile = float(np.percentile(current, ref_contact_ratio * 100.0)) if current.size else 0.0
    candidates = {
        "none_current": 0.0,
        "frame_delta_mean_ref_minus_current": finite_mean(ref_minus_current),
        "frame_delta_median_ref_minus_current": finite_percentile(ref_minus_current, 50),
        "clip_delta_median_of_medians_ref_minus_current": finite_percentile(clip_medians, 50),
        "mean_clearance_match_ref_minus_current": finite_mean(ref) - finite_mean(current),
        "contact_ratio_quantile_match": contact_threshold - contact_quantile,
        "minus_0p10m_diagnostic": -0.10,
    }
    return {
        "definition": "root_z_offset_m is added to current SOMA root/body z; negative lowers the current retarget.",
        "recommended_method": "frame_delta_median_ref_minus_current",
        "recommended_root_z_offset_m": candidates["frame_delta_median_ref_minus_current"],
        "equivalent_floor_height_offset_m": -candidates["frame_delta_median_ref_minus_current"],
        "candidate_offsets_m": candidates,
        "frame_ref_minus_current_summary_m": numeric_summary(ref_minus_current),
        "frame_current_minus_ref_summary_m": numeric_summary(current_minus_ref),
        "clip_median_ref_minus_current_summary_m": numeric_summary(clip_medians),
    }


def offset_risk_row(
    name: str,
    offset: float,
    store: FrameStore,
    contact_threshold: float,
) -> dict[str, Any]:
    ref_foot = finite_array(store.official_foot_clearance_m)
    current_foot = finite_array(store.current_soma_foot_clearance_m)
    ref_body = finite_array(store.official_body_clearance_m)
    current_body = finite_array(store.current_soma_body_clearance_m)
    adjusted_foot = current_foot + offset
    adjusted_body = current_body + offset
    foot_penetration = np.maximum(0.0, -adjusted_foot)
    body_penetration = np.maximum(0.0, -adjusted_body)
    return {
        "offset_name": name,
        "root_z_offset_m": offset,
        "floor_height_equivalent_m": -offset,
        "official_contact_ratio": contact_ratio(ref_foot, contact_threshold),
        "adjusted_contact_ratio": contact_ratio(adjusted_foot, contact_threshold),
        "contact_ratio_abs_delta": abs(contact_ratio(adjusted_foot, contact_threshold) - contact_ratio(ref_foot, contact_threshold)),
        "official_mean_foot_clearance_m": finite_mean(ref_foot),
        "adjusted_mean_foot_clearance_m": finite_mean(adjusted_foot),
        "mean_foot_clearance_delta_m": finite_mean(adjusted_foot) - finite_mean(ref_foot),
        "official_p50_foot_clearance_m": finite_percentile(ref_foot, 50),
        "adjusted_p50_foot_clearance_m": finite_percentile(adjusted_foot, 50),
        "adjusted_foot_below_floor_ratio": below_floor_ratio(adjusted_foot),
        "adjusted_body_below_floor_ratio": below_floor_ratio(adjusted_body),
        "max_foot_penetration_m": finite_max(foot_penetration),
        "p95_foot_penetration_m": finite_percentile(foot_penetration, 95),
        "max_body_penetration_m": finite_max(body_penetration),
        "p95_body_penetration_m": finite_percentile(body_penetration, 95),
        "official_foot_below_floor_ratio": below_floor_ratio(ref_foot),
        "official_body_below_floor_ratio": below_floor_ratio(ref_body),
        "official_max_foot_penetration_m": finite_max(np.maximum(0.0, -ref_foot)),
        "official_max_body_penetration_m": finite_max(np.maximum(0.0, -ref_body)),
    }


def distribution_summary(store: FrameStore, contact_threshold: float) -> dict[str, Any]:
    return {
        "official_g1_foot_clearance_m": numeric_summary(store.official_foot_clearance_m),
        "current_soma_retarget_foot_clearance_m": numeric_summary(store.current_soma_foot_clearance_m),
        "current_minus_official_foot_clearance_m": numeric_summary(store.current_minus_ref_foot_clearance_m),
        "official_g1_body_lowest_clearance_m": numeric_summary(store.official_body_clearance_m),
        "current_soma_body_lowest_clearance_m": numeric_summary(store.current_soma_body_clearance_m),
        "current_minus_official_root_z_m": numeric_summary(store.current_minus_ref_root_z_m),
        "root_rel_xy_error_m": numeric_summary(store.root_xy_rel_error_m),
        "root_rot_geodesic_rad": numeric_summary(store.root_rot_geodesic_rad),
        "official_contact_ratio": contact_ratio(store.official_foot_clearance_m, contact_threshold),
        "current_soma_contact_ratio": contact_ratio(store.current_soma_foot_clearance_m, contact_threshold),
        "contact_ratio_delta_current_abs": abs(
            contact_ratio(store.current_soma_foot_clearance_m, contact_threshold)
            - contact_ratio(store.official_foot_clearance_m, contact_threshold)
        ),
    }


def interpret_result(
    store: FrameStore,
    clip_records: Sequence[Mapping[str, Any]],
    risk_by_name: Mapping[str, Mapping[str, Any]],
    recommended: float,
    contact_threshold: float,
) -> dict[str, Any]:
    current_minus_ref = finite_array(store.current_minus_ref_foot_clearance_m)
    clip_offsets = finite_array([float(row["estimated_root_z_offset_median_m"]) for row in clip_records])
    rec_risk = risk_by_name.get("frame_delta_median_ref_minus_current", {})
    current_float_median = finite_percentile(current_minus_ref, 50)
    current_float_mean = finite_mean(current_minus_ref)
    explains_10cm = 0.06 <= current_float_median <= 0.14 and -0.14 <= recommended <= -0.06
    penetration_bounded = (
        float(rec_risk.get("max_body_penetration_m", 999.0)) <= 0.08
        and float(rec_risk.get("p95_body_penetration_m", 999.0)) <= 0.03
    )
    contact_delta_ok = float(rec_risk.get("contact_ratio_abs_delta", 999.0)) <= 0.15
    clip_stability_ok = (
        finite_percentile(clip_offsets, 75) - finite_percentile(clip_offsets, 25)
    ) <= 0.05
    return {
        "one_frozen_train_split_offset_explains_float": bool(
            explains_10cm and penetration_bounded and contact_delta_ok
        ),
        "recommended_root_z_offset_m": recommended,
        "equivalent_floor_height_offset_m": -recommended,
        "current_minus_official_foot_clearance_median_m": current_float_median,
        "current_minus_official_foot_clearance_mean_m": current_float_mean,
        "clip_offset_iqr_m": finite_percentile(clip_offsets, 75) - finite_percentile(clip_offsets, 25),
        "recommended_offset_contact_ratio_abs_delta": rec_risk.get("contact_ratio_abs_delta"),
        "recommended_offset_max_body_penetration_m": rec_risk.get("max_body_penetration_m"),
        "recommended_offset_p95_body_penetration_m": rec_risk.get("p95_body_penetration_m"),
        "root_xy_rot_unchanged_by_construction": True,
        "preconditions": {
            "float_gap_near_0p10m": bool(explains_10cm),
            "penetration_bounded": bool(penetration_bounded),
            "contact_ratio_within_0p15": bool(contact_delta_ok),
            "clip_offset_iqr_within_0p05m": bool(clip_stability_ok),
        },
        "review_note": (
            "This is an offline calibration judgment only. It does not implement "
            "d_contact_root_z_floor_offset_train_split_v1 or run smoke/mixed10/walk100."
        ),
        "contact_threshold_m": contact_threshold,
    }


def numeric_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    arr = finite_array(values)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def finite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def finite_mean(values: Sequence[float] | np.ndarray) -> float:
    arr = finite_array(values)
    return float(np.mean(arr)) if arr.size else 0.0


def finite_max(values: Sequence[float] | np.ndarray) -> float:
    arr = finite_array(values)
    return float(np.max(arr)) if arr.size else 0.0


def finite_percentile(values: Sequence[float] | np.ndarray, percentile: float, *, default: float = 0.0) -> float:
    arr = finite_array(values)
    return float(np.percentile(arr, percentile)) if arr.size else default


def contact_ratio(clearance: Sequence[float] | np.ndarray, threshold: float) -> float:
    arr = finite_array(clearance)
    return float(np.mean(arr <= threshold)) if arr.size else 0.0


def below_floor_ratio(clearance: Sequence[float] | np.ndarray) -> float:
    arr = finite_array(clearance)
    return float(np.mean(arr < 0.0)) if arr.size else 0.0


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key.startswith("_"):
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def write_readme(path: Path, report: Mapping[str, Any]) -> None:
    judgment = report["judgment"]
    lines = [
        "# LR-272 contact/root-Z floor offset offline report",
        "",
        "Scope: train-split calibration report only. No candidate implementation and no smoke/mixed10/walk100 run.",
        "",
        f"Rows measured: {report['train_rows_measured']} / {report['train_rows_selected']}",
        f"Frames measured: {report['train_frames_measured']}",
        f"Recommended additive root-Z offset: {judgment['recommended_root_z_offset_m']:.6f} m",
        f"Equivalent floor/contact height offset: {judgment['equivalent_floor_height_offset_m']:.6f} m",
        f"One frozen offset explains float: {judgment['one_frozen_train_split_offset_explains_float']}",
        "",
        "Primary artifacts:",
        "- contact_root_z_floor_offset_train_report.json",
        "- train_clip_contact_root_z_summary.csv",
        "- offset_risk_table.csv",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def public_evaluator(evaluator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in evaluator.items()
        if key not in {"model", "config"}
    }


if __name__ == "__main__":
    raise SystemExit(main())
