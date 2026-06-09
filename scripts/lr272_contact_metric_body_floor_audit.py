#!/usr/bin/env python3
"""Report-only LR-272 contact metric/body/floor audit.

The audit recomputes official G1 contact on train rows plus the current smoke
row under several MJCF body/geom/site definitions.  It does not write retarget
motions, implement candidates, or launch smoke/mixed/walk evaluation commands.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import struct
import sys
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lr272_bones_soma_candidate_runner import (  # noqa: E402
    G1_JOINT_COLUMNS,
    Motion,
    load_full_evaluator,
    load_official_motion,
    read_stage_csv,
    row_key,
    safe_name,
    split_value,
    write_json,
)


DEFAULT_PAIRING_CSV = Path(
    "/home/user/project/OnlineRetarget/outputs/"
    "lr272_bones_seed_pairing_probe_20260608T133932Z/"
    "bones_seed_walk100_pairing.csv"
)
DEFAULT_SMOKE_CSV = Path(
    "/home/user/project/OnlineRetarget/outputs/"
    "lr272_bones_soma_dof_map_train_split_v1_20260608T185456Z/"
    "stages/smoke1.csv"
)
DEFAULT_G1_TAR = Path("/home/user/data/motion_data/g1.tar")
CONTACT_THRESHOLD_M = 0.04


@dataclass(frozen=True)
class GeomSpec:
    name: str
    body_name: str
    side: str
    geom_type: str
    pos: np.ndarray
    quat: np.ndarray
    size: tuple[float, ...]
    mesh_name: str
    mesh_file: str
    collision: bool
    visual: bool
    extent_status: str
    mesh_vertices: np.ndarray
    attrs: Mapping[str, str]


@dataclass(frozen=True)
class SiteSpec:
    name: str
    body_name: str
    side: str
    pos: np.ndarray
    attrs: Mapping[str, str]


@dataclass(frozen=True)
class MJCFFeatures:
    geoms: tuple[GeomSpec, ...]
    sites: tuple[SiteSpec, ...]
    meshdir: str
    mesh_load_summary: Mapping[str, Any]


@dataclass(frozen=True)
class FrameFeature:
    z: float
    xy: tuple[float, float]
    name: str


@dataclass
class DefinitionStore:
    z: list[float]
    left_z: list[float]
    right_z: list[float]
    left_xy: list[tuple[float, float]]
    right_xy: list[tuple[float, float]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing-csv", type=Path, default=DEFAULT_PAIRING_CSV)
    parser.add_argument("--smoke-csv", type=Path, default=DEFAULT_SMOKE_CSV)
    parser.add_argument("--g1-tar", type=Path, default=DEFAULT_G1_TAR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-xml", type=Path, default=None)
    parser.add_argument("--contact-threshold-m", type=float, default=CONTACT_THRESHOLD_M)
    parser.add_argument("--random-seed", type=int, default=272)
    parser.add_argument("--max-train-rows", type=int, default=0, help="Debug cap only; 0 means all train rows.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = load_full_evaluator(args.model_xml)
    if evaluator.get("status") != "ok":
        payload = {
            "status": "blocked",
            "reason": "full_evaluator_unavailable",
            "evaluator": public_evaluator(evaluator),
        }
        write_json(args.output_dir / "contact_metric_body_floor_audit_report.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    model = evaluator["model"]
    model_xml = Path(evaluator["model_xml"])
    features = parse_mjcf_features(model_xml)
    definitions = build_definitions(model, features)
    rows = read_stage_csv(args.pairing_csv)
    train_rows = [row for row in rows if split_value(row).lower() == "train"]
    if args.max_train_rows > 0:
        train_rows = train_rows[: args.max_train_rows]
    smoke_rows = read_stage_csv(args.smoke_csv) if args.smoke_csv.exists() else []
    eval_rows = [("train", row) for row in train_rows] + [("smoke", row) for row in smoke_rows]
    config = {"provenance": {"inputs": {"g1_tar": str(args.g1_tar)}}}

    clip_records: list[dict[str, Any]] = []
    series_by_clip: dict[str, dict[str, DefinitionStore]] = {}
    errors: list[dict[str, str]] = []
    for idx, (subset, row) in enumerate(eval_rows, start=1):
        key = row_key(row)
        try:
            official = load_official_motion(row, config)
            clip_series = evaluate_motion(official, model, definitions)
        except Exception as exc:  # noqa: BLE001
            errors.append({"subset": subset, "key": key, "error": repr(exc)})
            continue
        series_key = f"{subset}:{key}"
        series_by_clip[series_key] = clip_series
        for name, store in clip_series.items():
            record = clip_metrics(
                subset=subset,
                row=row,
                definition=name,
                store=store,
                fps=official.fps,
                ground_height=0.0,
                contact_threshold=args.contact_threshold_m,
            )
            clip_records.append(record)
        if idx % 10 == 0:
            print(f"processed {idx}/{len(eval_rows)} official clips", flush=True)

    if not clip_records:
        payload = {
            "status": "blocked",
            "reason": "no_official_clips_measured",
            "row_errors": errors,
        }
        write_json(args.output_dir / "contact_metric_body_floor_audit_report.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    train_summary = aggregate_definition_records(
        [row for row in clip_records if row["subset"] == "train"],
        ground_height=0.0,
        contact_threshold=args.contact_threshold_m,
    )
    floor_candidates = build_floor_candidates(
        series_by_clip,
        definitions,
        contact_threshold=args.contact_threshold_m,
    )
    selected = select_floor_convention(floor_candidates, features)
    overlay_dir = args.output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_records = write_overlays(
        overlay_dir,
        series_by_clip,
        clip_records,
        selected,
        contact_threshold=args.contact_threshold_m,
        random_seed=args.random_seed,
    )
    geometry_plot = overlay_dir / "mjcf_foot_geometry.svg"
    write_mjcf_geometry_svg(geometry_plot, features, tuple(model.foot_body_names))

    report = {
        "status": "complete" if not errors else "incomplete",
        "scope": "report_only_E_contact_metric_body_floor_audit_no_motion_transform_no_candidate_no_smoke",
        "pairing_csv": str(args.pairing_csv),
        "smoke_csv": str(args.smoke_csv),
        "g1_tar": str(args.g1_tar),
        "model_xml": str(model_xml),
        "contact_threshold_m": args.contact_threshold_m,
        "rows": {
            "pairing_total": len(rows),
            "train_selected": len(train_rows),
            "smoke_selected": len(smoke_rows),
            "clip_records": len(clip_records),
            "row_errors": errors,
        },
        "mjcf_feature_inventory": feature_inventory(model, features, definitions),
        "definition_train_summary_ground0": train_summary,
        "floor_convention_candidates": floor_candidates,
        "selected_floor_convention": selected,
        "smoke_summary": [
            row for row in clip_records if row["subset"] == "smoke"
        ],
        "visual_diagnostics": {
            "isaac_mesh_status": "not_run_report_only; diagnostic SVG overlays emitted instead",
            "overlay_records": overlay_records,
            "mjcf_foot_geometry_svg": str(geometry_plot),
        },
        "artifacts": {
            "report_json": str(args.output_dir / "contact_metric_body_floor_audit_report.json"),
            "clip_csv": str(args.output_dir / "clip_contact_metric_body_floor_audit.csv"),
            "definition_train_csv": str(args.output_dir / "definition_train_summary_ground0.csv"),
            "floor_candidates_csv": str(args.output_dir / "floor_convention_candidates.csv"),
            "overlay_dir": str(overlay_dir),
            "readme": str(args.output_dir / "README.md"),
        },
    }

    write_json(args.output_dir / "contact_metric_body_floor_audit_report.json", report)
    write_csv(args.output_dir / "clip_contact_metric_body_floor_audit.csv", clip_records)
    write_csv(args.output_dir / "definition_train_summary_ground0.csv", list(train_summary.values()))
    write_csv(args.output_dir / "floor_convention_candidates.csv", flatten_floor_candidates(floor_candidates))
    write_readme(args.output_dir / "README.md", report)
    print(json.dumps(round_nested(report), indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 1


def parse_mjcf_features(model_xml: Path) -> MJCFFeatures:
    root = ET.parse(model_xml).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    mesh_base = (model_xml.parent / meshdir).resolve() if meshdir else model_xml.parent
    mesh_files: dict[str, str] = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            mesh_name = mesh.attrib.get("name") or Path(mesh.attrib.get("file", "")).stem
            mesh_files[mesh_name] = mesh.attrib.get("file", "")
    geoms: list[GeomSpec] = []
    sites: list[SiteSpec] = []
    mesh_status: dict[str, Any] = {}
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"MJCF missing worldbody: {model_xml}")

    def visit_body(element: ET.Element) -> None:
        body_name = element.attrib.get("name", "")
        for geom_idx, geom in enumerate(element.findall("geom")):
            geom_type = geom.attrib.get("type", "sphere")
            pos = np.asarray(parse_vec3(geom.attrib.get("pos", "0 0 0")), dtype=np.float64)
            quat = np.asarray(parse_quat(geom.attrib.get("quat", "1 0 0 0")), dtype=np.float64)
            size = parse_floats(geom.attrib.get("size", ""))
            mesh_name = geom.attrib.get("mesh", "")
            mesh_file = mesh_files.get(mesh_name, "")
            mesh_vertices = np.empty((0, 3), dtype=np.float64)
            extent_status = "analytic_primitive"
            if geom_type == "mesh" or mesh_name:
                mesh_path = (mesh_base / mesh_file).resolve() if mesh_file else Path()
                mesh_vertices, extent_status = load_stl_vertices(mesh_path)
                mesh_status[mesh_name or f"{body_name}_geom{geom_idx}"] = {
                    "mesh_file": str(mesh_path) if mesh_file else "",
                    "extent_status": extent_status,
                    "vertex_count": int(mesh_vertices.shape[0]),
                }
            visual = is_visual_geom(geom.attrib)
            name = geom.attrib.get("name") or f"{body_name}_geom{geom_idx}"
            geoms.append(
                GeomSpec(
                    name=name,
                    body_name=body_name,
                    side=side_from_name(body_name, name),
                    geom_type=geom_type,
                    pos=pos,
                    quat=quat,
                    size=size,
                    mesh_name=mesh_name,
                    mesh_file=mesh_file,
                    collision=not visual,
                    visual=visual,
                    extent_status=extent_status,
                    mesh_vertices=mesh_vertices,
                    attrs=dict(geom.attrib),
                )
            )
        for site in element.findall("site"):
            name = site.attrib.get("name", f"{body_name}_site{len(sites)}")
            sites.append(
                SiteSpec(
                    name=name,
                    body_name=body_name,
                    side=side_from_name(body_name, name),
                    pos=np.asarray(parse_vec3(site.attrib.get("pos", "0 0 0")), dtype=np.float64),
                    attrs=dict(site.attrib),
                )
            )
        for child in element:
            if child.tag == "body":
                visit_body(child)

    for child in worldbody:
        if child.tag == "body":
            visit_body(child)
    return MJCFFeatures(
        geoms=tuple(geoms),
        sites=tuple(sites),
        meshdir=str(mesh_base),
        mesh_load_summary=mesh_status,
    )


def build_definitions(model: Any, features: MJCFFeatures) -> dict[str, Mapping[str, Any]]:
    foot_bodies = tuple(model.foot_body_names)
    body_names = tuple(body.name for body in model.bodies)
    toe_sole_bodies = tuple(
        name
        for name in body_names
        if any(token in name.lower() for token in ("toe", "sole", "foot"))
        and name not in foot_bodies
    )
    foot_collision_geoms = tuple(
        geom
        for geom in features.geoms
        if geom.body_name in foot_bodies and geom.collision
    )
    foot_collision_spheres = tuple(
        geom
        for geom in foot_collision_geoms
        if geom.geom_type in ("sphere", "") and geom.size
    )
    foot_sites = tuple(
        site
        for site in features.sites
        if any(token in f"{site.name} {site.body_name}".lower() for token in ("foot", "toe", "sole", "ankle"))
    )
    return {
        "ankle_body_origin_z": {
            "kind": "body_origin",
            "body_names": foot_bodies,
            "feature_count": len(foot_bodies),
            "status": "ok" if foot_bodies else "missing",
        },
        "ankle_body_current_points_min_z": {
            "kind": "body_geom_points_no_radius",
            "body_names": foot_bodies,
            "feature_count": len(foot_bodies),
            "status": "ok" if foot_bodies else "missing",
        },
        "foot_collision_geom_min_z": {
            "kind": "collision_geoms_with_available_extents",
            "geoms": foot_collision_geoms,
            "feature_count": len(foot_collision_geoms),
            "status": "ok" if foot_collision_geoms else "missing",
        },
        "foot_collision_sphere_bottom_min_z": {
            "kind": "collision_sphere_bottoms",
            "geoms": foot_collision_spheres,
            "feature_count": len(foot_collision_spheres),
            "status": "ok" if foot_collision_spheres else "missing",
        },
        "toe_sole_body_min_z": {
            "kind": "toe_sole_body_geom_points",
            "body_names": toe_sole_bodies,
            "feature_count": len(toe_sole_bodies),
            "status": "ok" if toe_sole_bodies else "missing_no_toe_or_sole_bodies_in_mjcf",
        },
        "mjcf_foot_site_min_z": {
            "kind": "foot_sites",
            "sites": foot_sites,
            "feature_count": len(foot_sites),
            "status": "ok" if foot_sites else "missing_no_foot_sites_in_mjcf",
        },
    }


def evaluate_motion(
    motion: Motion,
    model: Any,
    definitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, DefinitionStore]:
    stores = {
        name: DefinitionStore(z=[], left_z=[], right_z=[], left_xy=[], right_xy=[])
        for name in definitions
    }
    for frame_idx in range(motion.frame_count):
        transforms = body_transforms(model, motion, frame_idx)
        for name, definition in definitions.items():
            side_features = definition_side_features(model, definition, transforms)
            store = stores[name]
            left = side_features.get("left")
            right = side_features.get("right")
            all_features = [item for item in side_features.values() if is_finite_feature(item)]
            low = min(all_features, key=lambda item: item.z) if all_features else None
            store.z.append(low.z if low is not None else float("nan"))
            store.left_z.append(left.z if left is not None else float("nan"))
            store.right_z.append(right.z if right is not None else float("nan"))
            store.left_xy.append(left.xy if left is not None else (float("nan"), float("nan")))
            store.right_xy.append(right.xy if right is not None else (float("nan"), float("nan")))
    return stores


def definition_side_features(
    model: Any,
    definition: Mapping[str, Any],
    transforms: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, FrameFeature]:
    kind = definition["kind"]
    out: dict[str, list[FrameFeature]] = {"left": [], "right": []}
    if kind == "body_origin":
        for body_name in definition.get("body_names", ()):
            if body_name in transforms:
                _, pos = transforms[body_name]
                out[side_from_name(body_name)].append(
                    FrameFeature(float(pos[2]), (float(pos[0]), float(pos[1])), body_name)
                )
    elif kind in {"body_geom_points_no_radius", "toe_sole_body_geom_points"}:
        body_by_name = {body.name: body for body in model.bodies}
        for body_name in definition.get("body_names", ()):
            if body_name not in transforms or body_name not in body_by_name:
                continue
            rot, pos = transforms[body_name]
            points = body_by_name[body_name].geom_points or ((0.0, 0.0, 0.0),)
            features = []
            for point_idx, point in enumerate(points):
                world = pos + rot @ np.asarray(point, dtype=np.float64)
                features.append(
                    FrameFeature(
                        float(world[2]),
                        (float(world[0]), float(world[1])),
                        f"{body_name}:point{point_idx}",
                    )
                )
            if features:
                out[side_from_name(body_name)].append(min(features, key=lambda item: item.z))
    elif kind in {"collision_geoms_with_available_extents", "collision_sphere_bottoms"}:
        for geom in definition.get("geoms", ()):
            if geom.body_name not in transforms:
                continue
            feature = geom_low_feature(geom, transforms[geom.body_name])
            if feature is not None:
                out[geom.side].append(feature)
    elif kind == "foot_sites":
        for site in definition.get("sites", ()):
            if site.body_name not in transforms:
                continue
            rot, pos = transforms[site.body_name]
            world = pos + rot @ site.pos
            out[site.side].append(
                FrameFeature(float(world[2]), (float(world[0]), float(world[1])), site.name)
            )
    return {
        side: min(features, key=lambda item: item.z)
        for side, features in out.items()
        if features
    }


def geom_low_feature(
    geom: GeomSpec,
    transform: tuple[np.ndarray, np.ndarray],
) -> FrameFeature | None:
    rot, body_pos = transform
    geom_rot = rot @ quat_wxyz_to_matrix(geom.quat)
    geom_pos = body_pos + rot @ geom.pos
    if geom.geom_type in ("sphere", "") and geom.size:
        radius = float(geom.size[0])
        return FrameFeature(
            float(geom_pos[2] - radius),
            (float(geom_pos[0]), float(geom_pos[1])),
            geom.name,
        )
    points = primitive_points(geom)
    if geom.mesh_vertices.size:
        points = geom.mesh_vertices
    if points.size == 0:
        points = np.zeros((1, 3), dtype=np.float64)
    world = geom_pos[None, :] + points @ geom_rot.T
    idx = int(np.argmin(world[:, 2]))
    low = world[idx]
    return FrameFeature(float(low[2]), (float(low[0]), float(low[1])), geom.name)


def body_transforms(
    model: Any,
    motion: Motion,
    frame_idx: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    joint_values = {
        column[:-4] if column.endswith("_dof") else column: float(value)
        for column, value in zip(G1_JOINT_COLUMNS, motion.dof[frame_idx])
    }
    transforms_list: list[tuple[np.ndarray, np.ndarray]] = []
    transforms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for body in model.bodies:
        local_rot = quat_wxyz_to_matrix(np.asarray(body.quat, dtype=np.float64))
        local_pos = np.asarray(body.pos, dtype=np.float64)
        if body.has_freejoint:
            local_rot = euler_xyz_to_matrix(motion.root_euler[frame_idx])
            local_pos = np.asarray(motion.root_pos[frame_idx], dtype=np.float64)
        if body.joint_name and body.joint_axis is not None:
            local_rot = local_rot @ axis_angle_matrix(
                np.asarray(body.joint_axis, dtype=np.float64),
                joint_values.get(body.joint_name, 0.0),
            )
        if body.parent is None:
            world_rot = local_rot
            world_pos = local_pos
        else:
            parent_rot, parent_pos = transforms_list[body.parent]
            world_rot = parent_rot @ local_rot
            world_pos = parent_pos + parent_rot @ local_pos
        transforms_list.append((world_rot, world_pos))
        transforms[body.name] = (world_rot, world_pos)
    return transforms


def clip_metrics(
    *,
    subset: str,
    row: Mapping[str, str],
    definition: str,
    store: DefinitionStore,
    fps: float,
    ground_height: float,
    contact_threshold: float,
) -> dict[str, Any]:
    z = finite_array(store.z)
    left_z = finite_array(store.left_z)
    right_z = finite_array(store.right_z)
    clearance = z - ground_height
    return {
        "subset": subset,
        "key": row_key(row),
        "split": split_value(row),
        "definition": definition,
        "frames": int(len(store.z)),
        "fps": fps,
        "ground_height_m": ground_height,
        "valid_frame_ratio": float(len(z) / max(1, len(store.z))),
        "contact_ratio": ratio(clearance <= contact_threshold),
        "left_contact_ratio": ratio((left_z - ground_height) <= contact_threshold),
        "right_contact_ratio": ratio((right_z - ground_height) <= contact_threshold),
        "lr_contact_ratio_abs_delta": abs(
            ratio((left_z - ground_height) <= contact_threshold)
            - ratio((right_z - ground_height) <= contact_threshold)
        ),
        "clearance_mean_m": finite_mean(clearance),
        "clearance_p01_m": finite_percentile(clearance, 1),
        "clearance_p05_m": finite_percentile(clearance, 5),
        "clearance_p50_m": finite_percentile(clearance, 50),
        "clearance_p95_m": finite_percentile(clearance, 95),
        "clearance_p99_m": finite_percentile(clearance, 99),
        "below_floor_ratio": ratio(clearance < 0.0),
        "p95_penetration_m": finite_percentile(np.maximum(0.0, -clearance), 95),
        "max_penetration_m": finite_max(np.maximum(0.0, -clearance)),
        "slide_speed_p95_mps": finite_percentile(contact_slide_speeds(store, ground_height, contact_threshold, fps), 95),
        "slide_speed_max_mps": finite_max(contact_slide_speeds(store, ground_height, contact_threshold, fps)),
    }


def aggregate_definition_records(
    records: Sequence[Mapping[str, Any]],
    *,
    ground_height: float,
    contact_threshold: float,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for definition in sorted({str(row["definition"]) for row in records}):
        rows = [row for row in records if row["definition"] == definition]
        frame_count = sum(int(row["frames"]) for row in rows)
        if not rows:
            continue
        out[definition] = {
            "definition": definition,
            "ground_height_m": ground_height,
            "contact_threshold_m": contact_threshold,
            "clip_count": len(rows),
            "frame_count": frame_count,
            "contact_ratio_mean_by_clip": finite_mean([float(row["contact_ratio"]) for row in rows]),
            "contact_ratio_min_clip": finite_min([float(row["contact_ratio"]) for row in rows]),
            "below_floor_ratio_mean_by_clip": finite_mean([float(row["below_floor_ratio"]) for row in rows]),
            "clearance_p50_median_by_clip": finite_percentile([float(row["clearance_p50_m"]) for row in rows], 50),
            "clearance_p95_median_by_clip": finite_percentile([float(row["clearance_p95_m"]) for row in rows], 50),
            "p95_penetration_median_by_clip": finite_percentile([float(row["p95_penetration_m"]) for row in rows], 50),
            "max_penetration_max_clip_m": finite_max([float(row["max_penetration_m"]) for row in rows]),
            "slide_speed_p95_median_by_clip_mps": finite_percentile([float(row["slide_speed_p95_mps"]) for row in rows], 50),
            "slide_speed_max_max_clip_mps": finite_max([float(row["slide_speed_max_mps"]) for row in rows]),
            "worst_contact_key": min(rows, key=lambda row: float(row["contact_ratio"]))["key"],
            "worst_penetration_key": max(rows, key=lambda row: float(row["max_penetration_m"]))["key"],
        }
    return out


def build_floor_candidates(
    series_by_clip: Mapping[str, Mapping[str, DefinitionStore]],
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    contact_threshold: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for definition, meta in definitions.items():
        train_z = []
        smoke_z = []
        for series_key, clip_series in series_by_clip.items():
            store = clip_series.get(definition)
            if store is None:
                continue
            if series_key.startswith("train:"):
                train_z.extend(store.z)
            elif series_key.startswith("smoke:"):
                smoke_z.extend(store.z)
        train_arr = finite_array(train_z)
        floors = {
            "ground_zero": 0.0,
            "train_official_p01_height": finite_percentile(train_arr, 1),
            "train_official_p05_height": finite_percentile(train_arr, 5),
            "train_official_p50_height": finite_percentile(train_arr, 50),
        }
        candidate_rows = []
        for floor_name, floor in floors.items():
            train_clearance = train_arr - floor
            smoke_clearance = finite_array(smoke_z) - floor
            candidate_rows.append(
                {
                    "definition": definition,
                    "definition_status": str(meta.get("status", "")),
                    "floor_name": floor_name,
                    "ground_height_m": floor,
                    "train_frame_count": int(train_arr.size),
                    "train_contact_ratio": ratio(train_clearance <= contact_threshold),
                    "train_below_floor_ratio": ratio(train_clearance < 0.0),
                    "train_p50_clearance_m": finite_percentile(train_clearance, 50),
                    "train_p95_clearance_m": finite_percentile(train_clearance, 95),
                    "train_p95_penetration_m": finite_percentile(np.maximum(0.0, -train_clearance), 95),
                    "train_max_penetration_m": finite_max(np.maximum(0.0, -train_clearance)),
                    "smoke_frame_count": int(finite_array(smoke_z).size),
                    "smoke_contact_ratio": ratio(smoke_clearance <= contact_threshold),
                    "smoke_below_floor_ratio": ratio(smoke_clearance < 0.0),
                    "smoke_p50_clearance_m": finite_percentile(smoke_clearance, 50),
                    "smoke_p95_clearance_m": finite_percentile(smoke_clearance, 95),
                    "smoke_p95_penetration_m": finite_percentile(np.maximum(0.0, -smoke_clearance), 95),
                    "smoke_max_penetration_m": finite_max(np.maximum(0.0, -smoke_clearance)),
                }
            )
        out[definition] = {
            "status": str(meta.get("status", "")),
            "feature_count": int(meta.get("feature_count", 0)),
            "raw_train_height_summary_m": numeric_summary(train_arr),
            "raw_smoke_height_summary_m": numeric_summary(smoke_z),
            "floor_candidates": candidate_rows,
        }
    return out


def select_floor_convention(
    candidates: Mapping[str, Any],
    features: MJCFFeatures,
) -> dict[str, Any]:
    preferred_definition = "foot_collision_sphere_bottom_min_z"
    if candidates.get("foot_collision_geom_min_z", {}).get("status") == "ok":
        preferred_definition = "foot_collision_geom_min_z"
    mesh_blocked = any(
        str(item.get("extent_status", "")).startswith("git_lfs_pointer")
        for item in features.mesh_load_summary.values()
    )
    if mesh_blocked:
        preferred_definition = "foot_collision_sphere_bottom_min_z"
    rows = candidates.get(preferred_definition, {}).get("floor_candidates", [])
    chosen = next((row for row in rows if row["floor_name"] == "train_official_p05_height"), None)
    return {
        "definition": preferred_definition,
        "floor_rule": "train_official_p05_height",
        "ground_height_m": chosen.get("ground_height_m") if chosen else None,
        "contact_threshold_m": CONTACT_THRESHOLD_M,
        "status": "provisional_report_only",
        "reason": (
            "Use explicit collision sphere bottoms because foot mesh STL extents are unavailable as "
            "git-lfs pointer files; set ground to the official train p05 height to bound official "
            "penetration without fitting SOMA/eval transforms."
            if mesh_blocked
            else "Use all available foot collision geom min-z with official train p05 height."
        ),
        "metrics_at_selected_floor": chosen or {},
        "mesh_extent_blocker": mesh_blocked,
    }


def write_overlays(
    overlay_dir: Path,
    series_by_clip: Mapping[str, Mapping[str, DefinitionStore]],
    clip_records: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any],
    *,
    contact_threshold: float,
    random_seed: int,
) -> list[dict[str, str]]:
    chosen_keys = set()
    train_collision = [
        row
        for row in clip_records
        if row["subset"] == "train" and row["definition"] == selected.get("definition")
    ]
    if train_collision:
        chosen_keys.add("train:" + max(train_collision, key=lambda row: float(row["max_penetration_m"]))["key"])
        chosen_keys.add("train:" + min(train_collision, key=lambda row: float(row["contact_ratio"]))["key"])
        rng = random.Random(random_seed)
        for row in rng.sample(train_collision, k=min(2, len(train_collision))):
            chosen_keys.add("train:" + str(row["key"]))
    for series_key in series_by_clip:
        if series_key.startswith("smoke:"):
            chosen_keys.add(series_key)
    overlay_records: list[dict[str, str]] = []
    for series_key in sorted(chosen_keys):
        if series_key not in series_by_clip:
            continue
        path = overlay_dir / f"{safe_name(series_key)}_contact_floor_overlay.svg"
        write_contact_overlay_svg(
            path,
            series_key,
            series_by_clip[series_key],
            selected,
            contact_threshold=contact_threshold,
        )
        overlay_records.append({"clip": series_key, "overlay_svg": str(path)})
    return overlay_records


def write_contact_overlay_svg(
    path: Path,
    title: str,
    series: Mapping[str, DefinitionStore],
    selected: Mapping[str, Any],
    *,
    contact_threshold: float,
) -> None:
    width, height, pad = 920, 420, 42
    names = [
        "ankle_body_origin_z",
        "ankle_body_current_points_min_z",
        "foot_collision_geom_min_z",
        "foot_collision_sphere_bottom_min_z",
    ]
    colors = ["#2354a6", "#d05a2a", "#247a3c", "#7b3fb2"]
    values = []
    for name in names:
        if name in series:
            arr = finite_array(series[name].z)
            if arr.size:
                values.append(arr)
    if not values:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return
    all_values = np.concatenate(values)
    selected_floor = selected.get("ground_height_m")
    floor_lines = [0.0]
    if isinstance(selected_floor, (float, int)):
        floor_lines.append(float(selected_floor))
        floor_lines.append(float(selected_floor) + contact_threshold)
    ymin = float(min(np.min(all_values), min(floor_lines)) - 0.03)
    ymax = float(max(np.max(all_values), max(floor_lines)) + 0.03)
    frames = max(len(series[name].z) for name in series if series[name].z)

    def x_at(idx: int) -> float:
        return pad + idx / max(1, frames - 1) * (width - 2 * pad)

    def y_at(value: float) -> float:
        return height - pad - (value - ymin) / max(1e-9, ymax - ymin) * (height - 2 * pad)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad}" y="24" font-family="monospace" font-size="13">{escape_xml(title)}</text>',
    ]
    for y_value, color, label in [
        (0.0, "#444444", "ground=0"),
        (float(selected_floor or 0.0), "#111111", "selected floor"),
        (float(selected_floor or 0.0) + contact_threshold, "#999999", "selected contact threshold"),
    ]:
        y = y_at(y_value)
        lines.append(f'<line x1="{pad}" y1="{y:.2f}" x2="{width - pad}" y2="{y:.2f}" stroke="{color}" stroke-width="1" stroke-dasharray="5 4"/>')
        lines.append(f'<text x="{width - pad + 4}" y="{y + 4:.2f}" font-family="monospace" font-size="10" fill="{color}">{label}</text>')
    for name, color in zip(names, colors):
        store = series.get(name)
        if store is None:
            continue
        points = [
            f"{x_at(idx):.2f},{y_at(float(value)):.2f}"
            for idx, value in enumerate(store.z)
            if math.isfinite(float(value))
        ]
        if points:
            lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    for idx, (name, color) in enumerate(zip(names, colors)):
        y = height - 18 - idx * 16
        lines.append(f'<rect x="{pad + idx * 210}" y="{y - 9}" width="18" height="3" fill="{color}"/>')
        lines.append(f'<text x="{pad + idx * 210 + 24}" y="{y:.2f}" font-family="monospace" font-size="10">{escape_xml(name)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mjcf_geometry_svg(
    path: Path,
    features: MJCFFeatures,
    foot_bodies: Sequence[str],
) -> None:
    geoms = [geom for geom in features.geoms if geom.body_name in foot_bodies]
    width, height, pad = 760, 360, 38
    points: list[tuple[float, float, str, str]] = []
    for geom in geoms:
        color = "#247a3c" if geom.collision else "#9a9a9a"
        label = geom.name
        if geom.geom_type in ("sphere", "") and geom.size:
            points.append((float(geom.pos[0]), float(geom.pos[2] - geom.size[0]), color, label + " bottom"))
            points.append((float(geom.pos[0]), float(geom.pos[2]), "#2354a6", label + " center"))
        else:
            points.append((float(geom.pos[0]), float(geom.pos[2]), color, label))
    if not points:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return
    xs = [point[0] for point in points]
    zs = [point[1] for point in points]
    xmin, xmax = min(xs) - 0.04, max(xs) + 0.04
    zmin, zmax = min(zs) - 0.02, max(zs) + 0.04

    def px(x: float) -> float:
        return pad + (x - xmin) / max(1e-9, xmax - xmin) * (width - 2 * pad)

    def py(z: float) -> float:
        return height - pad - (z - zmin) / max(1e-9, zmax - zmin) * (height - 2 * pad)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad}" y="24" font-family="monospace" font-size="13">G1 MJCF foot local geometry X-Z: {escape_xml(", ".join(foot_bodies))}</text>',
        f'<line x1="{pad}" y1="{py(0.0):.2f}" x2="{width - pad}" y2="{py(0.0):.2f}" stroke="#444" stroke-dasharray="5 4"/>',
    ]
    for x, z, color, label in points:
        lines.append(f'<circle cx="{px(x):.2f}" cy="{py(z):.2f}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{px(x) + 5:.2f}" y="{py(z) - 5:.2f}" font-family="monospace" font-size="9">{escape_xml(label)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def feature_inventory(
    model: Any,
    features: MJCFFeatures,
    definitions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    toe_sole_body_names = definitions["toe_sole_body_min_z"].get("body_names", ())
    foot_site_names = [site.name for site in definitions["mjcf_foot_site_min_z"].get("sites", ())]
    return {
        "foot_body_names_from_existing_evaluator": list(model.foot_body_names),
        "toe_sole_bodies_present": list(toe_sole_body_names),
        "mjcf_foot_sites_present": foot_site_names,
        "mjcf_site_count_total": len(features.sites),
        "foot_collision_geom_count": int(definitions["foot_collision_geom_min_z"]["feature_count"]),
        "foot_collision_sphere_count": int(definitions["foot_collision_sphere_bottom_min_z"]["feature_count"]),
        "meshdir": features.meshdir,
        "mesh_load_summary": features.mesh_load_summary,
        "definition_status": {
            name: {
                "status": meta.get("status", ""),
                "feature_count": meta.get("feature_count", 0),
                "kind": meta.get("kind", ""),
            }
            for name, meta in definitions.items()
        },
    }


def contact_slide_speeds(
    store: DefinitionStore,
    ground_height: float,
    contact_threshold: float,
    fps: float,
) -> np.ndarray:
    speeds: list[float] = []
    for z_values, xy_values in ((store.left_z, store.left_xy), (store.right_z, store.right_xy)):
        for idx in range(1, len(z_values)):
            prev_z = float(z_values[idx - 1])
            cur_z = float(z_values[idx])
            if not (math.isfinite(prev_z) and math.isfinite(cur_z)):
                continue
            if prev_z - ground_height > contact_threshold or cur_z - ground_height > contact_threshold:
                continue
            prev_xy = np.asarray(xy_values[idx - 1], dtype=np.float64)
            cur_xy = np.asarray(xy_values[idx], dtype=np.float64)
            if np.all(np.isfinite(prev_xy)) and np.all(np.isfinite(cur_xy)):
                speeds.append(float(np.linalg.norm(cur_xy - prev_xy) * fps))
    return np.asarray(speeds, dtype=np.float64)


def primitive_points(geom: GeomSpec) -> np.ndarray:
    size = geom.size
    if geom.geom_type == "box" and len(size) >= 3:
        sx, sy, sz = size[:3]
        return np.asarray(
            [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)],
            dtype=np.float64,
        )
    if geom.geom_type in {"cylinder", "capsule"} and len(size) >= 2:
        radius, half_len = size[:2]
        return np.asarray(
            [
                [0.0, 0.0, -half_len - radius],
                [0.0, 0.0, half_len + radius],
                [radius, 0.0, -half_len],
                [-radius, 0.0, -half_len],
                [0.0, radius, -half_len],
                [0.0, -radius, -half_len],
            ],
            dtype=np.float64,
        )
    return np.empty((0, 3), dtype=np.float64)


def load_stl_vertices(path: Path) -> tuple[np.ndarray, str]:
    if not path:
        return np.empty((0, 3), dtype=np.float64), "mesh_file_not_declared"
    if not path.exists():
        return np.empty((0, 3), dtype=np.float64), "mesh_file_missing"
    data = path.read_bytes()
    if data.startswith(b"version https://git-lfs.github.com/spec"):
        return np.empty((0, 3), dtype=np.float64), "git_lfs_pointer_no_mesh_vertices"
    if data.lstrip().lower().startswith(b"solid"):
        vertices = []
        for line in data.decode("utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.asarray(vertices, dtype=np.float64), "loaded_ascii_stl" if vertices else "ascii_stl_no_vertices"
    if len(data) < 84:
        return np.empty((0, 3), dtype=np.float64), "binary_stl_too_short"
    tri_count = struct.unpack("<I", data[80:84])[0]
    expected = 84 + tri_count * 50
    if expected > len(data):
        return np.empty((0, 3), dtype=np.float64), "binary_stl_truncated"
    vertices = []
    offset = 84
    for _ in range(tri_count):
        offset += 12
        for _vertex in range(3):
            vertices.append(struct.unpack("<fff", data[offset: offset + 12]))
            offset += 12
        offset += 2
    return np.asarray(vertices, dtype=np.float64), "loaded_binary_stl"


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames and not key.startswith("_"):
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def write_readme(path: Path, report: Mapping[str, Any]) -> None:
    selected = report["selected_floor_convention"]
    lines = [
        "# LR-272 contact metric/body/floor audit",
        "",
        "Scope: report-only E audit. No motion transform, candidate implementation, smoke, mixed10, or walk100 run.",
        "",
        f"Train rows: {report['rows']['train_selected']}",
        f"Smoke rows: {report['rows']['smoke_selected']}",
        f"Selected definition: {selected.get('definition')}",
        f"Selected floor rule: {selected.get('floor_rule')}",
        f"Selected ground height: {selected.get('ground_height_m')}",
        "",
        "Primary artifacts:",
        "- contact_metric_body_floor_audit_report.json",
        "- clip_contact_metric_body_floor_audit.csv",
        "- definition_train_summary_ground0.csv",
        "- floor_convention_candidates.csv",
        "- overlays/",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def flatten_floor_candidates(candidates: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for definition, payload in candidates.items():
        for row in payload.get("floor_candidates", []):
            rows.append(dict(row))
    return rows


def parse_vec3(value: str) -> tuple[float, float, float]:
    vals = parse_floats(value)
    padded = (*vals, 0.0, 0.0, 0.0)
    return (padded[0], padded[1], padded[2])


def parse_quat(value: str) -> tuple[float, float, float, float]:
    vals = parse_floats(value)
    padded = (*vals, 1.0, 0.0, 0.0, 0.0)
    return (padded[0], padded[1], padded[2], padded[3])


def parse_floats(value: str) -> tuple[float, ...]:
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


def is_finite_feature(feature: FrameFeature | None) -> bool:
    return feature is not None and math.isfinite(feature.z)


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
    mat_x = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    mat_y = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    mat_z = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return mat_x @ mat_y @ mat_z


def finite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def numeric_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    arr = finite_array(values)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def finite_mean(values: Sequence[float] | np.ndarray) -> float:
    arr = finite_array(values)
    return float(np.mean(arr)) if arr.size else 0.0


def finite_min(values: Sequence[float] | np.ndarray) -> float:
    arr = finite_array(values)
    return float(np.min(arr)) if arr.size else 0.0


def finite_max(values: Sequence[float] | np.ndarray) -> float:
    arr = finite_array(values)
    return float(np.max(arr)) if arr.size else 0.0


def finite_percentile(values: Sequence[float] | np.ndarray, percentile: float) -> float:
    arr = finite_array(values)
    return float(np.percentile(arr, percentile)) if arr.size else 0.0


def ratio(mask: Sequence[bool] | np.ndarray) -> float:
    arr = np.asarray(mask)
    return float(np.mean(arr)) if arr.size else 0.0


def round_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    if isinstance(value, tuple):
        return [round_nested(item) for item in value]
    if isinstance(value, np.ndarray):
        return round_nested(value.tolist())
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def public_evaluator(evaluator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in evaluator.items()
        if key not in {"model", "config"}
    }


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    raise SystemExit(main())
