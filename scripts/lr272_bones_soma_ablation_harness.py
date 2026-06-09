#!/usr/bin/env python3
"""Materialize the LR-272 BONES-SEED SOMA adapter ablation campaign.

This is a campaign harness, not a threshold-tuning script.  It writes a fixed
candidate matrix, per-candidate JSON configs, stage clip manifests, and command
manifests for the remote retarget/metric/visual runners.  The main comparison
source is native BONES-SEED SOMA BVH against the official G1 CSV members from
``g1.tar``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


ISSUE_ID = "996e6d5e-7596-453f-9134-9d5f841a1b52"
BASELINE_COMMIT = "b3ef2708"
DEFAULT_REPO_ROOT = Path("/home/user/project/OnlineRetarget")
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "outputs" / "lr272_bones_soma_ablation_campaign"
DEFAULT_PAIRING_CSV = (
    DEFAULT_REPO_ROOT
    / "outputs"
    / "lr272_bones_seed_pairing_probe_20260608T133932Z"
    / "bones_seed_walk100_pairing.csv"
)
DEFAULT_G1_TAR = Path("/home/user/data/motion_data/g1.tar")
DEFAULT_SOMA_BVH_TAR = Path("/home/user/data/motion_data/back_data/soma_proportional.tar")
DEFAULT_BASELINE_OUTPUT_ROOTS = (
    DEFAULT_REPO_ROOT / "outputs" / "lr272_adapter_probe_offline_pipeline_worst5_20260608T1444Z",
    DEFAULT_REPO_ROOT / "outputs" / "lr272_adapter_convention_diagnostics_20260608T1436Z",
    DEFAULT_REPO_ROOT / "outputs" / "lr272_axis_dof_convention_search_20260608T1455Z",
)
DEFAULT_FRAME_CONSISTENCY_REPORT = (
    DEFAULT_REPO_ROOT
    / "outputs"
    / "lr272_bones_soma_ablation_campaign_20260608T174631Z"
    / "evaluator_frame_consistency_mixed10_20260608T182931Z"
    / "frame_consistency_report.json"
)
DEFAULT_CONTACT_METRIC_BODY_FLOOR_AUDIT_REPORT = (
    DEFAULT_REPO_ROOT
    / "outputs"
    / "lr272_contact_metric_body_floor_audit_20260608T193304Z"
    / "contact_metric_body_floor_audit_report.json"
)
CORRECTED_CONTACT_GROUND_HEIGHT_M = -0.03624434809693291
CORRECTED_CONTACT_THRESHOLD_M = 0.04

DEFAULT_WORST_KEYS = (
    "230413__dance_hiphop_camel_walk_360_R_fast_002__A317",
    "220720__walk_sideway_left_loop_003__A030",
    "230313__walk_hands_on_back_180_loop_R_003__A265",
    "230223__injured_torso_walk_ff_start_225_R_001__A214",
    "231024__walk_the_dog_ff_000_pull_back_leash_R_002__A494",
)
FULL_EVALUATOR_THRESHOLDS = {
    "root_rel_rmse_m": 0.05,
    "root_rot_geodesic_rmse_rad": 0.15,
    "fk_rootrel_mpjpe_m": 0.05,
    "dof_rmse_rad": 0.15,
    "contact_frame_ratio_delta": 0.15,
    "contact_slide_delta_mps": 0.50,
}

KEY_COLUMNS = (
    "lr271_key",
    "motion_key",
    "key",
    "clip_key",
    "action_key",
    "pair_key",
    "bones_rel_key",
    "source_key",
    "bones_key",
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    limit: int
    include_worst_first: bool
    description: str


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    route: str
    enabled: bool
    expected_layer: str
    description: str
    root_world: dict[str, Any]
    summarizer: dict[str, Any]
    dof_convention: dict[str, Any]
    validation: dict[str, Any]
    tags: tuple[str, ...]

    def to_config(self, provenance: Mapping[str, Any], stage_paths: Mapping[str, Path]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "issue_id": ISSUE_ID,
            "candidate": asdict(self),
            "provenance": dict(provenance),
            "stages": {name: str(path) for name, path in stage_paths.items()},
            "main_reference": {
                "dataset": "BONES-SEED native SOMA BVH",
                "target": "official BONES G1 CSV from g1.tar",
                "frame_exact_source": "official CSV pairing/frame/fps/unit checks",
            },
            "excluded_reference": {
                "dataset": "BONES-SONIC 50fps NPZ",
                "reason": "not valid for the frame-exact main comparison unless a validated 50fps resampling adapter is supplied",
            },
            "acceptance": {
                "must_improve_visual": True,
                "must_improve_key_metrics": True,
                "must_explain_layer": (
                    "root-scale, lower-body-contact, IK/DoF convention, or a negative result with a concrete violated convention"
                ),
                "average_only_or_threshold_relaxation": "not accepted",
            },
            "full_evaluator": {
                "required_metrics": [
                    "root_rel_rmse_m",
                    "root_rot_geodesic_rmse_rad",
                    "dof_rmse_rad",
                    "dof_max_abs_rad",
                    "fk_world_mpjpe_m",
                    "fk_rootrel_mpjpe_m",
                    "foot_penetration_depth_pred_m",
                    "contact_frame_ratio_delta",
                    "contact_slide_delta_mps",
                    "contact_lr_asymmetry_delta",
                ],
                "thresholds": FULL_EVALUATOR_THRESHOLDS,
            },
        }


STAGES = (
    StageSpec(
        name="smoke1",
        limit=1,
        include_worst_first=True,
        description="One worst-case clip smoke run before wider spending.",
    ),
    StageSpec(
        name="mixed10",
        limit=10,
        include_worst_first=True,
        description="Ten-clip BONES-SEED mixed run with worst cases first, then deterministic manifest fill.",
    ),
    StageSpec(
        name="walk100",
        limit=100,
        include_worst_first=True,
        description="Existing walk100 paired gate, capped at 100 rows.",
    ),
)


def build_candidates() -> list[Candidate]:
    root_identity = {
        "axis_system": "z_up",
        "root_frame": "world",
        "xy_scale_mode": "identity",
        "xy_scale": 1.0,
        "yaw_alignment": "none",
        "root_translation_unit": "meters",
        "front_axis_policy": "current_soma_retarget",
    }
    summarizer_current = {
        "fps_policy": "preserve_source_bvh_fps",
        "target_fps": "official_g1_csv",
        "source_position_unit": "bvh_cm_to_meters",
        "per_clip_skeleton": False,
        "first_frame_root_init": "current_soma_retarget",
        "pre_roll_frames": 0,
        "stabilization_frames": 0,
        "ramp_policy": "none",
        "raw_action_contract": "current_soma_retarget_action",
    }
    dof_identity = {
        "joint_order": "current_soma_retarget",
        "angle_unit": "radians",
        "sign_overrides": {},
        "axis_swaps": {},
        "single_axis_fk_probe": None,
    }

    candidates: list[Candidate] = [
        Candidate(
            candidate_id="baseline_b3ef2708_soma",
            route="baseline",
            enabled=True,
            expected_layer="current_project_nvidia_soma_lane",
            description="Fixed project/NVIDIA b3ef2708 retarget_source=soma baseline.",
            root_world=root_identity,
            summarizer=summarizer_current,
            dof_convention=dof_identity,
            validation={
                "compare_to": "official_g1_csv",
                "role": "control",
                "run_start_gates": {
                    "frame_consistency_report": {
                        "required": True,
                        "expected_status": "passed",
                        "path": str(DEFAULT_FRAME_CONSISTENCY_REPORT),
                    },
                    "contact_metric_body_floor_audit": {
                        "required": True,
                        "path": str(DEFAULT_CONTACT_METRIC_BODY_FLOOR_AUDIT_REPORT),
                        "expected_definition": "foot_collision_sphere_bottom_min_z",
                        "expected_floor_rule": "train_official_p05_height",
                        "expected_ground_height_m": CORRECTED_CONTACT_GROUND_HEIGHT_M,
                        "expected_contact_threshold_m": CORRECTED_CONTACT_THRESHOLD_M,
                        "expected_collision_sphere_count": 8,
                        "expected_mesh_extent_blocker": True,
                    },
                },
            },
            tags=("baseline", "soma"),
        ),
        Candidate(
            candidate_id="a_root_xy_scale_global_0p90",
            route="A_root_world_adapter",
            enabled=True,
            expected_layer="root_scale",
            description="Global root XY scale down by 10 percent after BVH cm-to-m conversion.",
            root_world={**root_identity, "xy_scale_mode": "global", "xy_scale": 0.90},
            summarizer=summarizer_current,
            dof_convention=dof_identity,
            validation={"compare_to": "official_g1_csv", "metric_focus": ("root_xy_span", "foot_slide")},
            tags=("root", "scale", "global"),
        ),
        Candidate(
            candidate_id="a_root_xy_scale_global_1p10",
            route="A_root_world_adapter",
            enabled=True,
            expected_layer="root_scale",
            description="Global root XY scale up by 10 percent after BVH cm-to-m conversion.",
            root_world={**root_identity, "xy_scale_mode": "global", "xy_scale": 1.10},
            summarizer=summarizer_current,
            dof_convention=dof_identity,
            validation={"compare_to": "official_g1_csv", "metric_focus": ("root_xy_span", "foot_slide")},
            tags=("root", "scale", "global"),
        ),
        Candidate(
            candidate_id="a_root_xy_scale_per_clip_bestfit",
            route="A_root_world_adapter",
            enabled=True,
            expected_layer="root_scale",
            description="Per-clip XY scale estimated against official G1 root trajectory; diagnostic, not a clean seed default.",
            root_world={**root_identity, "xy_scale_mode": "per_clip_bestfit_xy", "xy_scale": "fit_to_official_g1_csv"},
            summarizer=summarizer_current,
            dof_convention=dof_identity,
            validation={"compare_to": "official_g1_csv", "metric_focus": ("root_xy_span",), "diagnostic_only": True},
            tags=("root", "scale", "per_clip", "diagnostic"),
        ),
        Candidate(
            candidate_id="a_root_front_train_split_calibrated",
            route="A_root_world_adapter",
            enabled=False,
            expected_layer="stopped_root_front_negative_mixed10_gate",
            description=(
                "Stopped after negative mixed10 gate: root improved but root-rot/DoF/contact failed and FK root-rel "
                "regressed, so this route is retained only as a diagnostic artifact."
            ),
            root_world={
                **root_identity,
                "xy_scale_mode": "train_split_calibrated",
                "yaw_alignment": "train_split_calibrated",
                "calibration_split": "train",
                "calibration_max_rows": 32,
                "calibration_max_frames": 240,
                "calibration_min_scale": 0.25,
                "calibration_max_scale": 4.0,
                "target_leakage_on_eval": False,
            },
            summarizer=summarizer_current,
            dof_convention=dof_identity,
            validation={
                "compare_to": "official_g1_csv",
                "metric_focus": ("root_xy_span", "root_rot_geodesic", "fk_rootrel_mpjpe", "contact_slide"),
                "calibration_split": "train",
                "eval_contract": "held_out_non_train_rows_only_for_acceptance",
                "target_leakage_on_eval": False,
                "deployable_candidate": False,
                "status": "stopped_negative_mixed10",
                "do_not_run_walk100": True,
            },
            tags=("root", "front", "yaw", "scale", "train_split", "stopped", "diagnostic"),
        ),
        Candidate(
            candidate_id="a_root_yaw_align_first_heading",
            route="A_root_world_adapter",
            enabled=True,
            expected_layer="root_world_frame",
            description="Rotate source root yaw so the first-frame forward heading matches the official G1 convention.",
            root_world={**root_identity, "yaw_alignment": "first_frame_forward_heading"},
            summarizer=summarizer_current,
            dof_convention=dof_identity,
            validation={"compare_to": "official_g1_csv", "metric_focus": ("root_yaw", "mpjpe")},
            tags=("root", "yaw", "front"),
        ),
        Candidate(
            candidate_id="a_root_yaw_align_velocity_heading",
            route="A_root_world_adapter",
            enabled=True,
            expected_layer="root_world_frame",
            description="Rotate source root yaw so early root XY velocity heading matches the official G1 convention.",
            root_world={**root_identity, "yaw_alignment": "early_velocity_heading"},
            summarizer=summarizer_current,
            dof_convention=dof_identity,
            validation={"compare_to": "official_g1_csv", "metric_focus": ("root_yaw", "root_xy_velocity")},
            tags=("root", "yaw", "front"),
        ),
        Candidate(
            candidate_id="a_root_unit_cm_to_m_guard",
            route="A_root_world_adapter",
            enabled=True,
            expected_layer="root_unit",
            description="Strictly assert one BVH cm-to-meter conversion and reject double conversion or meter-to-centimeter expansion.",
            root_world={**root_identity, "root_translation_unit": "assert_single_cm_to_m"},
            summarizer=summarizer_current,
            dof_convention=dof_identity,
            validation={"compare_to": "official_g1_csv", "metric_focus": ("root_height", "root_xy_span")},
            tags=("root", "unit", "guard"),
        ),
        Candidate(
            candidate_id="b_per_clip_skeleton_preroll_ramp",
            route="B_summarizer_preprocess",
            enabled=True,
            expected_layer="summarizer_preprocess",
            description="Use per-clip SOMA skeleton, initialize from clip frame 0, then 10 pre-roll and 5 stabilization ramp frames.",
            root_world=root_identity,
            summarizer={
                **summarizer_current,
                "per_clip_skeleton": True,
                "first_frame_root_init": "source_clip_frame0",
                "pre_roll_frames": 10,
                "stabilization_frames": 5,
                "ramp_policy": "smooth_zero_to_source",
            },
            dof_convention=dof_identity,
            validation={"compare_to": "official_g1_csv", "metric_focus": ("first_2_frames", "mpjpe", "contact")},
            tags=("summarizer", "skeleton", "preroll", "ramp"),
        ),
        Candidate(
            candidate_id="b_source_fps_strict",
            route="B_summarizer_preprocess",
            enabled=True,
            expected_layer="fps_contract",
            description="Preserve BVH source FPS through raw Action and require duration/frame agreement with official G1 CSV.",
            root_world=root_identity,
            summarizer={**summarizer_current, "fps_policy": "strict_source_to_official_duration_match"},
            dof_convention=dof_identity,
            validation={"compare_to": "official_g1_csv", "metric_focus": ("frame_count", "duration_sec")},
            tags=("summarizer", "fps", "contract"),
        ),
        Candidate(
            candidate_id="b_raw_action_contract_guard",
            route="B_summarizer_preprocess",
            enabled=True,
            expected_layer="raw_action_to_retarget_input",
            description="Fail fast when raw Action root/quaternion/DoF order, units, or lengths differ from the retarget input contract.",
            root_world=root_identity,
            summarizer={
                **summarizer_current,
                "raw_action_contract": "root_xyz_m_quat_xyzw_g1_29dof_rad",
                "fail_on_contract_mismatch": True,
            },
            dof_convention=dof_identity,
            validation={"compare_to": "official_g1_csv", "metric_focus": ("dof_order", "unit_consistency")},
            tags=("summarizer", "action", "contract"),
        ),
        Candidate(
            candidate_id="c_hip_pitch_sign_flip_probe",
            route="C_dof_convention",
            enabled=False,
            expected_layer="hip_axis_sign",
            description="Single-axis FK perturbation: flip left/right hip pitch sign and compare lower-body FK/contact response.",
            root_world=root_identity,
            summarizer=summarizer_current,
            dof_convention={
                **dof_identity,
                "sign_overrides": {"left_hip_pitch_joint": -1, "right_hip_pitch_joint": -1},
                "single_axis_fk_probe": {"joints": ("left_hip_pitch_joint", "right_hip_pitch_joint"), "delta_rad": 0.174533},
            },
            validation={"compare_to": "official_g1_csv", "metric_focus": ("hip_fk", "foot_contact", "mpjpe")},
            tags=("dof", "hip", "sign", "fk_probe"),
        ),
        Candidate(
            candidate_id="c_hip_roll_yaw_swap_probe",
            route="C_dof_convention",
            enabled=False,
            expected_layer="hip_neighboring_axis_order",
            description="Single-axis FK perturbation: swap hip roll/yaw neighboring axes to test lower-body axis ordering.",
            root_world=root_identity,
            summarizer=summarizer_current,
            dof_convention={
                **dof_identity,
                "axis_swaps": {
                    "left_hip_roll_joint": "left_hip_yaw_joint",
                    "right_hip_roll_joint": "right_hip_yaw_joint",
                },
                "single_axis_fk_probe": {"joints": ("left_hip_roll_joint", "left_hip_yaw_joint"), "delta_rad": 0.174533},
            },
            validation={"compare_to": "official_g1_csv", "metric_focus": ("hip_fk", "knee_fk", "foot_contact")},
            tags=("dof", "hip", "axis_swap", "fk_probe"),
        ),
        Candidate(
            candidate_id="c_waist_yaw_sign_flip_probe",
            route="C_dof_convention",
            enabled=False,
            expected_layer="waist_axis_sign",
            description="Single-axis FK perturbation: flip waist yaw sign and compare torso/root heading response.",
            root_world=root_identity,
            summarizer=summarizer_current,
            dof_convention={
                **dof_identity,
                "sign_overrides": {"waist_yaw_joint": -1},
                "single_axis_fk_probe": {"joints": ("waist_yaw_joint",), "delta_rad": 0.174533},
            },
            validation={"compare_to": "official_g1_csv", "metric_focus": ("waist_fk", "root_yaw", "upper_body_mpjpe")},
            tags=("dof", "waist", "sign", "fk_probe"),
        ),
        Candidate(
            candidate_id="c_shoulder_roll_sign_flip_probe",
            route="C_dof_convention",
            enabled=False,
            expected_layer="shoulder_axis_sign",
            description="Single-axis FK perturbation: flip shoulder roll sign and inspect arm FK response.",
            root_world=root_identity,
            summarizer=summarizer_current,
            dof_convention={
                **dof_identity,
                "sign_overrides": {"left_shoulder_roll_joint": -1, "right_shoulder_roll_joint": -1},
                "single_axis_fk_probe": {
                    "joints": ("left_shoulder_roll_joint", "right_shoulder_roll_joint"),
                    "delta_rad": 0.174533,
                },
            },
            validation={"compare_to": "official_g1_csv", "metric_focus": ("shoulder_fk", "elbow_fk", "upper_body_mpjpe")},
            tags=("dof", "shoulder", "sign", "fk_probe"),
        ),
        Candidate(
            candidate_id="c_elbow_sign_flip_probe",
            route="C_dof_convention",
            enabled=False,
            expected_layer="elbow_axis_sign",
            description="Single-axis FK perturbation: flip elbow sign and inspect arm bend direction.",
            root_world=root_identity,
            summarizer=summarizer_current,
            dof_convention={
                **dof_identity,
                "sign_overrides": {"left_elbow_joint": -1, "right_elbow_joint": -1},
                "single_axis_fk_probe": {"joints": ("left_elbow_joint", "right_elbow_joint"), "delta_rad": 0.174533},
            },
            validation={"compare_to": "official_g1_csv", "metric_focus": ("elbow_fk", "wrist_fk", "upper_body_mpjpe")},
            tags=("dof", "elbow", "sign", "fk_probe"),
        ),
        Candidate(
            candidate_id="c_lower_body_fk_signature_dof_map_train_split_v1",
            route="C_dof_convention",
            enabled=True,
            expected_layer="lower_body_dof_sign_order_neighbor_axis",
            description=(
                "Train-split-only global lower-body DoF sign/order/neighbor-axis map learned from root-aligned "
                "single-axis FK signatures; root/world and summarizer stay at the baseline contract."
            ),
            root_world=root_identity,
            summarizer=summarizer_current,
            dof_convention={
                **dof_identity,
                "train_split_fk_signature_map": {
                    "enabled": True,
                    "version": "v1",
                    "required": True,
                    "calibration_split": "train",
                    "target_leakage_on_eval": False,
                    "lower_body_groups": (
                        "left_hip",
                        "right_hip",
                        "left_knee",
                        "right_knee",
                        "left_ankle",
                        "right_ankle",
                    ),
                    "allow_waist": False,
                    "allow_shoulder_elbow": False,
                    "single_axis_delta_rad": 0.174533,
                    "calibration_max_rows": 16,
                    "calibration_max_frames": 160,
                    "min_relative_improvement": 0.0,
                },
            },
            validation={
                "compare_to": "official_g1_csv",
                "metric_focus": ("lower_body_fk", "dof_rmse", "contact_slide"),
                "calibration_split": "train",
                "eval_contract": "held_out_non_train_rows_only_for_acceptance",
                "target_leakage_on_eval": False,
                "allowed_stages": ("smoke1",),
                "run_start_gates": {
                    "frame_consistency_report": {
                        "required": True,
                        "expected_status": "passed",
                        "path": str(DEFAULT_FRAME_CONSISTENCY_REPORT),
                    }
                },
                "root_front_route_status": "stopped",
                "do_not_run_mixed10_until_smoke_no_regression": True,
                "do_not_run_walk100": True,
            },
            tags=("dof", "lower_body", "train_split", "fk_signature", "smoke_only"),
        ),
    ]
    return candidates


def build_campaign(
    *,
    pairing_csv: Path,
    output_dir: Path,
    repo_root: Path,
    g1_tar: Path,
    soma_bvh_tar: Path,
    baseline_commit: str,
    run_name: str,
    worst_keys: Sequence[str],
    retarget_template: str | None = None,
    metric_template: str | None = None,
    visual_template: str | None = None,
    runner_python: str | None = None,
    runner_script: Path | None = None,
    allow_missing_inputs: bool = False,
) -> dict[str, Any]:
    rows = read_pairing_rows(pairing_csv, allow_missing=allow_missing_inputs)
    candidates = build_candidates()
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_paths = write_stage_manifests(output_dir / "stages", rows, worst_keys)
    provenance = build_provenance(
        repo_root=repo_root,
        pairing_csv=pairing_csv,
        g1_tar=g1_tar,
        soma_bvh_tar=soma_bvh_tar,
        baseline_commit=baseline_commit,
        run_name=run_name,
    )
    config_paths = write_candidate_configs(output_dir / "configs", candidates, provenance, stage_paths)
    candidate_matrix = output_dir / "candidate_matrix.csv"
    write_candidate_matrix(candidate_matrix, candidates)
    command_rows = build_command_rows(
        candidates=candidates,
        stages=STAGES,
        stage_paths=stage_paths,
        config_paths=config_paths,
        output_dir=output_dir,
        repo_root=repo_root,
        baseline_commit=baseline_commit,
        retarget_template=retarget_template,
        metric_template=metric_template,
        visual_template=visual_template,
        runner_python=runner_python or sys.executable,
        runner_script=runner_script or Path("scripts/lr272_bones_soma_candidate_runner.py"),
    )
    commands_jsonl = output_dir / "commands.jsonl"
    commands_sh = output_dir / "commands.sh"
    write_commands_jsonl(commands_jsonl, command_rows)
    write_commands_sh(commands_sh, command_rows)

    manifest = {
        "schema_version": 1,
        "issue_id": ISSUE_ID,
        "run_name": run_name,
        "output_dir": str(output_dir),
        "provenance": provenance,
        "candidate_count": len(candidates),
        "enabled_candidate_count": sum(1 for item in candidates if item.enabled),
        "stage_count": len(STAGES),
        "pairing_row_count": len(rows),
        "candidate_matrix": str(candidate_matrix),
        "commands_jsonl": str(commands_jsonl),
        "commands_sh": str(commands_sh),
        "configs_dir": str(output_dir / "configs"),
        "stages_dir": str(output_dir / "stages"),
        "stage_rows": {stage.name: count_csv_rows(path) for stage, path in stage_paths.items()},
        "notes": [
            "Run smoke1 first, then mixed10. Expand only top 1-2 candidates to walk100.",
            "Do not use BONES-SONIC 50fps NPZ for the frame-exact main comparison.",
            "A candidate passes only if visual evidence and key metrics improve together.",
            "a_root_front_train_split_calibrated is stopped after the negative mixed10 gate and is disabled in new commands.",
            "c_lower_body_fk_signature_dof_map_train_split_v1 is smoke1-only until no frame/count/unit or root regression is observed.",
        ],
    }
    manifest_path = output_dir / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_summary_md(output_dir / "README.md", manifest, candidates, command_rows)
    return manifest


def read_pairing_rows(path: Path, *, allow_missing: bool = False) -> list[dict[str, str]]:
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"pairing CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows and not allow_missing:
        raise ValueError(f"pairing CSV has no rows: {path}")
    return rows


def write_stage_manifests(
    output_dir: Path,
    rows: Sequence[Mapping[str, str]],
    worst_keys: Sequence[str],
) -> dict[StageSpec, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[StageSpec, Path] = {}
    for stage in STAGES:
        selected = select_stage_rows(rows, stage, worst_keys)
        path = output_dir / f"{stage.name}.csv"
        write_csv_rows(path, selected)
        result[stage] = path
    return result


def select_stage_rows(
    rows: Sequence[Mapping[str, str]],
    stage: StageSpec,
    worst_keys: Sequence[str],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    selected_keys: set[str] = set()
    if stage.include_worst_first:
        wanted = [key for key in worst_keys if key]
        for key in wanted:
            row = find_row_by_key(rows, key)
            if row is not None and row_key(row) not in selected_keys:
                selected.append(dict(row))
                selected_keys.add(row_key(row))
            if len(selected) >= stage.limit:
                return selected

    for row in rows:
        key = row_key(row)
        if key in selected_keys:
            continue
        selected.append(dict(row))
        selected_keys.add(key)
        if len(selected) >= stage.limit:
            break
    return selected


def find_row_by_key(rows: Sequence[Mapping[str, str]], key: str) -> Mapping[str, str] | None:
    for row in rows:
        if row_key(row) == key:
            return row
    for row in rows:
        if key in row.values():
            return row
    return None


def row_key(row: Mapping[str, str]) -> str:
    for column in KEY_COLUMNS:
        value = str(row.get(column, "")).strip()
        if value:
            return value
    for column in (
        "move_g1_path",
        "official_bones_g1_csv_member",
        "source_bvh",
        "move_soma_proportional_path",
        "manifest_robot_motion_pkl",
    ):
        value = str(row.get(column, "")).strip()
        if value:
            return Path(value).stem
    return json.dumps(dict(row), sort_keys=True)


def write_candidate_configs(
    output_dir: Path,
    candidates: Sequence[Candidate],
    provenance: Mapping[str, Any],
    stage_paths: Mapping[StageSpec, Path],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_path_by_name = {stage.name: path for stage, path in stage_paths.items()}
    result: dict[str, Path] = {}
    for candidate in candidates:
        path = output_dir / f"{candidate.candidate_id}.json"
        payload = candidate.to_config(provenance, stage_path_by_name)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result[candidate.candidate_id] = path
    return result


def write_candidate_matrix(path: Path, candidates: Sequence[Candidate]) -> None:
    fieldnames = (
        "candidate_id",
        "route",
        "enabled",
        "expected_layer",
        "description",
        "tags",
        "root_world",
        "summarizer",
        "dof_convention",
        "validation",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "candidate_id": candidate.candidate_id,
                    "route": candidate.route,
                    "enabled": int(candidate.enabled),
                    "expected_layer": candidate.expected_layer,
                    "description": candidate.description,
                    "tags": ",".join(candidate.tags),
                    "root_world": json.dumps(candidate.root_world, sort_keys=True),
                    "summarizer": json.dumps(candidate.summarizer, sort_keys=True),
                    "dof_convention": json.dumps(candidate.dof_convention, sort_keys=True),
                    "validation": json.dumps(candidate.validation, sort_keys=True),
                }
            )


def build_command_rows(
    *,
    candidates: Sequence[Candidate],
    stages: Sequence[StageSpec],
    stage_paths: Mapping[StageSpec, Path],
    config_paths: Mapping[str, Path],
    output_dir: Path,
    repo_root: Path,
    baseline_commit: str,
    retarget_template: str | None,
    metric_template: str | None,
    visual_template: str | None,
    runner_python: str,
    runner_script: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in candidates:
        if not candidate.enabled:
            continue
        allowed_stages = candidate.validation.get("allowed_stages")
        allowed_stage_names = {str(stage_name) for stage_name in allowed_stages or ()}
        for stage in stages:
            if allowed_stage_names and stage.name not in allowed_stage_names:
                continue
            run_dir = output_dir / "runs" / stage.name / candidate.candidate_id
            runner_path = runner_script
            if not runner_path.is_absolute():
                runner_path = repo_root / runner_path
            values = {
                "candidate_id": candidate.candidate_id,
                "route": candidate.route,
                "stage": stage.name,
                "config": str(config_paths[candidate.candidate_id]),
                "stage_csv": str(stage_paths[stage]),
                "output_dir": str(run_dir),
                "repo_root": str(repo_root),
                "baseline_commit": baseline_commit,
                "runner_python": runner_python,
                "runner_script": str(runner_path),
            }
            default_retarget = default_runner_command(values, mode="retarget")
            default_metric = default_runner_command(values, mode="metric")
            default_visual = default_runner_command(values, mode="visual")
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "route": candidate.route,
                    "stage": stage.name,
                    "stage_csv": str(stage_paths[stage]),
                    "config": str(config_paths[candidate.candidate_id]),
                    "output_dir": str(run_dir),
                    "retarget_command": format_template(retarget_template, values) if retarget_template else default_retarget,
                    "metric_command": format_template(metric_template, values) if metric_template else default_metric,
                    "visual_command": format_template(visual_template, values) if visual_template else default_visual,
                    "baseline_commit": baseline_commit,
                }
            )
    return rows


def default_runner_command(values: Mapping[str, str], *, mode: str) -> str:
    command = (
        f"{shlex.quote(values['runner_python'])} {shlex.quote(values['runner_script'])} "
        f"--config {shlex.quote(values['config'])} "
        f"--stage-csv {shlex.quote(values['stage_csv'])} "
        f"--output-dir {shlex.quote(values['output_dir'])} "
        f"--mode {shlex.quote(mode)}"
    )
    if mode == "visual":
        command += " --render-isaac"
    return command


def format_template(template: str, values: Mapping[str, str]) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        raise KeyError(f"unknown template placeholder {exc.args[0]!r}; available keys: {sorted(values)}") from exc


def write_commands_jsonl(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def write_commands_sh(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by scripts/lr272_bones_soma_ablation_harness.py",
        "# Run smoke1 first, then mixed10. Expand only top 1-2 candidates to walk100.",
        "",
    ]
    for row in rows:
        lines.append(f"# {row['stage']} {row['candidate_id']} ({row['route']})")
        lines.append(str(row["retarget_command"]))
        if row.get("metric_command"):
            lines.append(str(row["metric_command"]))
        if row.get("visual_command"):
            lines.append(str(row["visual_command"]))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def write_summary_md(
    path: Path,
    manifest: Mapping[str, Any],
    candidates: Sequence[Candidate],
    command_rows: Sequence[Mapping[str, str]],
) -> None:
    route_counts: dict[str, int] = {}
    for candidate in candidates:
        route_counts[candidate.route] = route_counts.get(candidate.route, 0) + 1
    lines = [
        "# LR-272 BONES-SEED SOMA Ablation Campaign",
        "",
        f"- Issue: `{ISSUE_ID}`",
        f"- Run: `{manifest['run_name']}`",
        f"- Candidates: `{manifest['candidate_count']}`",
        f"- Commands: `{len(command_rows)}`",
        f"- Baseline commit: `{manifest['provenance']['baseline']['commit_expected']}`",
        "",
        "## Routes",
        "",
    ]
    for route, count in sorted(route_counts.items()):
        lines.append(f"- `{route}`: {count}")
    lines.extend(
        [
            "",
            "## Required Order",
            "",
            "1. Run `smoke1` for all enabled candidates.",
            "2. Run `mixed10` only after smoke artifacts are present.",
            "3. Expand only the strongest 1-2 candidates to `walk100`.",
            "",
            "## Acceptance",
            "",
            "- Visual evidence and key metrics must improve together.",
            "- The passing route must explain root-scale, lower-body-contact, or IK/DoF convention.",
            "- Wrapper-level averages or threshold relaxation are not sufficient.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_provenance(
    *,
    repo_root: Path,
    pairing_csv: Path,
    g1_tar: Path,
    soma_bvh_tar: Path,
    baseline_commit: str,
    run_name: str,
) -> dict[str, Any]:
    harness_root = Path(__file__).resolve().parents[1]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "harness": {
            "script": str(Path(__file__).resolve()),
            "repo_root": str(harness_root),
            "commit_actual": git_rev_parse(harness_root),
            "dirty_status": git_status_short(harness_root),
        },
        "baseline": {
            "repo_root": str(repo_root),
            "commit_expected": baseline_commit,
            "retarget_source": "soma",
            "commit_actual": git_rev_parse(repo_root),
        },
        "inputs": {
            "pairing_csv": str(pairing_csv),
            "g1_tar": str(g1_tar),
            "soma_bvh_tar": str(soma_bvh_tar),
            "baseline_output_roots": [str(path) for path in DEFAULT_BASELINE_OUTPUT_ROOTS],
            "frame_consistency_report_json": str(DEFAULT_FRAME_CONSISTENCY_REPORT),
            "contact_metric_body_floor_audit_report_json": str(DEFAULT_CONTACT_METRIC_BODY_FLOOR_AUDIT_REPORT),
        },
    }


def git_rev_parse(repo_root: Path) -> str:
    if not repo_root.exists():
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def git_status_short(repo_root: Path) -> str:
    if not repo_root.exists():
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames_for_rows(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fieldnames_for_rows(rows: Sequence[Mapping[str, str]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names or ["motion_key"]


def count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def parse_worst_keys(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def print_run(config: Path, stage: str, output_dir: Path) -> dict[str, Any]:
    payload = json.loads(config.read_text(encoding="utf-8"))
    stages = payload.get("stages", {})
    stage_csv = stages.get(stage)
    if not stage_csv:
        raise ValueError(f"stage {stage!r} is not present in config {config}")
    result = {
        "candidate_id": payload["candidate"]["candidate_id"],
        "route": payload["candidate"]["route"],
        "stage": stage,
        "stage_csv": stage_csv,
        "output_dir": str(output_dir),
        "config": str(config),
        "runner_note": "Replace this dry-run command with the remote retarget runner template for execution.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Write campaign configs, stage manifests, and command manifests.")
    plan.add_argument("--pairing-csv", type=Path, default=DEFAULT_PAIRING_CSV)
    plan.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    plan.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    plan.add_argument("--g1-tar", type=Path, default=DEFAULT_G1_TAR)
    plan.add_argument("--soma-bvh-tar", type=Path, default=DEFAULT_SOMA_BVH_TAR)
    plan.add_argument("--baseline-commit", default=BASELINE_COMMIT)
    plan.add_argument("--run-name", default="lr272_bones_soma_ablation_campaign")
    plan.add_argument("--worst-keys", default=",".join(DEFAULT_WORST_KEYS))
    plan.add_argument(
        "--retarget-template",
        default=None,
        help=(
            "Optional format string for remote retarget execution. Available placeholders: "
            "{candidate_id}, {route}, {stage}, {config}, {stage_csv}, {output_dir}, "
            "{repo_root}, {baseline_commit}, {runner_python}, {runner_script}."
        ),
    )
    plan.add_argument("--metric-template", default=None, help="Optional metric command template with the same placeholders.")
    plan.add_argument("--visual-template", default=None, help="Optional visual command template with the same placeholders.")
    plan.add_argument(
        "--runner-python",
        default=sys.executable,
        help="Python interpreter used in generated runner commands.",
    )
    plan.add_argument(
        "--runner-script",
        type=Path,
        default=Path("scripts/lr272_bones_soma_candidate_runner.py"),
        help="Runner script used in generated commands; relative paths resolve under --repo-root.",
    )
    plan.add_argument("--allow-missing-inputs", action="store_true")

    print_run_parser = subparsers.add_parser("print-run", help="Print one resolved candidate/stage run spec.")
    print_run_parser.add_argument("--config", type=Path, required=True)
    print_run_parser.add_argument("--stage", required=True)
    print_run_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        manifest = build_campaign(
            pairing_csv=args.pairing_csv,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            g1_tar=args.g1_tar,
            soma_bvh_tar=args.soma_bvh_tar,
            baseline_commit=args.baseline_commit,
            run_name=args.run_name,
            worst_keys=parse_worst_keys(args.worst_keys),
            retarget_template=args.retarget_template,
            metric_template=args.metric_template,
            visual_template=args.visual_template,
            runner_python=args.runner_python,
            runner_script=args.runner_script,
            allow_missing_inputs=args.allow_missing_inputs,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "print-run":
        print_run(args.config, args.stage, args.output_dir)
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
