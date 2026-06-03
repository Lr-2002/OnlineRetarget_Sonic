"""LR-239 Isaac/SRC replay exporter contract and preflight helpers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence


SCHEMA_VERSION = "lr239.isaac_src_contact_packets.v1"
DEFAULT_G1_USD = (
    "/mnt/data_cpfs/code/wxh/OnlineRetarget/runs/isaaclab_urdf_cache/"
    "g1_main/main.usd"
)
DEFAULT_5090_REPO = "/mnt/data_cpfs/code/wxh/OnlineRetarget"
DEFAULT_ISAAC_PYTHON = "/workspace/isaaclab/_isaac_sim/python.sh"
BODY_PAIR_CONTACT_BLOCKED_REASON = (
    "verified body-body/self-collision contact source is not bound; "
    "foot-ground sensors are support-only"
)
SRC_BLOCKED_REASON = "SRC geometry checker is not bound for this exporter"
FOOT_GROUND_FILTER_BLOCKED_REASON = (
    "filtered foot-ground force_matrix_w is unavailable; aggregate net_forces_w "
    "is not ground-contact evidence"
)

SONIC_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

DEFAULT_FOOT_LINKS: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
DEFAULT_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

REQUIRED_STATE_DATASETS: Mapping[str, tuple[str, ...]] = {
    "pred root_pos_world_m": ("pred_g1_state/root_pos_world_m", "pred/root_pos_world_m"),
    "target root_pos_world_m": (
        "target_g1_state/root_pos_world_m",
        "target/root_pos_world_m",
    ),
    "pred root_quat_wxyz": (
        "pred_g1_state/root_quat_wxyz",
        "pred/root_quat_wxyz",
    ),
    "target root_quat_wxyz": (
        "target_g1_state/root_quat_wxyz",
        "target/root_quat_wxyz",
    ),
    "pred joint_q_rad": ("pred_g1_state/joint_q_rad", "pred/joint_q_rad"),
    "target joint_q_rad": ("target_g1_state/joint_q_rad", "target/joint_q_rad"),
}


@dataclass(frozen=True)
class SupportContract:
    ground_prim_path: str = "/World/Ground"
    ground_height_m: float = 0.0
    up_axis: str = "z"
    support_force_threshold_n: float = 20.0
    floating_clearance_threshold_m: float = 0.04
    support_margin_m: float = 0.05


@dataclass(frozen=True)
class CrossRatioContract:
    algorithm: str = "src_mesh_cross_ratio_v1"
    blocked_threshold: float = 0.0
    ratio_field: str = "self_intersection_ratio"
    frame_flag_field: str = "self_intersection_frames"
    required_backend: str = "SRC geometry checker or equivalent Isaac mesh intersection checker"


@dataclass(frozen=True)
class ContactContract:
    foot_links: tuple[str, ...] = DEFAULT_FOOT_LINKS
    disabled_collision_pairs: tuple[tuple[str, str], ...] = ()
    contact_filter_prim_paths: tuple[str, ...] = ("/World/Ground",)
    enable_self_collisions: bool = True
    activate_contact_sensors: bool = True
    contact_force_threshold_n: float = 1.0
    foot_slide_speed_threshold_mps: float = 0.25
    foot_skate_distance_threshold_m: float = 0.02
    support: SupportContract = field(default_factory=SupportContract)
    cross_ratio: CrossRatioContract = field(default_factory=CrossRatioContract)


@dataclass(frozen=True)
class ReplayConfig:
    schema_version: str = SCHEMA_VERSION
    root_prim_path: str = "/World/Robot"
    robot_usd: str = DEFAULT_G1_USD
    paired_state_h5: str = ""
    variant: str = ""
    fps: float = 50.0
    root_rot_format: str = "wxyz"
    joint_names: tuple[str, ...] = SONIC_JOINT_NAMES
    body_names: tuple[str, ...] = DEFAULT_BODY_NAMES
    contact: ContactContract = field(default_factory=ContactContract)


@dataclass(frozen=True)
class ArtifactSummary:
    path: str
    exists: bool
    bytes: int
    sha256: str
    hdf5_status: str
    frame_count: int | None = None
    joint_count: int | None = None
    fps: float | None = None
    joint_names: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    missing_required_datasets: tuple[str, ...] = ()
    state_dataset_paths: tuple[tuple[str, str], ...] = ()
    state_dataset_shapes: tuple[tuple[str, tuple[int, ...]], ...] = ()
    shape_errors: tuple[str, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class PairedStateData:
    frame_count: int
    fps: float
    joint_names: tuple[str, ...]
    pred_root_pos_world_m: Any
    target_root_pos_world_m: Any
    pred_root_quat_wxyz: Any
    target_root_quat_wxyz: Any
    pred_joint_q_rad: Any
    target_joint_q_rad: Any


@dataclass(frozen=True)
class ReplayStateFrame:
    frame_idx: int
    label: str
    root_pos_world_m: Sequence[float]
    root_quat_wxyz: Sequence[float]
    joint_q_rad: Sequence[float]


@dataclass(frozen=True)
class FilteredContactForce:
    force_n: float | None
    status: str
    reason: str = ""


@dataclass(frozen=True)
class FootGroundContactState:
    forces_n: tuple[float | None, ...]
    flags: tuple[bool, ...]
    status: str
    reason: str = ""


@dataclass(frozen=True)
class FootArtifactState:
    foot_slide_speed_mps: tuple[float | None, ...]
    foot_slide_flags: tuple[bool | None, ...]
    foot_skate_distance_m: tuple[float | None, ...]
    foot_skate_flags: tuple[bool | None, ...]
    foot_float_clearance_m: tuple[float | None, ...]
    foot_float_flags: tuple[bool | None, ...]
    foot_artifact_status: str
    foot_artifact_reason: str = ""


class ReplayBackend(Protocol):
    def __enter__(self) -> "ReplayBackend":
        ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        ...

    def collect_state_packet(self, state: ReplayStateFrame) -> dict[str, Any]:
        ...

    def report(self) -> dict[str, Any]:
        ...


def default_replay_config(
    *,
    paired_state_h5: Path | None = None,
    robot_usd: Path | None = None,
) -> ReplayConfig:
    """Return the controlling LR-239 Isaac/SRC packet contract."""

    return ReplayConfig(
        paired_state_h5=str(paired_state_h5) if paired_state_h5 is not None else "",
        robot_usd=str(robot_usd) if robot_usd is not None else DEFAULT_G1_USD,
    )


def load_replay_config(path: Path | None) -> ReplayConfig:
    if path is None:
        return default_replay_config()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must contain a JSON object: {path}")
    return replay_config_from_mapping(payload)


def replay_config_from_mapping(payload: Mapping[str, Any]) -> ReplayConfig:
    contact_payload = _mapping(payload.get("contact", {}), "contact")
    support_payload = _mapping(contact_payload.get("support", {}), "contact.support")
    cross_payload = _mapping(contact_payload.get("cross_ratio", {}), "contact.cross_ratio")
    support = SupportContract(
        ground_prim_path=str(
            support_payload.get("ground_prim_path", SupportContract.ground_prim_path)
        ),
        ground_height_m=float(
            support_payload.get("ground_height_m", SupportContract.ground_height_m)
        ),
        up_axis=str(support_payload.get("up_axis", SupportContract.up_axis)),
        support_force_threshold_n=float(
            support_payload.get(
                "support_force_threshold_n",
                SupportContract.support_force_threshold_n,
            )
        ),
        floating_clearance_threshold_m=float(
            support_payload.get(
                "floating_clearance_threshold_m",
                SupportContract.floating_clearance_threshold_m,
            )
        ),
        support_margin_m=float(
            support_payload.get("support_margin_m", SupportContract.support_margin_m)
        ),
    )
    cross_ratio = CrossRatioContract(
        algorithm=str(cross_payload.get("algorithm", CrossRatioContract.algorithm)),
        blocked_threshold=float(
            cross_payload.get("blocked_threshold", CrossRatioContract.blocked_threshold)
        ),
        ratio_field=str(cross_payload.get("ratio_field", CrossRatioContract.ratio_field)),
        frame_flag_field=str(
            cross_payload.get("frame_flag_field", CrossRatioContract.frame_flag_field)
        ),
        required_backend=str(
            cross_payload.get("required_backend", CrossRatioContract.required_backend)
        ),
    )
    contact = ContactContract(
        foot_links=tuple(
            str(value) for value in contact_payload.get("foot_links", DEFAULT_FOOT_LINKS)
        ),
        disabled_collision_pairs=tuple(
            _pair(value, "contact.disabled_collision_pairs")
            for value in contact_payload.get("disabled_collision_pairs", ())
        ),
        contact_filter_prim_paths=tuple(
            str(value)
            for value in contact_payload.get("contact_filter_prim_paths", ("/World/Ground",))
        ),
        enable_self_collisions=bool(contact_payload.get("enable_self_collisions", True)),
        activate_contact_sensors=bool(contact_payload.get("activate_contact_sensors", True)),
        contact_force_threshold_n=float(contact_payload.get("contact_force_threshold_n", 1.0)),
        foot_slide_speed_threshold_mps=float(
            contact_payload.get(
                "foot_slide_speed_threshold_mps",
                ContactContract.foot_slide_speed_threshold_mps,
            )
        ),
        foot_skate_distance_threshold_m=float(
            contact_payload.get(
                "foot_skate_distance_threshold_m",
                ContactContract.foot_skate_distance_threshold_m,
            )
        ),
        support=support,
        cross_ratio=cross_ratio,
    )
    return ReplayConfig(
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        root_prim_path=str(payload.get("root_prim_path", ReplayConfig.root_prim_path)),
        robot_usd=str(payload.get("robot_usd", ReplayConfig.robot_usd)),
        paired_state_h5=str(payload.get("paired_state_h5", "")),
        variant=str(payload.get("variant", "")),
        fps=float(payload.get("fps", 50.0)),
        root_rot_format=str(payload.get("root_rot_format", "wxyz")),
        joint_names=tuple(str(value) for value in payload.get("joint_names", SONIC_JOINT_NAMES)),
        body_names=tuple(str(value) for value in payload.get("body_names", DEFAULT_BODY_NAMES)),
        contact=contact,
    )


def validate_replay_config(config: ReplayConfig) -> list[str]:
    errors: list[str] = []
    if config.schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not config.paired_state_h5:
        errors.append("paired_state_h5 is required")
    if config.fps <= 0:
        errors.append("fps must be positive")
    if config.root_rot_format != "wxyz":
        errors.append("root_rot_format must be wxyz for IsaacLab write_root_state_to_sim")
    if len(config.joint_names) != 29:
        errors.append(f"joint_names must contain 29 G1 joints, got {len(config.joint_names)}")
    if set(config.contact.foot_links) - set(config.body_names):
        errors.append("body_names must include the configured foot links")
    if set(config.contact.foot_links) - set(_canonical_link_names(config)):
        errors.append("contact.foot_links must be named G1 links in the canonical model contract")
    if not config.contact.foot_links:
        errors.append("contact.foot_links must not be empty")
    if not config.contact.contact_filter_prim_paths:
        errors.append("contact.contact_filter_prim_paths must include ground/filter prims")
    elif config.contact.support.ground_prim_path not in config.contact.contact_filter_prim_paths:
        errors.append(
            "contact.contact_filter_prim_paths must include contact.support.ground_prim_path"
        )
    if not config.contact.enable_self_collisions:
        errors.append(
            "contact.enable_self_collisions must be true for body-body contact readiness"
        )
    if not config.contact.activate_contact_sensors:
        errors.append("contact.activate_contact_sensors must be true for PhysX contact packets")
    if config.contact.contact_force_threshold_n < 0:
        errors.append("contact.contact_force_threshold_n must be non-negative")
    if config.contact.foot_slide_speed_threshold_mps < 0:
        errors.append("contact.foot_slide_speed_threshold_mps must be non-negative")
    if config.contact.foot_skate_distance_threshold_m < 0:
        errors.append("contact.foot_skate_distance_threshold_m must be non-negative")
    if config.contact.support.up_axis.lower() not in {"x", "y", "z"}:
        errors.append("contact.support.up_axis must be x, y, or z")
    if config.contact.support.floating_clearance_threshold_m < 0:
        errors.append("floating_clearance_threshold_m must be non-negative")
    if config.contact.support.support_force_threshold_n < 0:
        errors.append("support_force_threshold_n must be non-negative")
    if config.contact.cross_ratio.blocked_threshold < 0:
        errors.append("cross_ratio.blocked_threshold must be non-negative")
    return errors


def inspect_paired_state_h5(path: Path) -> ArtifactSummary:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    sha = _sha256(path) if exists and path.is_file() else ""
    if not exists:
        return ArtifactSummary(
            str(path),
            False,
            0,
            "",
            "missing",
            message="paired_g1_state.h5 is missing",
        )
    try:
        import h5py  # type: ignore
    except ImportError:
        return ArtifactSummary(
            str(path),
            True,
            size,
            sha,
            "unverified_missing_h5py",
            message="h5py is not installed",
        )

    try:
        with h5py.File(path, "r") as handle:
            datasets = tuple(sorted(_walk_h5_datasets(handle)))
            state_datasets = _state_datasets(handle)
            missing = tuple(_missing_state_datasets(state_datasets))
            shape_errors = tuple(_state_shape_errors(state_datasets))
            pred_joint = state_datasets.get("pred joint_q_rad")
            frame_count = int(pred_joint.shape[0]) if pred_joint is not None else None
            joint_count = (
                int(pred_joint.shape[1])
                if pred_joint is not None and len(pred_joint.shape) > 1
                else None
            )
            fps = _read_h5_float(handle, ("fps", "metadata/fps"))
            joint_names = _read_h5_joint_names(handle)
            return ArtifactSummary(
                str(path),
                True,
                size,
                sha,
                "ok",
                frame_count=frame_count,
                joint_count=joint_count,
                fps=fps,
                joint_names=joint_names,
                datasets=datasets,
                missing_required_datasets=missing,
                state_dataset_paths=tuple(
                    (label, dataset.name.lstrip("/"))
                    for label, dataset in sorted(state_datasets.items())
                ),
                state_dataset_shapes=tuple(
                    (label, tuple(int(dim) for dim in dataset.shape))
                    for label, dataset in sorted(state_datasets.items())
                ),
                shape_errors=shape_errors,
            )
    except OSError as exc:
        return ArtifactSummary(str(path), True, size, sha, "invalid", message=str(exc))


def build_manifest(
    *,
    config: ReplayConfig,
    artifact: ArtifactSummary,
    output_dir: Path,
    dry_run: bool,
    max_frames: int,
) -> dict[str, Any]:
    config_errors = validate_replay_config(config)
    artifact_errors = validate_artifact_summary(artifact)
    errors = config_errors + artifact_errors
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_jsonl = output_dir / "isaac_src_packets.jsonl"
    packet_schema = packet_schema_payload(config)
    manifest = {
        "status": "dry_run" if dry_run and not errors else "blocked" if errors else "ready",
        "schema_version": SCHEMA_VERSION,
        "variant": config.variant,
        "paired_state_h5": artifact_summary_payload(artifact),
        "robot_usd": str(config.robot_usd),
        "robot_usd_exists": Path(config.robot_usd).exists(),
        "robot_usd_sha256": (
            _sha256(Path(config.robot_usd)) if Path(config.robot_usd).exists() else ""
        ),
        "output_dir": str(output_dir),
        "packet_jsonl": str(packet_jsonl),
        "packet_schema_json": str(output_dir / "packet_schema.json"),
        "dry_run": dry_run,
        "max_frames": int(max_frames),
        "config": replay_config_payload(config),
        "config_errors": config_errors,
        "artifact_errors": artifact_errors,
        "acceptance_smoke": acceptance_smoke(
            config=config,
            output_dir=output_dir,
            max_frames=max_frames,
        ),
        "runtime_blocker": (
            "Non-dry run must execute on the verified 5090 Isaac/SRC runtime via "
            "$ISAAC_PYTHON after Code Reviewer approves this contact/body-pair binding."
        ),
    }
    _write_json_file(output_dir / "packet_schema.json", packet_schema)
    _write_json_file(output_dir / "replay_manifest.json", manifest)
    if dry_run:
        packet_jsonl.write_text("", encoding="utf-8")
    return manifest


def validate_artifact_summary(artifact: ArtifactSummary) -> list[str]:
    errors: list[str] = []
    if not artifact.exists:
        errors.append("paired_state_h5 is missing")
    if artifact.hdf5_status not in {"ok", "unverified_missing_h5py"}:
        errors.append(f"paired_state_h5 hdf5_status is {artifact.hdf5_status}")
    if artifact.hdf5_status == "ok":
        if artifact.missing_required_datasets:
            errors.append(
                "paired_state_h5 missing required datasets: "
                + ", ".join(artifact.missing_required_datasets)
            )
        for shape_error in artifact.shape_errors:
            errors.append(f"paired_state_h5 {shape_error}")
        if artifact.frame_count is None or artifact.frame_count <= 0:
            errors.append("paired_state_h5 frame_count must be positive")
        if artifact.joint_count != 29:
            errors.append(f"paired_state_h5 joint_count must be 29, got {artifact.joint_count}")
        if artifact.joint_names and artifact.joint_names != SONIC_JOINT_NAMES:
            errors.append("paired_state_h5 joint_names do not match SONIC_JOINT_NAMES")
    return errors


def packet_schema_payload(config: ReplayConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "format": "jsonl",
        "one_line_per_frame": True,
        "frame_packet_fields": {
            "schema_version": "string",
            "variant": "string",
            "frame_idx": "int",
            "t": "float seconds",
            "dt": "float seconds",
            "pred": "state_packet",
            "target": "state_packet",
            "contract": "stable contract hash and metric thresholds",
        },
        "state_packet_fields": {
            "root_pos_world_m": "[3] float",
            "root_quat_wxyz": "[4] float",
            "joint_q_rad": f"[{len(config.joint_names)}] float in joint_names order",
            "body_pos_world_m": f"[{len(config.body_names)}, 3] float in body_names order",
            "foot_contact_force_n": (
                f"[{len(config.contact.foot_links)}] float or null when filtered "
                "ground force is blocked"
            ),
            "foot_in_contact": f"[{len(config.contact.foot_links)}] bool",
            "foot_ground_contact_status": "available or blocked",
            "foot_ground_contact_reason": "string reason when blocked",
            "support_margin_m": "float, signed distance to configured support region",
            "floating_guard": "bool or null when foot-ground support is blocked",
            "floating_guard_status": "available or blocked",
            "floating_guard_reason": "string reason when blocked",
            "foot_slide_speed_mps": (
                f"[{len(config.contact.foot_links)}] float or null; per-foot horizontal "
                "speed while verified current and previous /World/Ground contact are true"
            ),
            "foot_slide_flags": (
                f"[{len(config.contact.foot_links)}] bool or null; true when slide speed "
                "exceeds contact.foot_slide_speed_threshold_mps"
            ),
            "foot_skate_distance_m": (
                f"[{len(config.contact.foot_links)}] float or null; rolling horizontal "
                "distance accumulated across the active verified contact segment"
            ),
            "foot_skate_flags": (
                f"[{len(config.contact.foot_links)}] bool or null; true when rolling "
                "contact segment distance exceeds contact.foot_skate_distance_threshold_m"
            ),
            "foot_float_clearance_m": (
                f"[{len(config.contact.foot_links)}] float or null; foot clearance during "
                "verified contact frames"
            ),
            "foot_float_flags": (
                f"[{len(config.contact.foot_links)}] bool or null; true when verified "
                "contact foot clearance exceeds support.floating_clearance_threshold_m"
            ),
            "foot_artifact_status": "available or blocked",
            "foot_artifact_reason": "string reason when blocked",
            "foot_ground_contact_pairs": (
                "list of verified support-only {body_a, body_b, force_n, "
                "position_world_m, source} pairs from single-body foot sensor "
                "filtered force_matrix_w"
            ),
            "contact_pairs": (
                "compatibility alias for foot_ground_contact_pairs only; not a "
                "body-body/self-collision source"
            ),
            "body_pair_contacts": (
                "list of verified body-body pairs or null when body-pair source is blocked"
            ),
            "body_pair_contact_status": "available or blocked",
            "body_pair_contact_reason": "string reason when blocked",
            "self_collision_count": (
                "int only when verified body-body source is available, otherwise null"
            ),
            "self_collision_status": "available or blocked",
            "self_collision_reason": "string reason when blocked",
            "cross_ratio": "float or null",
            "cross_ratio_guard": "bool or null",
            "cross_ratio_status": "available or blocked",
            "cross_ratio_reason": "string reason when blocked",
        },
        "joint_names": list(config.joint_names),
        "body_names": list(config.body_names),
        "foot_links": list(config.contact.foot_links),
        "disabled_collision_pairs": [
            list(pair) for pair in config.contact.disabled_collision_pairs
        ],
        "contact_filter_prim_paths": list(config.contact.contact_filter_prim_paths),
        "support_contract": asdict(config.contact.support),
        "cross_ratio_contract": asdict(config.contact.cross_ratio),
    }


def replay_config_payload(config: ReplayConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["joint_names"] = list(config.joint_names)
    payload["body_names"] = list(config.body_names)
    payload["contact"]["foot_links"] = list(config.contact.foot_links)
    payload["contact"]["disabled_collision_pairs"] = [
        list(pair) for pair in config.contact.disabled_collision_pairs
    ]
    payload["contact"]["contact_filter_prim_paths"] = list(
        config.contact.contact_filter_prim_paths
    )
    return payload


def artifact_summary_payload(summary: ArtifactSummary) -> dict[str, Any]:
    payload = asdict(summary)
    payload["joint_names"] = list(summary.joint_names)
    payload["datasets"] = list(summary.datasets)
    payload["state_dataset_paths"] = [
        [label, path] for label, path in summary.state_dataset_paths
    ]
    payload["state_dataset_shapes"] = [
        [label, list(shape)] for label, shape in summary.state_dataset_shapes
    ]
    payload["shape_errors"] = list(summary.shape_errors)
    return payload


def _mark_manifest_completed(
    *,
    manifest: dict[str, Any],
    export_result: dict[str, Any],
) -> None:
    packet_jsonl = Path(str(manifest["packet_jsonl"]))
    manifest["status"] = "completed"
    manifest["runtime_blocker"] = ""
    manifest["export_result"] = export_result
    manifest["packet_jsonl_exists"] = packet_jsonl.exists()
    manifest["packet_jsonl_bytes"] = (
        packet_jsonl.stat().st_size if packet_jsonl.exists() else 0
    )


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)
    try:
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _exit_process_success() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def acceptance_smoke(*, config: ReplayConfig, output_dir: Path, max_frames: int = 64) -> str:
    return (
        f"cd {DEFAULT_5090_REPO} && "
        f"PYTHONPATH=src:. {DEFAULT_ISAAC_PYTHON} "
        "scripts/export_lr239_isaac_src_packets.py "
        f"--paired-state-h5 {config.paired_state_h5} "
        f"--robot-usd {config.robot_usd} "
        f"--output-dir {output_dir} "
        f"--max-frames {max_frames} "
        "--dry-run && "
        f"test -s {output_dir / 'replay_manifest.json'} && "
        f"test -s {output_dir / 'packet_schema.json'}"
    )


def run_dry_or_blocked_export(args: argparse.Namespace) -> dict[str, Any]:
    config = _config_from_args(args)
    artifact = inspect_paired_state_h5(Path(config.paired_state_h5))
    manifest = build_manifest(
        config=config,
        artifact=artifact,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        max_frames=args.max_frames,
    )
    if args.dry_run:
        return manifest
    if manifest["status"] == "blocked":
        return manifest
    manifest_path = args.output_dir / "replay_manifest.json"

    def persist_completed_manifest(export_result: dict[str, Any]) -> None:
        _mark_manifest_completed(
            manifest=manifest,
            export_result=export_result,
        )
        _write_json_file(manifest_path, manifest)

    try:
        state = load_paired_state_data(
            Path(config.paired_state_h5),
            config=config,
            max_frames=args.max_frames,
        )
        backend_factory = args.backend_factory or (
            lambda replay_config: IsaacLabReplayBackend(
                replay_config,
                device=args.device,
            )
        )
        export_result = export_replay_packets(
            config=config,
            state=state,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            backend_factory=backend_factory,
            completion_callback=persist_completed_manifest,
        )
    except Exception as exc:
        manifest["status"] = "blocked"
        manifest["blocked"] = f"IsaacLab replay extraction failed: {type(exc).__name__}: {exc}"
        _write_json_file(manifest_path, manifest)
        return manifest

    persist_completed_manifest(export_result)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LR-239 Isaac/SRC replay contact packets.")
    parser.add_argument("--paired-state-h5", type=Path, required=True)
    parser.add_argument("--robot-usd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--variant", default="")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.backend_factory = None
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_dry_or_blocked_export(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest.get("status") in {"dry_run", "completed"} else 2


def load_paired_state_data(
    path: Path,
    *,
    config: ReplayConfig,
    max_frames: int,
) -> PairedStateData:
    summary = inspect_paired_state_h5(path)
    errors = validate_artifact_summary(summary)
    if errors:
        raise ValueError("; ".join(errors))
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("h5py and numpy are required to load paired G1 state") from exc

    limit = max(1, int(max_frames))
    with h5py.File(path, "r") as handle:
        datasets = _state_datasets(handle)
        frame_count = int(summary.frame_count or 0)
        usable = min(frame_count, limit)
        pred_root_pos = np.asarray(datasets["pred root_pos_world_m"][:usable], dtype=np.float32)
        target_root_pos = np.asarray(datasets["target root_pos_world_m"][:usable], dtype=np.float32)
        pred_root_quat = _normalize_quat_array(
            np.asarray(datasets["pred root_quat_wxyz"][:usable], dtype=np.float32)
        )
        target_root_quat = _normalize_quat_array(
            np.asarray(datasets["target root_quat_wxyz"][:usable], dtype=np.float32)
        )
        pred_joint = np.asarray(datasets["pred joint_q_rad"][:usable], dtype=np.float32)
        target_joint = np.asarray(datasets["target joint_q_rad"][:usable], dtype=np.float32)
    joint_names = summary.joint_names or config.joint_names
    return PairedStateData(
        frame_count=usable,
        fps=float(summary.fps or config.fps),
        joint_names=tuple(joint_names),
        pred_root_pos_world_m=pred_root_pos,
        target_root_pos_world_m=target_root_pos,
        pred_root_quat_wxyz=pred_root_quat,
        target_root_quat_wxyz=target_root_quat,
        pred_joint_q_rad=pred_joint,
        target_joint_q_rad=target_joint,
    )


def export_replay_packets(
    *,
    config: ReplayConfig,
    state: PairedStateData,
    output_dir: Path,
    max_frames: int,
    backend_factory: Callable[[ReplayConfig], ReplayBackend],
    completion_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "isaac_src_packets.jsonl"
    frame_limit = min(state.frame_count, max(1, int(max_frames)))
    dt = 1.0 / float(state.fps)
    packets_written = 0
    backend_report: dict[str, Any] = {}
    export_result: dict[str, Any] = {}
    with backend_factory(config) as backend:
        with packet_path.open("w", encoding="utf-8") as packet_file:
            for frame_idx in range(frame_limit):
                pred = backend.collect_state_packet(
                    ReplayStateFrame(
                        frame_idx=frame_idx,
                        label="pred",
                        root_pos_world_m=_row(state.pred_root_pos_world_m, frame_idx),
                        root_quat_wxyz=_row(state.pred_root_quat_wxyz, frame_idx),
                        joint_q_rad=_row(state.pred_joint_q_rad, frame_idx),
                    )
                )
                target = backend.collect_state_packet(
                    ReplayStateFrame(
                        frame_idx=frame_idx,
                        label="target",
                        root_pos_world_m=_row(state.target_root_pos_world_m, frame_idx),
                        root_quat_wxyz=_row(state.target_root_quat_wxyz, frame_idx),
                        joint_q_rad=_row(state.target_joint_q_rad, frame_idx),
                    )
                )
                packet = {
                    "schema_version": SCHEMA_VERSION,
                    "variant": config.variant,
                    "frame_idx": frame_idx,
                    "t": frame_idx * dt,
                    "dt": dt,
                    "pred": pred,
                    "target": target,
                    "contract": _contract_hash(config),
                }
                packet_file.write(json.dumps(packet, sort_keys=True) + "\n")
                packets_written += 1
        backend_report = backend.report()
        hard_exit_after_success = bool(
            getattr(backend, "requires_hard_exit_after_success", False)
        )
        export_result = {
            "backend": backend_report,
            "packet_jsonl": str(packet_path),
            "packet_jsonl_exists": packet_path.exists(),
            "packet_jsonl_bytes": packet_path.stat().st_size if packet_path.exists() else 0,
            "packets_written": packets_written,
            "frame_limit": frame_limit,
            "fps": float(state.fps),
            "lifecycle_exit_strategy": (
                "os._exit(0)_after_completed_manifest"
                if hard_exit_after_success
                else "normal_context_exit"
            ),
        }
        if completion_callback is not None:
            completion_callback(export_result)
        if hard_exit_after_success:
            _exit_process_success()
    return export_result


class IsaacLabReplayBackend:
    """Bounded IsaacLab replay backend used only for non-dry packet extraction."""

    requires_hard_exit_after_success = True

    def __init__(self, config: ReplayConfig, *, device: str = "cuda:0") -> None:
        self.config = config
        self.device = device
        self.sim: Any | None = None
        self.robot: Any | None = None
        self.foot_contact_sensors: dict[str, Any] = {}
        self.body_pair_contact_source: Any | None = None
        self.app_launcher: Any | None = None
        self.simulation_app: Any | None = None
        self.torch: Any | None = None
        self.body_names: tuple[str, ...] = config.body_names
        self.robot_joint_order: list[int] = []
        self.body_indices: list[int] = []
        self.foot_body_indices: list[int] = []
        self.foot_artifact_tracker = FootArtifactTracker(config)
        self.frames_collected = 0

    def __enter__(self) -> "IsaacLabReplayBackend":
        try:
            from isaaclab.app import AppLauncher  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "IsaacLab non-dry export requires isaaclab.app.AppLauncher; run with "
                f"{DEFAULT_ISAAC_PYTHON} on the verified 5090 Isaac runtime"
            ) from exc

        launcher_parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(launcher_parser)
        launcher_args = launcher_parser.parse_args([])
        launcher_args.headless = True
        launcher_args.enable_cameras = False
        if hasattr(launcher_args, "device"):
            launcher_args.device = self.device
        self.app_launcher = AppLauncher(launcher_args)
        self.simulation_app = self.app_launcher.app
        try:
            import torch  # type: ignore
            import isaacsim.core.utils.stage as stage_utils  # type: ignore
            import isaaclab.sim as sim_utils  # type: ignore
            from isaaclab.actuators import ImplicitActuatorCfg  # type: ignore
            from isaaclab.assets import Articulation, ArticulationCfg  # type: ignore
            from isaaclab.sensors import ContactSensor, ContactSensorCfg  # type: ignore
        except ImportError as exc:
            self._close_simulation_app()
            raise RuntimeError(
                "IsaacLab non-dry export requires isaacsim/isaaclab modules; run with "
                f"{DEFAULT_ISAAC_PYTHON} on the verified 5090 Isaac runtime"
            ) from exc

        self.torch = torch
        stage_utils.create_new_stage()
        dt = 1.0 / max(float(self.config.fps), 1.0)
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=dt, render_interval=1, device=self.device)
        )
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func(self.config.contact.support.ground_prim_path, ground_cfg)
        robot_cfg = ArticulationCfg(
            prim_path=self.config.root_prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.config.robot_usd,
                activate_contact_sensors=self.config.contact.activate_contact_sensors,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=self.config.contact.enable_self_collisions,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=True,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.8),
                joint_pos={".*": 0.0},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0,
                    damping=0.0,
                )
            },
        )
        self.robot = Articulation(robot_cfg)
        filter_paths = tuple(self.config.contact.contact_filter_prim_paths)
        self.foot_contact_sensors = {
            foot_link: ContactSensor(
                ContactSensorCfg(
                    prim_path=single_body_contact_sensor_prim_path(self.config, foot_link),
                    update_period=0.0,
                    history_length=1,
                    debug_vis=False,
                    track_contact_points=True,
                    max_contact_data_count_per_prim=16,
                    force_threshold=self.config.contact.contact_force_threshold_n,
                    filter_prim_paths_expr=filter_paths,
                )
            )
            for foot_link in self.config.contact.foot_links
        }
        self.sim.reset()
        self.robot.update(self.sim.cfg.dt)
        for sensor in self.foot_contact_sensors.values():
            sensor.update(dt=self.sim.cfg.dt)
        self.robot_joint_order = _index_order(
            self.robot.joint_names,
            self.config.joint_names,
            label="robot joints",
        )
        robot_body_names = tuple(str(name) for name in self.robot.body_names)
        self.body_indices = _index_order(
            robot_body_names,
            self.config.body_names,
            label="robot bodies",
        )
        self.body_names = tuple(robot_body_names[index] for index in self.body_indices)
        self.foot_body_indices = [
            self.body_names.index(foot_link) for foot_link in self.config.contact.foot_links
        ]
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.sim is not None:
            callbacks = (
                getattr(self.sim, "stop", None),
                getattr(self.sim, "clear_instance", None),
            )
            for callback in callbacks:
                if callback is None:
                    continue
                try:
                    callback()
                except Exception:
                    pass
        self._close_simulation_app()

    def _close_simulation_app(self) -> None:
        app = self.simulation_app
        if app is None:
            return
        try:
            if app.is_running():
                app.close()
        except Exception:
            pass
        finally:
            self.simulation_app = None

    def collect_state_packet(self, state: ReplayStateFrame) -> dict[str, Any]:
        if self.robot is None or self.sim is None or self.torch is None:
            raise RuntimeError("IsaacLabReplayBackend must be entered before collection")
        torch = self.torch
        joint_pos = [float(state.joint_q_rad[index]) for index in self.robot_joint_order]
        joint_vel = [0.0 for _ in joint_pos]
        root_state = torch.zeros((1, 13), dtype=torch.float32, device=self.sim.device)
        root_state[0, :3] = torch.as_tensor(
            state.root_pos_world_m,
            dtype=torch.float32,
            device=self.sim.device,
        )
        root_state[0, 3:7] = torch.as_tensor(
            _normalize_quat_list(state.root_quat_wxyz),
            dtype=torch.float32,
            device=self.sim.device,
        )
        self.robot.write_root_state_to_sim(root_state)
        self.robot.write_joint_state_to_sim(
            torch.as_tensor([joint_pos], dtype=torch.float32, device=self.sim.device),
            torch.as_tensor([joint_vel], dtype=torch.float32, device=self.sim.device),
        )
        self.robot.set_joint_position_target(
            torch.as_tensor([joint_pos], dtype=torch.float32, device=self.sim.device)
        )
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(self.sim.cfg.dt)
        for sensor in self.foot_contact_sensors.values():
            sensor.update(dt=self.sim.cfg.dt)
        self.frames_collected += 1
        body_pos = _tensor_to_nested_list(self.robot.data.body_pos_w[0, self.body_indices, :])
        foot_ground_state = self._foot_contact_state()
        foot_ground_contact_pairs = self._foot_ground_contact_pairs(
            body_pos=body_pos,
            foot_ground_state=foot_ground_state,
        )
        foot_positions = [body_pos[index] for index in self.foot_body_indices]
        foot_artifacts = self.foot_artifact_tracker.update(
            label=state.label,
            foot_positions=foot_positions,
            foot_in_contact=foot_ground_state.flags,
            foot_ground_contact_status=foot_ground_state.status,
            foot_ground_contact_reason=foot_ground_state.reason,
        )
        body_pair_payload = _blocked_body_pair_payload()
        src_payload = _blocked_src_payload()
        support_margin = self._support_margin(body_pos)
        floating_guard = None
        if foot_ground_state.status == "available":
            floating_guard = all(not flag for flag in foot_ground_state.flags) and (
                support_margin > self.config.contact.support.floating_clearance_threshold_m
            )
        return {
            "root_pos_world_m": _float_list(state.root_pos_world_m),
            "root_quat_wxyz": _normalize_quat_list(state.root_quat_wxyz),
            "joint_q_rad": _float_list(state.joint_q_rad),
            "body_names": list(self.body_names),
            "body_pos_world_m": body_pos,
            "foot_contact_force_n": list(foot_ground_state.forces_n),
            "foot_in_contact": list(foot_ground_state.flags),
            "foot_ground_contact_status": foot_ground_state.status,
            "foot_ground_contact_reason": foot_ground_state.reason,
            "support_margin_m": support_margin,
            "floating_guard": floating_guard,
            "floating_guard_status": foot_ground_state.status,
            "floating_guard_reason": foot_ground_state.reason,
            "foot_slide_speed_mps": list(foot_artifacts.foot_slide_speed_mps),
            "foot_slide_flags": list(foot_artifacts.foot_slide_flags),
            "foot_skate_distance_m": list(foot_artifacts.foot_skate_distance_m),
            "foot_skate_flags": list(foot_artifacts.foot_skate_flags),
            "foot_float_clearance_m": list(foot_artifacts.foot_float_clearance_m),
            "foot_float_flags": list(foot_artifacts.foot_float_flags),
            "foot_artifact_status": foot_artifacts.foot_artifact_status,
            "foot_artifact_reason": foot_artifacts.foot_artifact_reason,
            "foot_ground_contact_pairs": foot_ground_contact_pairs,
            "contact_pairs": foot_ground_contact_pairs,
            **body_pair_payload,
            **src_payload,
        }

    def report(self) -> dict[str, Any]:
        return {
            "backend": "isaaclab_usd_g1_contact_replay",
            "device": self.device,
            "robot_usd": self.config.robot_usd,
            "root_prim_path": self.config.root_prim_path,
            "body_names": list(self.body_names),
            "foot_links": list(self.config.contact.foot_links),
            "foot_ground_contact_sensor_prim_paths": foot_ground_contact_sensor_prim_paths(
                self.config
            ),
            "frames_collected": self.frames_collected,
            "foot_ground_contact_status": "filtered_force_matrix_required",
            "body_pair_contact_status": "blocked",
            "body_pair_contact_reason": BODY_PAIR_CONTACT_BLOCKED_REASON,
            "cross_ratio_status": "blocked",
            "cross_ratio_reason": SRC_BLOCKED_REASON,
        }

    def _foot_contact_state(self) -> FootGroundContactState:
        forces: list[float | None] = []
        flags: list[bool] = []
        blocked_reasons: list[str] = []
        for foot_link in self.config.contact.foot_links:
            reading = filtered_ground_contact_force_norm(
                self.foot_contact_sensors.get(foot_link),
                self.config,
            )
            forces.append(reading.force_n)
            flags.append(
                reading.force_n is not None
                and reading.force_n >= self.config.contact.contact_force_threshold_n
            )
            if reading.status != "available":
                blocked_reasons.append(f"{foot_link}: {reading.reason}")
        if blocked_reasons:
            return FootGroundContactState(
                forces_n=tuple(forces),
                flags=tuple(flags),
                status="blocked",
                reason="; ".join(blocked_reasons),
            )
        return FootGroundContactState(
            forces_n=tuple(forces),
            flags=tuple(flags),
            status="available",
        )

    def _foot_ground_contact_pairs(
        self,
        *,
        body_pos: Sequence[Sequence[float]],
        foot_ground_state: FootGroundContactState,
    ) -> list[dict[str, Any]]:
        if foot_ground_state.status != "available":
            return []
        pairs: list[dict[str, Any]] = []
        for foot_link, force in zip(self.config.contact.foot_links, foot_ground_state.forces_n):
            if force is None or force < self.config.contact.contact_force_threshold_n:
                continue
            body_index = self.body_names.index(foot_link)
            pairs.append(
                {
                    "body_a": foot_link,
                    "body_b": self.config.contact.support.ground_prim_path,
                    "force_n": float(force),
                    "position_world_m": list(body_pos[body_index]),
                    "source": "single_body_foot_ground_filtered_force_matrix",
                }
            )
        return pairs

    def _support_margin(self, body_pos: Sequence[Sequence[float]]) -> float:
        axis_index = {"x": 0, "y": 1, "z": 2}[self.config.contact.support.up_axis.lower()]
        if not self.foot_body_indices:
            return math.inf
        min_foot_height = min(
            float(body_pos[index][axis_index]) for index in self.foot_body_indices
        )
        return min_foot_height - float(self.config.contact.support.ground_height_m)


class FootArtifactTracker:
    """Rolling foot artifact state derived only from verified foot-ground contact."""

    def __init__(self, config: ReplayConfig) -> None:
        self.config = config
        self.axis_index = {"x": 0, "y": 1, "z": 2}[
            config.contact.support.up_axis.lower()
        ]
        self.horizontal_axes = tuple(index for index in range(3) if index != self.axis_index)
        self.dt = 1.0 / max(float(config.fps), 1.0)
        self._previous: dict[str, tuple[tuple[float, float, float], ...]] = {}
        self._previous_contact: dict[str, tuple[bool, ...]] = {}
        self._segment_start: dict[tuple[str, int], tuple[float, float, float]] = {}

    def update(
        self,
        *,
        label: str,
        foot_positions: Sequence[Sequence[float]],
        foot_in_contact: Sequence[bool],
        foot_ground_contact_status: str,
        foot_ground_contact_reason: str = "",
    ) -> FootArtifactState:
        foot_count = len(self.config.contact.foot_links)
        if foot_ground_contact_status != "available":
            self._clear_label(label)
            reason = foot_ground_contact_reason or "verified foot-ground contact is unavailable"
            return FootArtifactState(
                foot_slide_speed_mps=tuple(None for _ in range(foot_count)),
                foot_slide_flags=tuple(None for _ in range(foot_count)),
                foot_skate_distance_m=tuple(None for _ in range(foot_count)),
                foot_skate_flags=tuple(None for _ in range(foot_count)),
                foot_float_clearance_m=tuple(None for _ in range(foot_count)),
                foot_float_flags=tuple(None for _ in range(foot_count)),
                foot_artifact_status="blocked",
                foot_artifact_reason=reason,
            )
        if len(foot_positions) != foot_count or len(foot_in_contact) != foot_count:
            self._clear_label(label)
            return FootArtifactState(
                foot_slide_speed_mps=tuple(None for _ in range(foot_count)),
                foot_slide_flags=tuple(None for _ in range(foot_count)),
                foot_skate_distance_m=tuple(None for _ in range(foot_count)),
                foot_skate_flags=tuple(None for _ in range(foot_count)),
                foot_float_clearance_m=tuple(None for _ in range(foot_count)),
                foot_float_flags=tuple(None for _ in range(foot_count)),
                foot_artifact_status="blocked",
                foot_artifact_reason="foot pose/contact width does not match foot_links",
            )

        current = tuple(_point3(position) for position in foot_positions)
        current_contact = tuple(bool(flag) for flag in foot_in_contact)
        previous = self._previous.get(label)
        previous_contact = self._previous_contact.get(label)
        slide_speeds: list[float | None] = []
        slide_flags: list[bool | None] = []
        skate_distances: list[float | None] = []
        skate_flags: list[bool | None] = []
        float_clearances: list[float | None] = []
        float_flags: list[bool | None] = []

        for foot_index, position in enumerate(current):
            in_contact = current_contact[foot_index]
            prev_in_contact = (
                previous_contact is not None
                and foot_index < len(previous_contact)
                and previous_contact[foot_index]
            )
            segment_key = (label, foot_index)
            if in_contact:
                self._segment_start.setdefault(segment_key, position)
                clearance = self._clearance(position)
                float_clearances.append(clearance)
                float_flags.append(
                    clearance > self.config.contact.support.floating_clearance_threshold_m
                )
                distance = self._horizontal_distance(
                    self._segment_start[segment_key],
                    position,
                )
                skate_distances.append(distance)
                skate_flags.append(distance > self.config.contact.foot_skate_distance_threshold_m)
                if previous is not None and prev_in_contact:
                    speed = self._horizontal_distance(previous[foot_index], position) / self.dt
                    slide_speeds.append(speed)
                    slide_flags.append(
                        speed > self.config.contact.foot_slide_speed_threshold_mps
                    )
                else:
                    slide_speeds.append(None)
                    slide_flags.append(None)
            else:
                self._segment_start.pop(segment_key, None)
                slide_speeds.append(None)
                slide_flags.append(None)
                skate_distances.append(None)
                skate_flags.append(None)
                float_clearances.append(None)
                float_flags.append(None)

        self._previous[label] = current
        self._previous_contact[label] = current_contact
        return FootArtifactState(
            foot_slide_speed_mps=tuple(slide_speeds),
            foot_slide_flags=tuple(slide_flags),
            foot_skate_distance_m=tuple(skate_distances),
            foot_skate_flags=tuple(skate_flags),
            foot_float_clearance_m=tuple(float_clearances),
            foot_float_flags=tuple(float_flags),
            foot_artifact_status="available",
        )

    def _clear_label(self, label: str) -> None:
        self._previous.pop(label, None)
        self._previous_contact.pop(label, None)
        for key in tuple(self._segment_start):
            if key[0] == label:
                self._segment_start.pop(key, None)

    def _clearance(self, point: Sequence[float]) -> float:
        return float(point[self.axis_index]) - float(
            self.config.contact.support.ground_height_m
        )

    def _horizontal_distance(
        self,
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        return math.sqrt(
            sum(
                (float(left[index]) - float(right[index])) ** 2
                for index in self.horizontal_axes
            )
        )


def _config_from_args(args: argparse.Namespace) -> ReplayConfig:
    base = load_replay_config(args.config)
    payload = replay_config_payload(base)
    payload["paired_state_h5"] = str(args.paired_state_h5)
    payload["robot_usd"] = str(args.robot_usd)
    if args.variant:
        payload["variant"] = args.variant
    return replay_config_from_mapping(payload)


def _row(array: Any, index: int) -> list[float]:
    return _float_list(array[index])


def _float_list(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def _point3(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"foot position must have three values, got {len(values)}")
    return (float(values[0]), float(values[1]), float(values[2]))


def _tensor_to_nested_list(value: Any) -> list[list[float]]:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        pass
    return [[float(component) for component in row] for row in value]


def _normalize_quat_array(values: Any) -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy is required to normalize root_quat_wxyz arrays") from exc
    quats = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if bool(np.any(norms <= 0.0)):
        raise ValueError("root_quat_wxyz contains zero-norm quaternion")
    return quats / norms


def _normalize_quat_list(values: Sequence[float]) -> list[float]:
    quat = [float(value) for value in values]
    if len(quat) != 4:
        raise ValueError(f"root_quat_wxyz must have four values, got {len(quat)}")
    norm = math.sqrt(sum(value * value for value in quat))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("root_quat_wxyz contains zero or non-finite norm")
    return [value / norm for value in quat]


def _index_order(available: Sequence[str], required: Sequence[str], *, label: str) -> list[int]:
    lookup = {str(name): index for index, name in enumerate(available)}
    missing = [str(name) for name in required if str(name) not in lookup]
    if missing:
        raise RuntimeError(f"{label} missing required names: {missing}")
    return [lookup[str(name)] for name in required]


def single_body_contact_sensor_prim_path(config: ReplayConfig, body_name: str) -> str:
    return f"{config.root_prim_path.rstrip('/')}/{body_name}"


def foot_ground_contact_sensor_prim_paths(config: ReplayConfig) -> dict[str, str]:
    return {
        foot_link: single_body_contact_sensor_prim_path(config, foot_link)
        for foot_link in config.contact.foot_links
    }


def filtered_ground_contact_force_norm(
    contact_sensor: Any | None,
    config: ReplayConfig,
) -> FilteredContactForce:
    if contact_sensor is None or not hasattr(contact_sensor, "data"):
        return FilteredContactForce(None, "blocked", "foot contact sensor is unavailable")
    try:
        ground_filter_index = config.contact.contact_filter_prim_paths.index(
            config.contact.support.ground_prim_path
        )
    except ValueError:
        return FilteredContactForce(
            None,
            "blocked",
            "configured contact filters do not include the support ground prim",
        )
    forces = getattr(contact_sensor.data, "force_matrix_w", None)
    if forces is None:
        return FilteredContactForce(None, "blocked", FOOT_GROUND_FILTER_BLOCKED_REASON)
    try:
        forces = forces.detach().cpu().numpy()
    except AttributeError:
        pass
    if getattr(forces, "ndim", 0) == 3:
        forces = forces[0]
    elif getattr(forces, "ndim", 0) == 4:
        if len(forces) == 0:
            return FilteredContactForce(None, "blocked", FOOT_GROUND_FILTER_BLOCKED_REASON)
        forces = forces[0]
    if getattr(forces, "ndim", 0) == 3:
        if len(forces) == 0:
            return FilteredContactForce(None, "blocked", FOOT_GROUND_FILTER_BLOCKED_REASON)
        forces = forces[0]
    if getattr(forces, "ndim", 0) == 2:
        if ground_filter_index >= len(forces):
            return FilteredContactForce(None, "blocked", FOOT_GROUND_FILTER_BLOCKED_REASON)
        vector = forces[ground_filter_index]
    else:
        return FilteredContactForce(None, "blocked", FOOT_GROUND_FILTER_BLOCKED_REASON)
    return FilteredContactForce(
        math.sqrt(sum(float(component) ** 2 for component in vector)),
        "available",
    )


def _blocked_body_pair_payload() -> dict[str, Any]:
    return {
        "body_pair_contacts": None,
        "body_pair_contact_status": "blocked",
        "body_pair_contact_reason": BODY_PAIR_CONTACT_BLOCKED_REASON,
        "self_collision_count": None,
        "self_collision_status": "blocked",
        "self_collision_reason": BODY_PAIR_CONTACT_BLOCKED_REASON,
    }


def _blocked_src_payload() -> dict[str, Any]:
    return {
        "cross_ratio": None,
        "cross_ratio_guard": None,
        "cross_ratio_status": "blocked",
        "cross_ratio_reason": SRC_BLOCKED_REASON,
    }


def _contract_hash(config: ReplayConfig) -> str:
    payload = json.dumps(replay_config_payload(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _pair(value: object, label: str) -> tuple[str, str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} entries must be two-item sequences")
    return (str(value[0]), str(value[1]))


def _canonical_link_names(config: ReplayConfig) -> frozenset[str]:
    links = {"pelvis", "torso_link"}
    for joint_name in config.joint_names:
        if joint_name.endswith("_joint"):
            links.add(joint_name[: -len("_joint")] + "_link")
    links.update({"left_ankle_roll_link", "right_ankle_roll_link"})
    return frozenset(links)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _walk_h5_datasets(group: Any, prefix: str = "") -> list[str]:
    try:
        import h5py  # type: ignore
    except ImportError:  # pragma: no cover - caller guards this.
        return []
    datasets: list[str] = []
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, h5py.Dataset):
            datasets.append(path)
        elif isinstance(value, h5py.Group):
            datasets.extend(_walk_h5_datasets(value, path))
    return datasets


def _first_dataset(handle: Any, names: Sequence[str]) -> Any | None:
    for name in names:
        if name in handle:
            return handle[name]
    return None


def _state_datasets(handle: Any) -> dict[str, Any]:
    return {
        label: dataset
        for label, candidates in REQUIRED_STATE_DATASETS.items()
        if (dataset := _first_dataset(handle, candidates)) is not None
    }


def _missing_state_datasets(state_datasets: Mapping[str, Any]) -> list[str]:
    return [label for label in REQUIRED_STATE_DATASETS if label not in state_datasets]


def _state_shape_errors(state_datasets: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_last_dim = {
        "pred root_pos_world_m": 3,
        "target root_pos_world_m": 3,
        "pred root_quat_wxyz": 4,
        "target root_quat_wxyz": 4,
        "pred joint_q_rad": 29,
        "target joint_q_rad": 29,
    }
    frame_counts: dict[str, int] = {}
    for label, width in expected_last_dim.items():
        dataset = state_datasets.get(label)
        if dataset is None:
            continue
        shape = tuple(int(dim) for dim in dataset.shape)
        if len(shape) != 2:
            errors.append(f"{label} must be rank-2 [N,{width}], got {shape}")
            continue
        if shape[1] != width:
            errors.append(f"{label} width must be {width}, got {shape[1]}")
        frame_counts[label] = shape[0]
    positive_counts = {label: count for label, count in frame_counts.items() if count > 0}
    if len(positive_counts) != len(frame_counts):
        bad = ", ".join(label for label, count in frame_counts.items() if count <= 0)
        errors.append(f"state datasets must have positive frame count: {bad}")
    if frame_counts:
        unique_counts = sorted(set(frame_counts.values()))
        if len(unique_counts) > 1:
            rendered = ", ".join(f"{label}={count}" for label, count in frame_counts.items())
            errors.append(f"pred/target state frame counts must match: {rendered}")
    return errors


def _read_h5_float(handle: Any, names: Sequence[str]) -> float | None:
    for name in names:
        if name in handle.attrs:
            return float(handle.attrs[name])
        if name in handle:
            value = handle[name][()]
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _read_h5_joint_names(handle: Any) -> tuple[str, ...]:
    for owner in (handle, handle.get("pred_g1_state"), handle.get("target_g1_state")):
        if owner is None:
            continue
        for key in ("joint_names_json", "joint_names"):
            if key in owner.attrs:
                return _parse_joint_names(owner.attrs[key])
            if key in owner:
                return _parse_joint_names(owner[key][()])
    return ()


def _parse_joint_names(value: object) -> tuple[str, ...]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = [part for part in value.split(",") if part]
        return tuple(str(part) for part in decoded)
    try:
        return tuple(
            str(part.decode("utf-8") if isinstance(part, bytes) else part)
            for part in value  # type: ignore[union-attr]
        )
    except TypeError:
        return ()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
