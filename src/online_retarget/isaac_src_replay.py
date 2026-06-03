"""LR-239 Isaac/SRC replay exporter contract and preflight helpers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "lr239.isaac_src_contact_packets.v1"
DEFAULT_G1_USD = (
    "/mnt/data_cpfs/code/wxh/OnlineRetarget/runs/isaaclab_urdf_cache/"
    "g1_main/main.usd"
)
DEFAULT_5090_REPO = "/mnt/data_cpfs/code/wxh/OnlineRetarget"
DEFAULT_ISAAC_PYTHON = "/workspace/isaaclab/_isaac_sim/python.sh"

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
    message: str = ""


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
    if set(config.contact.foot_links) - set(_canonical_link_names(config)):
        errors.append("contact.foot_links must be named G1 links in the canonical model contract")
    if not config.contact.foot_links:
        errors.append("contact.foot_links must not be empty")
    if not config.contact.contact_filter_prim_paths:
        errors.append("contact.contact_filter_prim_paths must include ground/filter prims")
    if not config.contact.enable_self_collisions:
        errors.append(
            "contact.enable_self_collisions must be true for self_collision_count packets"
        )
    if not config.contact.activate_contact_sensors:
        errors.append("contact.activate_contact_sensors must be true for PhysX contact packets")
    if config.contact.contact_force_threshold_n < 0:
        errors.append("contact.contact_force_threshold_n must be non-negative")
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
            pred_joint = _first_dataset(handle, ("pred_g1_state/joint_q_rad", "pred/joint_q_rad"))
            target_joint = _first_dataset(
                handle,
                ("target_g1_state/joint_q_rad", "target/joint_q_rad"),
            )
            missing = tuple(_missing_state_datasets(handle))
            if pred_joint is None or target_joint is None:
                return ArtifactSummary(
                    str(path),
                    True,
                    size,
                    sha,
                    "invalid",
                    datasets=datasets,
                    missing_required_datasets=missing,
                    message="missing pred/target joint_q_rad datasets",
                )
            frame_count = int(pred_joint.shape[0])
            joint_count = int(pred_joint.shape[1]) if len(pred_joint.shape) > 1 else None
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
            "Non-dry run must execute on the 5090 Isaac/SRC runtime via $ISAAC_PYTHON after the "
            "contact sensor/body-pair extraction code is bound to the verified G1 USD."
        ),
    }
    (output_dir / "packet_schema.json").write_text(
        json.dumps(packet_schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
            "foot_contact_force_n": f"[{len(config.contact.foot_links)}] float",
            "foot_in_contact": f"[{len(config.contact.foot_links)}] bool",
            "support_margin_m": "float, signed distance to configured support region",
            "floating_guard": "bool",
            "self_collision_count": "int",
            "contact_pairs": "list of {body_a, body_b, force_n, position_world_m}",
            "cross_ratio": "float or null",
            "cross_ratio_guard": "bool or null",
        },
        "joint_names": list(config.joint_names),
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
    return payload


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
    manifest["status"] = "blocked"
    manifest["blocked"] = (
        "Isaac/SRC replay packet extraction is not implemented in this local pass. "
        "Patch surface is scripts/export_lr239_isaac_src_packets.py plus "
        "online_retarget.isaac_src_replay: bind AppLauncher, spawn G1 USD with "
        "activate_contact_sensors=True/enabled_self_collisions=True, instantiate "
        "ContactSensorCfg for foot links filtered to ground, replay pred/target states, "
        "and serialize packet_schema.json-compatible JSONL."
    )
    (args.output_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LR-239 Isaac/SRC replay contact packets.")
    parser.add_argument("--paired-state-h5", type=Path, required=True)
    parser.add_argument("--robot-usd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--variant", default="")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_dry_or_blocked_export(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest.get("status") in {"dry_run", "ready"} else 2


def _config_from_args(args: argparse.Namespace) -> ReplayConfig:
    base = load_replay_config(args.config)
    payload = replay_config_payload(base)
    payload["paired_state_h5"] = str(args.paired_state_h5)
    payload["robot_usd"] = str(args.robot_usd)
    if args.variant:
        payload["variant"] = args.variant
    return replay_config_from_mapping(payload)


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


def _missing_state_datasets(handle: Any) -> list[str]:
    checks = {
        "pred root_pos": (
            "pred_g1_state/root_pos_world_m",
            "pred_g1_state/root_pos",
            "pred/root_pos_world_m",
            "pred/root_pos",
        ),
        "target root_pos": (
            "target_g1_state/root_pos_world_m",
            "target_g1_state/root_pos",
            "target/root_pos_world_m",
            "target/root_pos",
        ),
        "pred root_rot": (
            "pred_g1_state/root_quat_wxyz",
            "pred_g1_state/root_rot_wxyz",
            "pred_g1_state/root_rot",
            "pred/root_quat_wxyz",
            "pred/root_rot_wxyz",
            "pred/root_rot",
        ),
        "target root_rot": (
            "target_g1_state/root_quat_wxyz",
            "target_g1_state/root_rot_wxyz",
            "target_g1_state/root_rot",
            "target/root_quat_wxyz",
            "target/root_rot_wxyz",
            "target/root_rot",
        ),
        "pred joint_q_rad": ("pred_g1_state/joint_q_rad", "pred/joint_q_rad"),
        "target joint_q_rad": ("target_g1_state/joint_q_rad", "target/joint_q_rad"),
    }
    return [
        label
        for label, candidates in checks.items()
        if _first_dataset(handle, candidates) is None
    ]


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
