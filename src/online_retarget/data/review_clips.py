"""Export reviewable source/target clips from BONES-SEED tar archives."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from typing import Mapping, Sequence

from .bones_seed import G1_JOINT_COLUMNS
from .g1_quality import G1KinematicModel, g1_fk_body_positions, load_g1_kinematic_model
from .windowed_builder import (
    BVHMotion,
    DEFAULT_SOURCE_BODY_NAMES,
    global_body_position_maps_from_bvh,
    parse_bvh_motion,
)


DEFAULT_REVIEW_CLIP_COLUMNS = (
    "quality_action",
    "quality_flags",
    "merged_quality_action",
    "merged_quality_flags",
    "review_family",
    "row_index",
    "split",
    "category",
    "actor_uid",
    "filename",
    "move_soma_proportional_path",
    "move_g1_path",
    "source_frame_count",
    "g1_frame_count",
    "abs_frame_count_delta",
    "abs_duration_delta_sec",
    "source_duration_sec",
    "g1_duration_sec",
)

QUALITY_METRIC_COLUMNS = (
    "max_abs_joint_velocity",
    "joint_jump_rate",
    "max_abs_joint_acceleration",
    "max_root_acceleration",
    "max_root_jerk",
    "max_root_speed",
    "max_start_end_root_speed",
    "joint_limit_violation_rate",
    "max_joint_limit_violation",
    "mean_foot_clearance",
    "penetration_depth",
    "contact_frame_ratio",
    "contact_slide_rate",
    "max_contact_slide_speed",
    "self_collision_proxy_rate",
    "min_self_collision_distance",
)

_FAMILY_METRIC_COLUMNS = {
    "g1_foot_slide": ("contact_slide_rate", "max_contact_slide_speed"),
    "g1_joint_limit_violation": ("joint_limit_violation_rate", "max_joint_limit_violation"),
    "g1_unstable_start_end": ("max_start_end_root_speed",),
    "g1_ground_penetration": ("penetration_depth",),
    "joint_velocity_jump": ("joint_jump_rate", "max_abs_joint_velocity"),
    "g1_foot_float": ("mean_foot_clearance",),
    "g1_self_collision_proxy": ("self_collision_proxy_rate", "min_self_collision_distance"),
    "g1_low_foot_contact": ("contact_frame_ratio",),
}

SOURCE_CAPSULE_BODY_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)

G1_CAPSULE_IGNORE_BODIES = (
    "head_mocap",
    "imu_in_torso",
    "pelvis_contour_link",
)


@dataclass(frozen=True)
class ReviewClipExportConfig:
    """Configuration for exporting clip files and optional G1 preview videos."""

    limit: int = 8
    source_tar_name: str = "soma_proportional.tar"
    g1_tar_name: str = "g1.tar"
    source_path_column: str = "move_soma_proportional_path"
    g1_path_column: str = "move_g1_path"
    render_g1: bool = False
    render_source_capsules: bool = False
    render_g1_capsules: bool = False
    render_max_frames: int = 120
    render_width: int = 640
    render_height: int = 360
    fps: float = 120.0
    root_position_scale: float = 0.01
    source_position_scale: float = 0.01
    angle_scale: float = math.pi / 180.0
    render_frames: bool = True
    model_xml: Path | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_xml"] = str(self.model_xml) if self.model_xml is not None else None
        return payload


@dataclass(frozen=True)
class ReviewClipExportResult:
    output_dir: Path
    summary_csv: Path
    summary_json: Path
    readme_md: Path
    exported_rows: int
    render_counts: dict[str, int]
    source_render_counts: dict[str, int]
    g1_capsule_render_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["summary_csv"] = str(self.summary_csv)
        payload["summary_json"] = str(self.summary_json)
        payload["readme_md"] = str(self.readme_md)
        return payload


def export_review_clips(
    *,
    data_root: Path,
    input_csv: Path,
    output_root: Path,
    run_name: str,
    label: str,
    config: ReviewClipExportConfig | None = None,
) -> ReviewClipExportResult:
    """Export source BVH, G1 CSV, metadata, and optional G1 MP4 previews."""

    config = config or ReviewClipExportConfig()
    _validate_config(config)
    data_root = data_root.expanduser()
    input_csv = input_csv.expanduser()
    output_dir = output_root.expanduser() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_csv, config.limit)
    source_tar_path = data_root / config.source_tar_name
    g1_tar_path = data_root / config.g1_tar_name
    render_counts: Counter[str] = Counter()
    source_render_counts: Counter[str] = Counter()
    g1_capsule_render_counts: Counter[str] = Counter()
    exported: list[dict[str, object]] = []
    renderer = _G1Renderer(config) if config.render_g1 else None
    source_renderer = _SourceCapsuleRenderer(config) if config.render_source_capsules else None
    g1_capsule_renderer = _G1CapsuleRenderer(config) if config.render_g1_capsules else None

    with tarfile.open(source_tar_path, "r:*") as source_tar, tarfile.open(g1_tar_path, "r:*") as g1_tar:
        for index, row in enumerate(rows):
            clip_dir = output_dir / _clip_dir_name(index, label, row)
            clip_dir.mkdir(parents=True, exist_ok=True)
            source_path = clip_dir / "source_soma_proportional.bvh"
            g1_path = clip_dir / "target_g1.csv"
            metadata_path = clip_dir / "metadata.json"
            source_bytes = _extract_member(source_tar, str(row[config.source_path_column]), source_path)
            g1_bytes = _extract_member(g1_tar, str(row[config.g1_path_column]), g1_path)
            render_report: dict[str, object] = {"status": "not_requested"}
            video_path = clip_dir / "target_g1_mujoco.mp4"
            if renderer is not None:
                render_report = renderer.render_csv(g1_path, video_path)
            render_counts[str(render_report.get("status", "unknown"))] += 1
            source_render_report: dict[str, object] = {"status": "not_requested"}
            source_video_path = clip_dir / "source_bvh_capsules.mp4"
            if source_renderer is not None:
                source_render_report = source_renderer.render_bvh(source_path, source_video_path)
            source_render_counts[str(source_render_report.get("status", "unknown"))] += 1
            g1_capsule_render_report: dict[str, object] = {"status": "not_requested"}
            g1_capsule_video_path = clip_dir / "target_g1_3d_capsules.mp4"
            if g1_capsule_renderer is not None:
                g1_capsule_render_report = g1_capsule_renderer.render_csv(
                    g1_path,
                    g1_capsule_video_path,
                )
            g1_capsule_render_counts[str(g1_capsule_render_report.get("status", "unknown"))] += 1

            metadata = _metadata_for_row(
                row,
                label=label,
                sample_index=index,
                config=config,
                source_tar_path=source_tar_path,
                g1_tar_path=g1_tar_path,
                source_path=source_path,
                g1_path=g1_path,
                video_path=video_path if video_path.exists() else None,
                source_video_path=source_video_path if source_video_path.exists() else None,
                g1_capsule_video_path=(
                    g1_capsule_video_path if g1_capsule_video_path.exists() else None
                ),
                source_bytes=source_bytes,
                g1_bytes=g1_bytes,
                render_report=render_report,
                source_render_report=source_render_report,
                g1_capsule_render_report=g1_capsule_render_report,
            )
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            exported.append(metadata)

    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    readme_md = output_dir / "README.md"
    _write_summary_csv(summary_csv, exported)
    _write_summary_json(
        summary_json,
        output_dir,
        input_csv,
        exported,
        render_counts,
        source_render_counts,
        g1_capsule_render_counts,
        config,
    )
    _write_readme(
        readme_md,
        input_csv,
        exported,
        render_counts,
        source_render_counts,
        g1_capsule_render_counts,
        config,
    )
    return ReviewClipExportResult(
        output_dir=output_dir,
        summary_csv=summary_csv,
        summary_json=summary_json,
        readme_md=readme_md,
        exported_rows=len(exported),
        render_counts=dict(sorted(render_counts.items())),
        source_render_counts=dict(sorted(source_render_counts.items())),
        g1_capsule_render_counts=dict(sorted(g1_capsule_render_counts.items())),
    )


def _read_rows(input_csv: Path, limit: int) -> list[dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for _, row in zip(range(limit), reader)]


def _extract_member(tar: tarfile.TarFile, member_name: str, output_path: Path) -> int:
    member = tar.getmember(member_name)
    extracted = tar.extractfile(member)
    if extracted is None:
        raise ValueError(f"empty tar member: {member_name}")
    with extracted, output_path.open("wb") as handle:
        while True:
            chunk = extracted.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return output_path.stat().st_size


def _metadata_for_row(
    row: Mapping[str, str],
    *,
    label: str,
    sample_index: int,
    config: ReviewClipExportConfig,
    source_tar_path: Path,
    g1_tar_path: Path,
    source_path: Path,
    g1_path: Path,
    video_path: Path | None,
    source_video_path: Path | None,
    g1_capsule_video_path: Path | None,
    source_bytes: int,
    g1_bytes: int,
    render_report: Mapping[str, object],
    source_render_report: Mapping[str, object],
    g1_capsule_render_report: Mapping[str, object],
) -> dict[str, object]:
    metadata = {
        "label": label,
        "sample_index": sample_index,
        "source_archive": str(source_tar_path),
        "target_archive": str(g1_tar_path),
        "source_bvh": str(source_path),
        "target_g1_csv": str(g1_path),
        "target_g1_video": str(video_path) if video_path is not None else "",
        "source_bvh_capsule_video": str(source_video_path) if source_video_path is not None else "",
        "target_g1_capsule_video": (
            str(g1_capsule_video_path) if g1_capsule_video_path is not None else ""
        ),
        "source_bvh_bytes": source_bytes,
        "target_g1_csv_bytes": g1_bytes,
        "render": dict(render_report),
        "source_render": dict(source_render_report),
        "g1_capsule_render": dict(g1_capsule_render_report),
        "render_note": (
            "G1 target CSV visualization only; not learned retargeter output or Isaac Lab evaluation."
        ),
        "source_render_note": (
            "3D capsule visualization of BVH FK source soma_proportional motion; not SMPL mesh rendering."
        ),
        "g1_capsule_render_note": (
            "3D capsule visualization of the paired G1 target CSV via MJCF FK; not Isaac Lab rollout."
        ),
    }
    for column in DEFAULT_REVIEW_CLIP_COLUMNS:
        metadata[column] = row.get(column, "")
    for column in QUALITY_METRIC_COLUMNS:
        metadata[column] = row.get(column, "")
    metadata["quality_action"] = str(
        row.get("quality_action") or row.get("merged_quality_action") or ""
    )
    metadata["quality_flags"] = str(
        row.get("quality_flags") or row.get("merged_quality_flags") or ""
    )
    metadata["render_fps"] = config.fps
    metadata["render_max_frames"] = config.render_max_frames
    return metadata


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "label",
        "sample_index",
        "quality_action",
        "quality_flags",
        "merged_quality_action",
        "merged_quality_flags",
        "review_family",
        "split",
        "category",
        "actor_uid",
        "filename",
        "move_soma_proportional_path",
        "move_g1_path",
        "source_frame_count",
        "g1_frame_count",
        "abs_frame_count_delta",
        "abs_duration_delta_sec",
        "source_duration_sec",
        "g1_duration_sec",
        "source_bvh",
        "source_bvh_capsule_video",
        "target_g1_csv",
        "target_g1_video",
        "target_g1_capsule_video",
        "source_render_status",
        "source_render_message",
        "render_status",
        "render_message",
        "g1_capsule_render_status",
        "g1_capsule_render_message",
        *QUALITY_METRIC_COLUMNS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            render = row.get("render") if isinstance(row.get("render"), Mapping) else {}
            source_render = row.get("source_render") if isinstance(row.get("source_render"), Mapping) else {}
            g1_capsule_render = (
                row.get("g1_capsule_render")
                if isinstance(row.get("g1_capsule_render"), Mapping)
                else {}
            )
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fieldnames},
                    "source_render_status": source_render.get("status", ""),
                    "source_render_message": source_render.get("message", ""),
                    "render_status": render.get("status", ""),
                    "render_message": render.get("message", ""),
                    "g1_capsule_render_status": g1_capsule_render.get("status", ""),
                    "g1_capsule_render_message": g1_capsule_render.get("message", ""),
                }
            )


def _write_summary_json(
    path: Path,
    output_dir: Path,
    input_csv: Path,
    rows: Sequence[Mapping[str, object]],
    render_counts: Mapping[str, int],
    source_render_counts: Mapping[str, int],
    g1_capsule_render_counts: Mapping[str, int],
    config: ReviewClipExportConfig,
) -> None:
    payload = {
        "output_root": str(output_dir),
        "input_csv": str(input_csv),
        "sample_count": len(rows),
        "render_status": dict(sorted(render_counts.items())),
        "source_render_status": dict(sorted(source_render_counts.items())),
        "g1_capsule_render_status": dict(sorted(g1_capsule_render_counts.items())),
        "summary_csv": str(output_dir / "summary.csv"),
        "config": config.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_readme(
    path: Path,
    input_csv: Path,
    rows: Sequence[Mapping[str, object]],
    render_counts: Mapping[str, int],
    source_render_counts: Mapping[str, int],
    g1_capsule_render_counts: Mapping[str, int],
    config: ReviewClipExportConfig,
) -> None:
    text = f"""# Review Clips

Input CSV: `{input_csv}`

Exported rows: {len(rows)}
G1 render status: {dict(sorted(render_counts.items()))}
Source BVH 3D capsule render status: {dict(sorted(source_render_counts.items()))}
G1 target 3D capsule render status: {dict(sorted(g1_capsule_render_counts.items()))}

Each sample directory contains:

- `source_soma_proportional.bvh`
- `source_bvh_capsules.mp4` when `render_source_capsules=true` and 3D capsule rendering succeeds
- `target_g1.csv`
- `target_g1_mujoco.mp4` when `render_g1=true` and rendering succeeds
- `target_g1_3d_capsules.mp4` when `render_g1_capsules=true` and rendering succeeds
- `metadata.json`

The G1 videos visualize the paired G1 target CSV only. The capsule videos are 3D FK review views with a perspective camera and ground grid; they are not learned retargeter predictions, SMPL mesh renders, or Isaac Lab rollouts.
When present, `review_family` and metric columns come from the input review CSV so each clip can be traced back to the specific quality flag that selected it.

## Samples

{_sample_table(rows)}

Config:

```json
{json.dumps(config.to_dict(), indent=2, sort_keys=True)}
```
"""
    path.write_text(text, encoding="utf-8")


def _sample_table(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return "_No samples exported._"
    headers = (
        "index",
        "review_family",
        "filename",
        "action",
        "source_render",
        "g1_render",
        "g1_capsule",
        "metrics",
        "source_video",
        "g1_video",
        "g1_capsule_video",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        render = row.get("render") if isinstance(row.get("render"), Mapping) else {}
        source_render = row.get("source_render") if isinstance(row.get("source_render"), Mapping) else {}
        g1_capsule_render = (
            row.get("g1_capsule_render")
            if isinstance(row.get("g1_capsule_render"), Mapping)
            else {}
        )
        cells = (
            str(row.get("sample_index", "")),
            str(row.get("review_family", "")),
            str(row.get("filename", "")),
            str(row.get("quality_action", "")),
            str(source_render.get("status", "")),
            str(render.get("status", "")),
            str(g1_capsule_render.get("status", "")),
            _metric_summary(row),
            str(row.get("source_bvh_capsule_video", "")),
            str(row.get("target_g1_video", "")),
            str(row.get("target_g1_capsule_video", "")),
        )
        lines.append("| " + " | ".join(_md_cell(cell) for cell in cells) + " |")
    return "\n".join(lines)


def _metric_summary(row: Mapping[str, object]) -> str:
    family = str(row.get("review_family", ""))
    columns = _FAMILY_METRIC_COLUMNS.get(family, QUALITY_METRIC_COLUMNS)
    parts: list[str] = []
    for column in columns:
        value = row.get(column, "")
        if value in (None, ""):
            continue
        parts.append(f"{column}={value}")
    return "; ".join(parts)


def _md_cell(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return text.replace("|", "\\|")


def _clip_dir_name(index: int, label: str, row: Mapping[str, str]) -> str:
    filename = row.get("filename") or Path(row.get("move_g1_path", "clip")).stem
    return f"{index:02d}_{_safe_name(label)}_{_safe_name(filename)}"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe[:120] or "clip"


def _validate_config(config: ReviewClipExportConfig) -> None:
    if config.limit < 0:
        raise ValueError("limit must be non-negative")
    if config.fps <= 0:
        raise ValueError("fps must be positive")
    if config.render_max_frames < 0:
        raise ValueError("render_max_frames must be non-negative; use 0 for full length")
    if config.render_width <= 0 or config.render_height <= 0:
        raise ValueError("render dimensions must be positive")
    if config.root_position_scale <= 0:
        raise ValueError("root_position_scale must be positive")
    if config.source_position_scale <= 0:
        raise ValueError("source_position_scale must be positive")
    if config.angle_scale <= 0:
        raise ValueError("angle_scale must be positive")
    if config.render_g1 and config.model_xml is None:
        raise ValueError("render_g1 requires model_xml")
    if config.render_g1_capsules and config.model_xml is None:
        raise ValueError("render_g1_capsules requires model_xml")


class _SourceCapsuleRenderer:
    def __init__(self, config: ReviewClipExportConfig) -> None:
        self._config = config

    def render_bvh(self, bvh_path: Path, video_path: Path) -> dict[str, object]:
        try:
            text = bvh_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
            motion = parse_bvh_motion(text)
            body_names = _source_capsule_body_names(motion)
            edges = _source_capsule_edges_from_motion(motion, body_names)
            frames = _limit_render_frames(
                global_body_position_maps_from_bvh(
                    motion,
                    body_names=body_names,
                    position_scale=self._config.source_position_scale,
                ),
                self._config.render_max_frames,
            )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return {"status": "failed", "message": f"Could not load BVH source motion: {exc}"}
        return _render_capsule_3d_video(
            frames=frames,
            edges=edges,
            video_path=video_path,
            config=self._config,
            label="source bvh 3d capsules",
            up_axis=1,
            capsule_color=(48, 132, 83),
            key_color=(132, 103, 34),
        )


class _G1CapsuleRenderer:
    def __init__(self, config: ReviewClipExportConfig) -> None:
        if config.model_xml is None:
            raise ValueError("render_g1_capsules requires model_xml")
        self._config = config
        self._model = load_g1_kinematic_model(config.model_xml)
        self._edges = _g1_capsule_edges(self._model)

    def render_csv(self, g1_csv: Path, video_path: Path) -> dict[str, object]:
        try:
            trajectory = _g1_csv_to_trajectory(g1_csv, self._config)
            frames = _g1_capsule_frames(self._model, trajectory)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return {"status": "failed", "message": f"Could not load G1 target motion: {exc}"}
        return _render_capsule_3d_video(
            frames=frames,
            edges=self._edges,
            video_path=video_path,
            config=self._config,
            label="g1 target 3d capsules",
            up_axis=2,
            capsule_color=(54, 105, 166),
            key_color=(130, 88, 35),
        )


class _G1Renderer:
    def __init__(self, config: ReviewClipExportConfig) -> None:
        if config.model_xml is None:
            raise ValueError("render_g1 requires model_xml")
        os.environ.setdefault("MUJOCO_GL", "egl")
        self._config = config
        try:
            import mujoco  # type: ignore[import-not-found]

            from online_retarget.web_pipeline import _render_mujoco_g1_video
        except Exception as exc:
            raise RuntimeError(f"MuJoCo G1 rendering dependencies are unavailable: {exc}") from exc
        self._mujoco = mujoco
        self._render_mujoco_g1_video = _render_mujoco_g1_video
        self._model = mujoco.MjModel.from_xml_path(str(config.model_xml))

    def render_csv(self, g1_csv: Path, video_path: Path) -> dict[str, object]:
        data = self._mujoco.MjData(self._model)
        trajectory = _g1_csv_to_trajectory(g1_csv, self._config)
        return self._render_mujoco_g1_video(
            self._mujoco,
            self._model,
            data,
            trajectory,
            video_path=video_path,
            frame_time=1.0 / self._config.fps,
            render_frames=self._config.render_frames,
            width=self._config.render_width,
            height=self._config.render_height,
        )


def _g1_csv_to_trajectory(path: Path, config: ReviewClipExportConfig) -> list[dict[str, object]]:
    trajectory: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if config.render_max_frames > 0 and index >= config.render_max_frames:
                break
            root_euler = [
                float(row["root_rotateX"]) * config.angle_scale,
                float(row["root_rotateY"]) * config.angle_scale,
                float(row["root_rotateZ"]) * config.angle_scale,
            ]
            trajectory.append(
                {
                    "frame": int(float(row.get("Frame") or index)),
                    "root": [
                        float(row["root_translateX"]) * config.root_position_scale,
                        float(row["root_translateY"]) * config.root_position_scale,
                        float(row["root_translateZ"]) * config.root_position_scale,
                    ],
                    "root_euler": root_euler,
                    "root_quat": _euler_xyz_to_quat_wxyz(root_euler),
                    "joints": {
                        column: float(row[column]) * config.angle_scale
                        for column in G1_JOINT_COLUMNS
                    },
                }
            )
    return trajectory


def _g1_capsule_edges(model: G1KinematicModel) -> tuple[tuple[str, str], ...]:
    ignored = set(G1_CAPSULE_IGNORE_BODIES)
    edges: list[tuple[str, str]] = []
    for body in model.bodies:
        if body.name in ignored or body.parent is None:
            continue
        parent_name = model.bodies[body.parent].name
        if parent_name in ignored:
            continue
        edges.append((parent_name, body.name))
    return tuple(edges)


def _g1_capsule_frames(
    model: G1KinematicModel,
    trajectory: Sequence[Mapping[str, object]],
) -> list[dict[str, tuple[float, float, float]]]:
    frames: list[dict[str, tuple[float, float, float]]] = []
    ignored = set(G1_CAPSULE_IGNORE_BODIES)
    for item in trajectory:
        joints = item.get("joints", {})
        if not isinstance(joints, Mapping):
            joints = {}
        joint_values = [float(joints.get(column, 0.0)) for column in G1_JOINT_COLUMNS]
        body_points = g1_fk_body_positions(
            model,
            joint_values,
            root_position=_float_tuple(item.get("root", (0.0, 0.0, 0.0)), 3),
            root_euler=_float_tuple(item.get("root_euler", (0.0, 0.0, 0.0)), 3),
            include_empty_body_origin=True,
        )
        frame: dict[str, tuple[float, float, float]] = {}
        for name, points in body_points.items():
            if name in ignored or not points:
                continue
            frame[name] = _centroid(points)
        frames.append(frame)
    return frames


def _render_capsule_3d_video(
    *,
    frames: Sequence[Mapping[str, tuple[float, float, float]]],
    edges: Sequence[tuple[str, str]],
    video_path: Path,
    config: ReviewClipExportConfig,
    label: str,
    up_axis: int,
    capsule_color: tuple[int, int, int],
    key_color: tuple[int, int, int],
) -> dict[str, object]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"status": "blocked", "message": "ffmpeg is required to encode 3D capsule video."}
    if not frames:
        return {"status": "blocked", "message": "No frames were available for 3D capsule rendering."}

    z_up_frames = _z_up_frames(frames, up_axis=up_axis)
    bounds = _capsule_scene_bounds(z_up_frames)
    width = config.render_width
    height = config.render_height
    fps = int(_clamp(round(config.fps), 1, 240))
    video_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    process = None
    frame_sums: list[int] = []
    changed_frames = 0
    previous_frame: bytes | None = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdin is None:
            return {"status": "failed", "message": "ffmpeg stdin was not available."}
        for frame_index, frame in enumerate(z_up_frames):
            image = _draw_capsule_3d_frame(
                frame,
                frame_index,
                bounds,
                edges,
                width,
                height,
                label=label,
                capsule_color=capsule_color,
                key_color=key_color,
            )
            frame_bytes = bytes(image)
            process.stdin.write(frame_bytes)
            frame_sums.append(sum(frame_bytes))
            if previous_frame is not None and frame_bytes != previous_frame:
                changed_frames += 1
            previous_frame = frame_bytes
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.stderr is not None:
            process.stderr.close()
        return_code = process.wait()
        if return_code != 0:
            return {
                "status": "failed",
                "message": "ffmpeg failed while encoding 3D capsule video.",
                "ffmpeg_tail": stderr[-800:],
            }
        if not video_path.exists() or video_path.stat().st_size == 0:
            return {"status": "failed", "message": "3D capsule video was not written."}
        return {
            "status": "ok",
            "message": "Encoded 3D capsule review video.",
            "video_path": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frames": len(z_up_frames),
            "capsule_edges": len(edges),
            "capsule_edge_pairs": [f"{start}->{end}" for start, end in edges],
            "changed_frames": changed_frames,
            "frame_sum_min": min(frame_sums),
            "frame_sum_max": max(frame_sums),
            "render_backend": "software_perspective_capsules",
            "up_axis": up_axis,
            "ground_z": 0.0,
        }
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.kill()
        return {"status": "failed", "message": f"3D capsule render failed: {exc}"}


def _z_up_frames(
    frames: Sequence[Mapping[str, tuple[float, float, float]]],
    *,
    up_axis: int,
) -> list[dict[str, tuple[float, float, float]]]:
    converted: list[dict[str, tuple[float, float, float]]] = []
    min_z: float | None = None
    for frame in frames:
        converted_frame: dict[str, tuple[float, float, float]] = {}
        for name, point in frame.items():
            world_point = _to_z_up(point, up_axis=up_axis)
            if not all(math.isfinite(value) for value in world_point):
                continue
            converted_frame[name] = world_point
            min_z = world_point[2] if min_z is None else min(min_z, world_point[2])
        converted.append(converted_frame)
    if min_z is None:
        return converted
    return [
        {name: (point[0], point[1], point[2] - min_z) for name, point in frame.items()}
        for frame in converted
    ]


def _to_z_up(point: Sequence[float], *, up_axis: int) -> tuple[float, float, float]:
    x, y, z = (float(point[index]) if index < len(point) else 0.0 for index in range(3))
    if up_axis == 1:
        return (x, z, y)
    return (x, y, z)


def _capsule_scene_bounds(
    frames: Sequence[Mapping[str, tuple[float, float, float]]],
) -> dict[str, float]:
    points = [point for frame in frames for point in frame.values()]
    if not points:
        return {
            "min_x": -0.5,
            "max_x": 0.5,
            "min_y": -0.5,
            "max_y": 0.5,
            "min_z": 0.0,
            "max_z": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "center_z": 0.5,
            "radius": 1.0,
        }
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    span_x = max(max_x - min_x, 0.2)
    span_y = max(max_y - min_y, 0.2)
    span_z = max(max_z - min_z, 0.2)
    radius = max(math.sqrt(span_x * span_x + span_y * span_y + span_z * span_z) * 0.5, 0.5)
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z,
        "center_x": (min_x + max_x) * 0.5,
        "center_y": (min_y + max_y) * 0.5,
        "center_z": (min_z + max_z) * 0.5,
        "radius": radius,
    }


def _draw_capsule_3d_frame(
    frame: Mapping[str, tuple[float, float, float]],
    frame_index: int,
    bounds: Mapping[str, float],
    edges: Sequence[tuple[str, str]],
    width: int,
    height: int,
    *,
    label: str,
    capsule_color: tuple[int, int, int],
    key_color: tuple[int, int, int],
) -> bytearray:
    image = bytearray((242, 244, 239) * width * height)
    camera = _capsule_camera(bounds, width, height)
    _draw_3d_ground_grid(image, width, height, camera, bounds)
    projected = {
        name: _project_camera(point, camera, width, height)
        for name, point in frame.items()
    }

    segments: list[tuple[float, tuple[int, int], tuple[int, int], int, tuple[int, int, int]]] = []
    shadows: list[tuple[float, tuple[int, int], tuple[int, int], int]] = []
    for start, end in edges:
        a = projected.get(start)
        b = projected.get(end)
        if a is None or b is None:
            continue
        depth = (a[2] + b[2]) * 0.5
        radius = int(_clamp(0.035 * float(camera["focal"]) / max(depth, 0.05), 3, 18))
        segments.append(
            (
                depth,
                (a[0], a[1]),
                (b[0], b[1]),
                radius,
                _depth_color(capsule_color, depth, float(camera["distance"])),
            )
        )
        shadow_a = _project_camera((frame[start][0], frame[start][1], 0.0), camera, width, height)
        shadow_b = _project_camera((frame[end][0], frame[end][1], 0.0), camera, width, height)
        if shadow_a is not None and shadow_b is not None:
            shadows.append((depth + 0.01, (shadow_a[0], shadow_a[1]), (shadow_b[0], shadow_b[1]), radius))

    for _, start, end, radius in sorted(shadows, key=lambda item: item[0], reverse=True):
        _draw_line(image, width, height, start, end, radius=max(1, radius // 2), color=(196, 201, 194))
    for depth, start, end, radius, color in sorted(segments, key=lambda item: item[0], reverse=True):
        outline = _depth_color((34, 47, 47), depth, float(camera["distance"]))
        _draw_line(image, width, height, start, end, radius=radius + 2, color=outline)
        _draw_line(image, width, height, start, end, radius=radius, color=color)

    points: list[tuple[float, tuple[int, int], int, tuple[int, int, int]]] = []
    for name, projected_point in projected.items():
        if projected_point is None:
            continue
        depth = projected_point[2]
        key = _key_source_point(name) or name in ("pelvis", "torso_link", "head_link")
        radius = int(_clamp((0.045 if key else 0.032) * float(camera["focal"]) / max(depth, 0.05), 3, 16))
        points.append(
            (
                depth,
                (projected_point[0], projected_point[1]),
                radius,
                _depth_color(key_color if key else capsule_color, depth, float(camera["distance"])),
            )
        )
    for _, center, radius, color in sorted(points, key=lambda item: item[0], reverse=True):
        _draw_circle(image, width, height, center, radius=radius + 1, color=(34, 47, 47))
        _draw_circle(image, width, height, center, radius=radius, color=color)

    _draw_text(image, width, height, 14, 14, f"{label} frame {frame_index:04d}", (61, 72, 67))
    return image


def _capsule_camera(bounds: Mapping[str, float], width: int, height: int) -> dict[str, object]:
    radius = float(bounds["radius"])
    fov = math.radians(38.0)
    distance = max(radius / math.tan(fov * 0.5) * 1.35, 1.6)
    azimuth = math.radians(135.0)
    elevation = math.radians(18.0)
    target = (
        float(bounds["center_x"]),
        float(bounds["center_y"]),
        max(0.35, float(bounds["center_z"])),
    )
    camera_position = (
        target[0] + distance * math.cos(elevation) * math.cos(azimuth),
        target[1] + distance * math.cos(elevation) * math.sin(azimuth),
        target[2] + distance * math.sin(elevation),
    )
    forward = _normalize3(_sub3(target, camera_position))
    right = _normalize3(_cross3(forward, (0.0, 0.0, 1.0)))
    if _norm3(right) <= 1e-9:
        right = (1.0, 0.0, 0.0)
    up = _normalize3(_cross3(right, forward))
    return {
        "position": camera_position,
        "forward": forward,
        "right": right,
        "up": up,
        "distance": distance,
        "focal": 0.5 * min(width, height) / math.tan(fov * 0.5),
    }


def _project_camera(
    point: tuple[float, float, float],
    camera: Mapping[str, object],
    width: int,
    height: int,
) -> tuple[int, int, float] | None:
    position = camera["position"]
    forward = camera["forward"]
    right = camera["right"]
    up = camera["up"]
    if not isinstance(position, tuple) or not isinstance(forward, tuple):
        return None
    if not isinstance(right, tuple) or not isinstance(up, tuple):
        return None
    rel = _sub3(point, position)
    depth = _dot3(rel, forward)
    if depth <= 0.05:
        return None
    focal = float(camera["focal"])
    x = width * 0.5 + (_dot3(rel, right) * focal / depth)
    y = height * 0.57 - (_dot3(rel, up) * focal / depth)
    return (int(round(x)), int(round(y)), depth)


def _draw_3d_ground_grid(
    image: bytearray,
    width: int,
    height: int,
    camera: Mapping[str, object],
    bounds: Mapping[str, float],
) -> None:
    half = max(
        float(bounds["max_x"]) - float(bounds["min_x"]),
        float(bounds["max_y"]) - float(bounds["min_y"]),
        1.0,
    ) * 0.75
    center_x = float(bounds["center_x"])
    center_y = float(bounds["center_y"])
    step = max(0.25, half / 4.0)
    start_x = center_x - half
    end_x = center_x + half
    start_y = center_y - half
    end_y = center_y + half
    count = int(math.ceil((2.0 * half) / step))
    for index in range(count + 1):
        x = start_x + index * step
        a = _project_camera((x, start_y, 0.0), camera, width, height)
        b = _project_camera((x, end_y, 0.0), camera, width, height)
        if a is not None and b is not None:
            _draw_line(image, width, height, (a[0], a[1]), (b[0], b[1]), radius=0, color=(211, 216, 208))
        y = start_y + index * step
        a = _project_camera((start_x, y, 0.0), camera, width, height)
        b = _project_camera((end_x, y, 0.0), camera, width, height)
        if a is not None and b is not None:
            _draw_line(image, width, height, (a[0], a[1]), (b[0], b[1]), radius=0, color=(211, 216, 208))


def _depth_color(
    color: tuple[int, int, int],
    depth: float,
    camera_distance: float,
) -> tuple[int, int, int]:
    factor = _clamp(1.12 - depth / max(camera_distance * 2.5, 1.0), 0.55, 1.08)
    return tuple(int(_clamp(channel * factor, 0, 255)) for channel in color)


def _centroid(points: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    count = len(points)
    if count == 0:
        return (0.0, 0.0, 0.0)
    return (
        sum(float(point[0]) for point in points) / count,
        sum(float(point[1]) for point in points) / count,
        sum(float(point[2]) for point in points) / count,
    )


def _float_tuple(value: object, length: int) -> tuple[float, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed = [float(item) for item in value[:length]]
    else:
        parsed = []
    while len(parsed) < length:
        parsed.append(0.0)
    return tuple(parsed[:length])


def _sub3(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _dot3(left: Sequence[float], right: Sequence[float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross3(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm3(value: Sequence[float]) -> float:
    return math.sqrt(_dot3(value, value))


def _normalize3(value: Sequence[float]) -> tuple[float, float, float]:
    norm = _norm3(value)
    if norm <= 1e-9:
        return (0.0, 0.0, 0.0)
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def _limit_render_frames(
    frames: list[dict[str, tuple[float, float, float]]],
    max_frames: int,
) -> list[dict[str, tuple[float, float, float]]]:
    if max_frames <= 0:
        return frames
    return frames[:max_frames]


def _source_capsule_body_names(motion: BVHMotion) -> tuple[str, ...]:
    available = {joint.name for joint in motion.joints}
    names = [name for name in SOURCE_CAPSULE_BODY_NAMES if name in available]
    if names:
        return tuple(names)
    return tuple(name for name in DEFAULT_SOURCE_BODY_NAMES if name in available)


def _source_capsule_edges_from_motion(
    motion: BVHMotion,
    body_names: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    selected = set(body_names)
    edges: list[tuple[str, str]] = []
    for joint in motion.joints:
        if joint.parent is None or joint.name not in selected:
            continue
        parent_name = motion.joints[joint.parent].name
        if parent_name in selected:
            edges.append((parent_name, joint.name))
    return tuple(edges)


def _projection_bounds(frames: Sequence[Mapping[str, tuple[float, float, float]]]) -> dict[str, float]:
    projected: list[tuple[float, float]] = []
    for frame in frames:
        for point in frame.values():
            projected.append(_project_iso(point))
    if not projected:
        return {"min_x": -1.0, "max_x": 1.0, "min_y": -1.0, "max_y": 1.0}
    xs = [point[0] for point in projected]
    ys = [point[1] for point in projected]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if max_x - min_x < 0.1:
        min_x -= 0.05
        max_x += 0.05
    if max_y - min_y < 0.1:
        min_y -= 0.05
        max_y += 0.05
    return {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}


def _draw_source_capsule_frame(
    frame: Mapping[str, tuple[float, float, float]],
    frame_index: int,
    bounds: Mapping[str, float],
    edges: Sequence[tuple[str, str]],
    width: int,
    height: int,
) -> bytes:
    image = bytearray((239, 242, 237) * width * height)
    _draw_grid(image, width, height)
    projected = {
        name: _project_to_pixel(point, bounds, width, height)
        for name, point in frame.items()
    }
    for start, end in edges:
        if start not in projected or end not in projected:
            continue
        a = projected[start]
        b = projected[end]
        _draw_line(image, width, height, a, b, radius=9, color=(166, 185, 188))
        _draw_line(image, width, height, a, b, radius=4, color=(55, 128, 91))
    for name, point in projected.items():
        radius = 5 if _key_source_point(name) else 3
        color = (126, 106, 43) if _key_source_point(name) else (31, 122, 77)
        _draw_circle(image, width, height, point, radius=radius, color=color)
    _draw_text(image, width, height, 14, 14, f"source BVH capsules  frame {frame_index:04d}", (70, 82, 74))
    return bytes(image)


def _project_iso(point: Sequence[float]) -> tuple[float, float]:
    x, y, z = (float(point[index]) if index < len(point) else 0.0 for index in range(3))
    return (x - y * 0.35, z + y * 0.18)


def _project_to_pixel(
    point: Sequence[float],
    bounds: Mapping[str, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    iso_x, iso_y = _project_iso(point)
    min_x = float(bounds["min_x"])
    max_x = float(bounds["max_x"])
    min_y = float(bounds["min_y"])
    max_y = float(bounds["max_y"])
    span_x = max(0.1, max_x - min_x)
    span_y = max(0.1, max_y - min_y)
    scale = min((width * 0.78) / span_x, (height * 0.72) / span_y)
    x = width / 2.0 + (iso_x - (min_x + max_x) / 2.0) * scale
    y = height * 0.72 - (iso_y - min_y) * scale
    return (int(round(x)), int(round(y)))


def _draw_grid(image: bytearray, width: int, height: int) -> None:
    color = (218, 224, 216)
    for x in range(0, width, 46):
        _draw_line(image, width, height, (x, 0), (x, height - 1), radius=0, color=color)
    for y in range(0, height, 46):
        _draw_line(image, width, height, (0, y), (width - 1, y), radius=0, color=color)


def _draw_line(
    image: bytearray,
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy), 1)
    for index in range(steps + 1):
        t = index / steps
        x = int(round(x0 + dx * t))
        y = int(round(y0 + dy * t))
        if radius <= 0:
            _set_pixel(image, width, height, x, y, color)
        else:
            _draw_circle(image, width, height, (x, y), radius=radius, color=color)


def _draw_circle(
    image: bytearray,
    width: int,
    height: int,
    center: tuple[int, int],
    *,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    cx, cy = center
    radius_sq = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius_sq:
                _set_pixel(image, width, height, x, y, color)


def _set_pixel(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    offset = (y * width + x) * 3
    image[offset : offset + 3] = bytes(color)


_FONT_5X7: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "a": ("00000", "01110", "00001", "01111", "10001", "10011", "01101"),
    "b": ("10000", "10000", "10110", "11001", "10001", "10001", "11110"),
    "c": ("00000", "01110", "10001", "10000", "10000", "10001", "01110"),
    "d": ("00001", "00001", "01101", "10011", "10001", "10001", "01111"),
    "e": ("00000", "01110", "10001", "11111", "10000", "10001", "01110"),
    "f": ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    "g": ("00000", "01111", "10001", "10001", "01111", "00001", "01110"),
    "h": ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "m": ("00000", "11010", "10101", "10101", "10101", "10101", "10101"),
    "n": ("00000", "10110", "11001", "10001", "10001", "10001", "10001"),
    "o": ("00000", "01110", "10001", "10001", "10001", "10001", "01110"),
    "p": ("00000", "11110", "10001", "10001", "11110", "10000", "10000"),
    "r": ("00000", "10110", "11001", "10000", "10000", "10000", "10000"),
    "s": ("00000", "01111", "10000", "01110", "00001", "00001", "11110"),
    "t": ("01000", "01000", "11100", "01000", "01000", "01001", "00110"),
    "u": ("00000", "10001", "10001", "10001", "10001", "10011", "01101"),
    "v": ("00000", "10001", "10001", "10001", "10001", "01010", "00100"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
}


def _draw_text(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
) -> None:
    cursor = x
    for char in text.lower():
        glyph = _FONT_5X7.get(char)
        if glyph is None:
            cursor += 6
            continue
        for row_index, row in enumerate(glyph):
            for col_index, value in enumerate(row):
                if value == "1":
                    _set_pixel(image, width, height, cursor + col_index, y + row_index, color)
        cursor += 6


def _key_source_point(name: str) -> bool:
    return (
        "Head" in name
        or "Hand" in name
        or "Foot" in name
        or "Toe" in name
        or "toe" in name
        or "hand" in name
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _euler_xyz_to_quat_wxyz(euler_xyz: Sequence[float]) -> list[float]:
    rx, ry, rz = (float(euler_xyz[index]) if index < len(euler_xyz) else 0.0 for index in range(3))
    qx = (math.cos(rx / 2.0), math.sin(rx / 2.0), 0.0, 0.0)
    qy = (math.cos(ry / 2.0), 0.0, math.sin(ry / 2.0), 0.0)
    qz = (math.cos(rz / 2.0), 0.0, 0.0, math.sin(rz / 2.0))
    return _normalize_quat(_quat_mul(_quat_mul(qx, qy), qz))


def _quat_mul(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _normalize_quat(q: tuple[float, float, float, float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in q))
    if norm <= 0.0 or not math.isfinite(norm):
        return [1.0, 0.0, 0.0, 0.0]
    return [value / norm for value in q]
