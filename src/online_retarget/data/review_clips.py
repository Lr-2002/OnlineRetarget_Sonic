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
import tarfile
from typing import Mapping, Sequence

from .bones_seed import G1_JOINT_COLUMNS


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


@dataclass(frozen=True)
class ReviewClipExportConfig:
    """Configuration for exporting clip files and optional G1 preview videos."""

    limit: int = 8
    source_tar_name: str = "soma_proportional.tar"
    g1_tar_name: str = "g1.tar"
    source_path_column: str = "move_soma_proportional_path"
    g1_path_column: str = "move_g1_path"
    render_g1: bool = False
    render_max_frames: int = 120
    render_width: int = 640
    render_height: int = 360
    fps: float = 120.0
    root_position_scale: float = 0.01
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
    exported: list[dict[str, object]] = []
    renderer = _G1Renderer(config) if config.render_g1 else None

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
                source_bytes=source_bytes,
                g1_bytes=g1_bytes,
                render_report=render_report,
            )
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            exported.append(metadata)

    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    readme_md = output_dir / "README.md"
    _write_summary_csv(summary_csv, exported)
    _write_summary_json(summary_json, output_dir, input_csv, exported, render_counts, config)
    _write_readme(readme_md, input_csv, exported, render_counts, config)
    return ReviewClipExportResult(
        output_dir=output_dir,
        summary_csv=summary_csv,
        summary_json=summary_json,
        readme_md=readme_md,
        exported_rows=len(exported),
        render_counts=dict(sorted(render_counts.items())),
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
    source_bytes: int,
    g1_bytes: int,
    render_report: Mapping[str, object],
) -> dict[str, object]:
    metadata = {
        "label": label,
        "sample_index": sample_index,
        "source_archive": str(source_tar_path),
        "target_archive": str(g1_tar_path),
        "source_bvh": str(source_path),
        "target_g1_csv": str(g1_path),
        "target_g1_video": str(video_path) if video_path is not None else "",
        "source_bvh_bytes": source_bytes,
        "target_g1_csv_bytes": g1_bytes,
        "render": dict(render_report),
        "render_note": (
            "G1 target CSV visualization only; not learned retargeter output or Isaac Lab evaluation."
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
        "target_g1_csv",
        "target_g1_video",
        "render_status",
        "render_message",
        *QUALITY_METRIC_COLUMNS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            render = row.get("render") if isinstance(row.get("render"), Mapping) else {}
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fieldnames},
                    "render_status": render.get("status", ""),
                    "render_message": render.get("message", ""),
                }
            )


def _write_summary_json(
    path: Path,
    output_dir: Path,
    input_csv: Path,
    rows: Sequence[Mapping[str, object]],
    render_counts: Mapping[str, int],
    config: ReviewClipExportConfig,
) -> None:
    payload = {
        "output_root": str(output_dir),
        "input_csv": str(input_csv),
        "sample_count": len(rows),
        "render_status": dict(sorted(render_counts.items())),
        "summary_csv": str(output_dir / "summary.csv"),
        "config": config.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_readme(
    path: Path,
    input_csv: Path,
    rows: Sequence[Mapping[str, object]],
    render_counts: Mapping[str, int],
    config: ReviewClipExportConfig,
) -> None:
    text = f"""# Review Clips

Input CSV: `{input_csv}`

Exported rows: {len(rows)}
Render status: {dict(sorted(render_counts.items()))}

Each sample directory contains:

- `source_soma_proportional.bvh`
- `target_g1.csv`
- `target_g1_mujoco.mp4` when `render_g1=true` and rendering succeeds
- `metadata.json`

These videos visualize the paired G1 target CSV only. They are not learned retargeter predictions and not Isaac Lab rollouts.
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
    headers = ("index", "review_family", "filename", "action", "render", "metrics", "video")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        render = row.get("render") if isinstance(row.get("render"), Mapping) else {}
        cells = (
            str(row.get("sample_index", "")),
            str(row.get("review_family", "")),
            str(row.get("filename", "")),
            str(row.get("quality_action", "")),
            str(render.get("status", "")),
            _metric_summary(row),
            str(row.get("target_g1_video", "")),
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
    if config.render_max_frames <= 0:
        raise ValueError("render_max_frames must be positive")
    if config.render_width <= 0 or config.render_height <= 0:
        raise ValueError("render dimensions must be positive")
    if config.root_position_scale <= 0:
        raise ValueError("root_position_scale must be positive")
    if config.angle_scale <= 0:
        raise ValueError("angle_scale must be positive")
    if config.render_g1 and config.model_xml is None:
        raise ValueError("render_g1 requires model_xml")


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
            if index >= config.render_max_frames:
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
