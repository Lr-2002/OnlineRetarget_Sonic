"""Export full-length BONES-SONIC 3D capsule review videos."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from .bones_sonic import SONIC_BODY_NAMES, SONIC_ORDER_NOTE
from .review_clips import ReviewClipExportConfig, _render_capsule_3d_video


SONIC_REVIEW_FAMILIES = (
    "sonic_ground_penetration",
    "sonic_joint_velocity_jump",
    "sonic_joint_position_jump",
    "sonic_unstable_start_end",
)

SONIC_PRUNED_BODY_NAMES = (
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
)

SONIC_PRUNED_CAPSULE_EDGES = (
    ("pelvis", "waist_yaw_link"),
    ("waist_yaw_link", "waist_roll_link"),
    ("waist_roll_link", "torso_link"),
    ("pelvis", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
)

SONIC_REVIEW_METRICS = (
    "quality_action",
    "quality_flags",
    "fps",
    "frame_count",
    "max_abs_joint_velocity",
    "max_abs_joint_step_velocity",
    "max_start_end_root_speed",
    "penetration_depth",
    "joint_limit_violation_rate",
    "self_collision_proxy_rate",
    "contact_frame_ratio",
    "mean_foot_clearance",
)


@dataclass(frozen=True)
class SonicReviewClipExportConfig:
    stats_jsonl: Path
    output_root: Path
    run_name: str = "sonic_review_clips_v0"
    max_per_family: int = 1
    keep_examples: int = 2
    render_max_frames: int = 0
    render_width: int = 640
    render_height: int = 360
    fps: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["stats_jsonl"] = str(self.stats_jsonl)
        payload["output_root"] = str(self.output_root)
        return payload


@dataclass(frozen=True)
class SonicReviewClipExportResult:
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


def export_sonic_review_clips(
    config: SonicReviewClipExportConfig,
) -> SonicReviewClipExportResult:
    """Render selected SONIC NPZ clips as full-length 3D capsules."""

    np = _require_numpy()
    _validate_config(config)
    rows = _select_review_rows(_read_jsonl(config.stats_jsonl), config)
    output_dir = config.output_root.expanduser() / "review_clips" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    render_counts: Counter[str] = Counter()
    summary_rows: list[dict[str, object]] = []
    edges = _sonic_capsule_edges()
    for index, row in enumerate(rows):
        clip_dir = output_dir / _clip_dir_name(index, row)
        clip_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = clip_dir / "metadata.json"
        video_path = clip_dir / "sonic_g1_3d_capsules.mp4"
        render_report = _render_sonic_npz(row, video_path, config, edges, np)
        render_counts[str(render_report.get("status", "unknown"))] += 1
        metadata = {
            "row": row,
            "render": render_report,
            "source": "BONES-SONIC NPZ body_pos_w",
            "note": (
                "Full-length pruned 3D capsule visualization from SONIC body_pos_w. "
                "Wrist/hand and head/face-style distal markers are excluded to avoid noisy links. "
                "This is not BVH FK, not legacy G1 CSV, and not an Isaac Lab rollout. "
                + SONIC_ORDER_NOTE
            ),
        }
        _write_json(metadata_path, metadata)
        summary_rows.append(
            {
                "index": index,
                "review_family": row.get("review_family", ""),
                "filename": row.get("filename", ""),
                "actor_uid": row.get("actor_uid", ""),
                "category": row.get("category", ""),
                "quality_action": row.get("quality_action", ""),
                "quality_flags": row.get("quality_flags", ""),
                "sonic_relative_path": row.get("sonic_relative_path", ""),
                "sonic_path": row.get("sonic_path", ""),
                "video_path": str(video_path),
                "metadata_path": str(metadata_path),
                "render_status": render_report.get("status", ""),
                "render_message": render_report.get("message", ""),
                "render_frames": render_report.get("frames", ""),
                "render_fps": render_report.get("fps", ""),
                "changed_frames": render_report.get("changed_frames", ""),
                **{metric: row.get(metric, "") for metric in SONIC_REVIEW_METRICS},
            }
        )

    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    readme_md = output_dir / "README.md"
    _write_csv(summary_csv, summary_rows)
    _write_json(
        summary_json,
        {
            "config": config.to_dict(),
            "output_dir": str(output_dir),
            "summary_csv": str(summary_csv),
            "render_counts": dict(sorted(render_counts.items())),
            "exported_rows": len(summary_rows),
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
        },
    )
    readme_md.write_text(_readme(config, summary_rows, render_counts), encoding="utf-8")
    return SonicReviewClipExportResult(
        output_dir=output_dir,
        summary_csv=summary_csv,
        summary_json=summary_json,
        readme_md=readme_md,
        exported_rows=len(summary_rows),
        render_counts=dict(sorted(render_counts.items())),
    )


def _render_sonic_npz(
    row: Mapping[str, object],
    video_path: Path,
    config: SonicReviewClipExportConfig,
    edges: Sequence[tuple[str, str]],
    np: Any,
) -> dict[str, object]:
    sonic_path = Path(str(row.get("sonic_path", "")))
    try:
        with np.load(sonic_path) as data:
            body_pos = np.asarray(data["body_pos_w"], dtype=float)
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    except Exception as exc:
        return {"status": "failed", "message": f"Could not load SONIC NPZ: {exc}"}
    frames = _body_pos_frames(body_pos, config.render_max_frames)
    render_config = ReviewClipExportConfig(
        render_max_frames=config.render_max_frames,
        render_width=config.render_width,
        render_height=config.render_height,
        fps=config.fps or fps,
    )
    return _render_capsule_3d_video(
        frames=frames,
        edges=edges,
        video_path=video_path,
        config=render_config,
        label="sonic pruned body_pos_w capsules",
        up_axis=2,
        capsule_color=(61, 107, 160),
        key_color=(139, 91, 41),
    )


def _body_pos_frames(body_pos: Any, max_frames: int) -> list[dict[str, tuple[float, float, float]]]:
    if max_frames > 0:
        body_pos = body_pos[:max_frames]
    selected_indices = [
        (index, name)
        for index, name in enumerate(SONIC_BODY_NAMES)
        if name in set(SONIC_PRUNED_BODY_NAMES)
    ]
    frames: list[dict[str, tuple[float, float, float]]] = []
    for frame in body_pos:
        frames.append(
            {
                name: (
                    float(frame[index, 0]),
                    float(frame[index, 1]),
                    float(frame[index, 2]),
                )
                for index, name in selected_indices
            }
        )
    return frames


def _sonic_capsule_edges() -> tuple[tuple[str, str], ...]:
    available = set(SONIC_BODY_NAMES)
    return tuple(
        (start, end)
        for start, end in SONIC_PRUNED_CAPSULE_EDGES
        if start in available and end in available
    )


def _select_review_rows(
    rows: Sequence[dict[str, object]],
    config: SonicReviewClipExportConfig,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    used_paths: set[str] = set()
    for family in SONIC_REVIEW_FAMILIES:
        family_rows = [
            row
            for row in rows
            if family in str(row.get("quality_flags", "")).split("|")
        ]
        for row in sorted(family_rows, key=lambda item: _family_review_rank(family, item)):
            path = str(row.get("sonic_path", ""))
            if not path or path in used_paths:
                continue
            payload = dict(row)
            payload["review_family"] = family
            selected.append(payload)
            used_paths.add(path)
            if sum(1 for item in selected if item.get("review_family") == family) >= config.max_per_family:
                break
    keep_rows = [row for row in rows if row.get("quality_action") == "keep"]
    for row in sorted(keep_rows, key=_keep_review_rank):
        if sum(1 for item in selected if item.get("review_family") == "sonic_keep") >= config.keep_examples:
            break
        path = str(row.get("sonic_path", ""))
        if not path or path in used_paths:
            continue
        payload = dict(row)
        payload["review_family"] = "sonic_keep"
        selected.append(payload)
        used_paths.add(path)
    return selected


def _family_review_rank(family: str, row: Mapping[str, object]) -> tuple[float, float, str]:
    # Keep review clips representative and quick enough while preserving full length.
    # The first term prefers clips near 10 seconds at SONIC's 50 Hz.
    return (
        abs(_float(row.get("original_frame_count") or row.get("frame_count")) - 500.0),
        -_rank_metric(family, row),
        str(row.get("sonic_relative_path", "")),
    )


def _rank_metric(family: str, row: Mapping[str, object]) -> float:
    metric_by_family = {
        "sonic_ground_penetration": "penetration_depth",
        "sonic_joint_velocity_jump": "max_abs_joint_velocity",
        "sonic_joint_position_jump": "max_abs_joint_step_velocity",
        "sonic_unstable_start_end": "max_start_end_root_speed",
    }
    return _float(row.get(metric_by_family.get(family, "frame_count")))


def _keep_review_rank(row: Mapping[str, object]) -> tuple[float, str]:
    # Prefer a representative full-length keep clip without selecting extreme idle loops.
    return (abs(_float(row.get("frame_count")) - 360.0), str(row.get("sonic_relative_path", "")))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _readme(
    config: SonicReviewClipExportConfig,
    rows: Sequence[Mapping[str, object]],
    render_counts: Mapping[str, int],
) -> str:
    lines = [
        "# SONIC 3D Capsule Review Clips",
        "",
        "These videos are rendered directly from BONES-SONIC NPZ `body_pos_w` arrays.",
        "",
        "The capsule graph is pruned: wrist/hand and head/face-style distal markers are excluded.",
        "",
        "They are not BVH FK previews, not legacy G1 CSV previews, and not Isaac Lab rollouts.",
        "",
        SONIC_ORDER_NOTE,
        "",
        f"- stats JSONL: `{config.stats_jsonl}`",
        f"- render max frames: `{config.render_max_frames}` (`0` means full length)",
        f"- render counts: `{dict(sorted(render_counts.items()))}`",
        "",
        "| Family | Filename | Action | Frames | Metrics | Video |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        metrics = "; ".join(
            f"{metric}={row.get(metric)}"
            for metric in SONIC_REVIEW_METRICS
            if metric not in {"quality_action", "quality_flags"} and row.get(metric) not in (None, "")
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(row.get("review_family", "")),
                    _md_cell(row.get("filename", "")),
                    _md_cell(row.get("quality_action", "")),
                    _md_cell(row.get("render_frames", "")),
                    _md_cell(metrics),
                    _md_cell(row.get("video_path", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _clip_dir_name(index: int, row: Mapping[str, object]) -> str:
    return f"{index:02d}_{_safe_name(str(row.get('review_family', 'sonic')))}_{_safe_name(str(row.get('filename', 'clip')))}"


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value).strip("._")
    return safe[:120] or "clip"


def _validate_config(config: SonicReviewClipExportConfig) -> None:
    if config.max_per_family < 0:
        raise ValueError("max_per_family must be non-negative")
    if config.keep_examples < 0:
        raise ValueError("keep_examples must be non-negative")
    if config.render_max_frames < 0:
        raise ValueError("render_max_frames must be non-negative; use 0 for full length")
    if config.render_width <= 0 or config.render_height <= 0:
        raise ValueError("render dimensions must be positive")
    if config.fps is not None and config.fps <= 0:
        raise ValueError("fps must be positive when set")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for SONIC review video export")


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SONIC review rendering requires numpy. Use the project conda environment or another "
            "Python environment with numpy installed."
        ) from exc
    return np


def _md_cell(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(result.strip())
    except Exception:
        return False
