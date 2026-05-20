#!/usr/bin/env python3
"""Export orientation-debug capsule videos for SOMA BVH and BONES-SONIC G1 pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_retarget.data.bones_sonic import SONIC_BODY_NAMES
from online_retarget.data.review_clips import (  # noqa: PLC2701
    ReviewClipExportConfig,
    _capsule_camera,
    _capsule_scene_bounds,
    _cross3,
    _draw_capsule_3d_frame,
    _draw_circle,
    _draw_line,
    _draw_text,
    _normalize3,
    _project_camera,
    _source_capsule_body_names,
    _source_capsule_edges_from_motion,
    _sub3,
    _z_up_frames,
)
from online_retarget.data.sonic_review_clips import (
    SONIC_PRUNED_BODY_NAMES,
    SONIC_PRUNED_CAPSULE_EDGES,
)
from online_retarget.data.windowed_builder import (
    global_body_position_maps_from_bvh,
    parse_bvh_motion,
)


DEFAULT_FILENAMES = (
    "reach_jump_R_001__A419",
    "jump_over_obstacle_1_5m_L_001__A415",
    "point_laugh_head_arm_high_R_001__A550",
    "neutral_itching_head_R_001__A542",
    "drinking_bottle_throw_270_R_001__A550",
)

AXIS_COLORS = {
    "+x": (214, 57, 57),
    "+y": (55, 137, 72),
    "+z": (56, 90, 190),
}
FRONT_COLOR = (205, 91, 25)
LEFT_COLOR = (126, 53, 160)
RIGHT_COLOR = (20, 116, 150)
TARGET_DEBUG_BODY_NAMES = tuple(
    dict.fromkeys(
        list(SONIC_PRUNED_BODY_NAMES)
        + [
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
    )
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("runs/indices/seed_clean_pair_mapping_v0/seed_clean_pair_mapping.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/vis_pair_check"))
    parser.add_argument("--run-name", default="seed_orientation_debug_vis_v0")
    parser.add_argument("--filename", action="append", default=[])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--source-position-scale", type=float, default=0.01)
    args = parser.parse_args()

    np = _require_numpy()
    rows = _select_rows(
        _read_mapping(args.mapping_csv),
        args.filename or list(DEFAULT_FILENAMES),
        args.limit,
    )
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        clip_dir = output_dir / f"{index:02d}_{_safe_name(row['filename'])}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        source_video = clip_dir / "source_soma_bvh_orientation_debug.mp4"
        target_video = clip_dir / "target_g1_sonic_orientation_debug.mp4"

        source_payload = _source_payload(Path(row["clean_bvh_path"]), args.source_position_scale)
        source_report = _render_orientation_video(
            frames=source_payload["frames"],
            edges=source_payload["edges"],
            video_path=source_video,
            fps=_bvh_fps(Path(row["clean_bvh_path"])),
            width=args.width,
            height=args.height,
            label=f"source soma bvh orientation {row['filename']}",
            body_kind="source",
            capsule_color=(48, 132, 83),
            key_color=(132, 103, 34),
        )

        target_payload = _target_payload(Path(row["bones_sonic_path"]), np=np)
        target_report = _render_orientation_video(
            frames=target_payload["frames"],
            edges=target_payload["edges"],
            video_path=target_video,
            fps=target_payload["fps"],
            width=args.width,
            height=args.height,
            label=f"target g1 sonic orientation {row['filename']}",
            body_kind="target",
            capsule_color=(61, 107, 160),
            key_color=(139, 91, 41),
        )

        metadata = {
            "filename": row["filename"],
            "actor_uid": row.get("actor_uid", ""),
            "content_technical_description": row.get("content_technical_description", ""),
            "source_bvh": row["clean_bvh_path"],
            "target_g1_sonic_npz": row["bones_sonic_path"],
            "source_orientation_debug_video": str(source_video),
            "target_orientation_debug_video": str(target_video),
            "source_report": source_report,
            "target_report": target_report,
            "overlay_semantics": _overlay_semantics(),
        }
        metadata_path = clip_dir / "orientation_debug_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_rows.append(
            {
                "filename": row["filename"],
                "source_video": str(source_video),
                "target_video": str(target_video),
                "source_status": source_report["status"],
                "target_status": target_report["status"],
                "source_frames": source_report.get("frames", ""),
                "target_frames": target_report.get("frames", ""),
                "source_fps": source_report.get("fps", ""),
                "target_fps": target_report.get("fps", ""),
                "metadata": str(metadata_path),
            }
        )

    _write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "mapping_csv": str(args.mapping_csv),
                "output_dir": str(output_dir),
                "overlay_semantics": _overlay_semantics(),
                "rows": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_readme(summary_rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": summary_rows}, indent=2, sort_keys=True))


def _source_payload(bvh_path: Path, position_scale: float) -> dict[str, object]:
    text = bvh_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    motion = parse_bvh_motion(text)
    body_names = _source_capsule_body_names(motion)
    frames = _z_up_frames(
        global_body_position_maps_from_bvh(motion, body_names=body_names, position_scale=position_scale),
        up_axis=1,
    )
    edges = _source_capsule_edges_from_motion(motion, body_names)
    return {"frames": frames, "edges": edges}


def _target_payload(npz_path: Path, *, np: Any) -> dict[str, object]:
    with np.load(npz_path) as data:
        body_pos = np.asarray(data["body_pos_w"], dtype=float)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    selected = [
        (index, name)
        for index, name in enumerate(SONIC_BODY_NAMES)
        if name in set(TARGET_DEBUG_BODY_NAMES)
    ]
    frames = [
        {
            name: (float(frame[index, 0]), float(frame[index, 1]), float(frame[index, 2]))
            for index, name in selected
        }
        for frame in body_pos
    ]
    edges = tuple((start, end) for start, end in SONIC_PRUNED_CAPSULE_EDGES if start in set(SONIC_BODY_NAMES))
    return {"frames": _z_up_frames(frames, up_axis=2), "edges": edges, "fps": fps}


def _render_orientation_video(
    *,
    frames: Sequence[Mapping[str, tuple[float, float, float]]],
    edges: Sequence[tuple[str, str]],
    video_path: Path,
    fps: float,
    width: int,
    height: int,
    label: str,
    body_kind: str,
    capsule_color: tuple[int, int, int],
    key_color: tuple[int, int, int],
) -> dict[str, object]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"status": "blocked", "message": "ffmpeg is required."}
    if not frames:
        return {"status": "blocked", "message": "no frames available."}

    video_path.parent.mkdir(parents=True, exist_ok=True)
    config = ReviewClipExportConfig(render_width=width, render_height=height, fps=fps, render_max_frames=0)
    bounds = _capsule_scene_bounds(frames)
    camera = _capsule_camera(bounds, width, height)
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
        str(int(round(fps))),
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
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        return {"status": "failed", "message": "ffmpeg stdin was unavailable."}
    frame_sums: list[int] = []
    changed_frames = 0
    previous_frame: bytes | None = None
    try:
        for frame_index, frame in enumerate(frames):
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
            _draw_orientation_overlay(image, width, height, frame, bounds, camera, body_kind)
            frame_bytes = bytes(image)
            process.stdin.write(frame_bytes)
            frame_sums.append(sum(frame_bytes))
            if previous_frame is not None and previous_frame != frame_bytes:
                changed_frames += 1
            previous_frame = frame_bytes
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.stderr is not None:
            process.stderr.close()
        return_code = process.wait()
    except Exception as exc:
        if process.poll() is None:
            process.kill()
        return {"status": "failed", "message": f"render failed: {exc}"}
    if return_code != 0:
        return {"status": "failed", "message": "ffmpeg failed", "ffmpeg_tail": stderr[-800:]}
    return {
        "status": "ok",
        "frames": len(frames),
        "fps": int(round(fps)),
        "changed_frames": changed_frames,
        "frame_sum_min": min(frame_sums),
        "frame_sum_max": max(frame_sums),
        "video_path": str(video_path),
        "overlay": _overlay_semantics(),
    }


def _draw_orientation_overlay(
    image: bytearray,
    width: int,
    height: int,
    frame: Mapping[str, tuple[float, float, float]],
    bounds: Mapping[str, float],
    camera: Mapping[str, object],
    body_kind: str,
) -> None:
    _draw_world_axes(image, width, height, frame, bounds, camera)
    _draw_body_front(image, width, height, frame, camera, body_kind)
    _draw_left_right_labels(image, width, height, frame, camera, body_kind)
    _draw_overlay_legend(image, width, height, camera)


def _draw_world_axes(
    image: bytearray,
    width: int,
    height: int,
    frame: Mapping[str, tuple[float, float, float]],
    bounds: Mapping[str, float],
    camera: Mapping[str, object],
) -> None:
    origin = _body_origin(frame, fallback=(float(bounds["center_x"]), float(bounds["center_y"]), 0.0))
    axis_len = max(float(bounds["radius"]) * 0.35, 0.35)
    axes = (
        ("+x", (axis_len, 0.0, 0.0)),
        ("+y", (0.0, axis_len, 0.0)),
        ("+z", (0.0, 0.0, axis_len)),
    )
    for label, delta in axes:
        start = _project_camera(origin, camera, width, height)
        end_point = (origin[0] + delta[0], origin[1] + delta[1], origin[2] + delta[2])
        end = _project_camera(end_point, camera, width, height)
        if start is None or end is None:
            continue
        _draw_arrow_2d(image, width, height, (start[0], start[1]), (end[0], end[1]), AXIS_COLORS[label], radius=3)
        _draw_debug_text(image, width, height, end[0] + 5, end[1] + 5, label.upper(), AXIS_COLORS[label])


def _draw_body_front(
    image: bytearray,
    width: int,
    height: int,
    frame: Mapping[str, tuple[float, float, float]],
    camera: Mapping[str, object],
    body_kind: str,
) -> None:
    origin = _body_origin(frame)
    forward = _body_forward(frame, body_kind)
    if origin is None or forward is None:
        return
    arrow_len = 0.55
    end_point = (
        origin[0] + forward[0] * arrow_len,
        origin[1] + forward[1] * arrow_len,
        origin[2] + forward[2] * arrow_len,
    )
    start = _project_camera(origin, camera, width, height)
    end = _project_camera(end_point, camera, width, height)
    if start is None or end is None:
        return
    _draw_arrow_2d(image, width, height, (start[0], start[1]), (end[0], end[1]), FRONT_COLOR, radius=5)
    _draw_debug_text(image, width, height, end[0] + 6, end[1] - 12, "FRONT", FRONT_COLOR)


def _draw_left_right_labels(
    image: bytearray,
    width: int,
    height: int,
    frame: Mapping[str, tuple[float, float, float]],
    camera: Mapping[str, object],
    body_kind: str,
) -> None:
    if body_kind == "source":
        left_name, right_name = "LeftHand", "RightHand"
    else:
        left_name, right_name = "left_wrist_yaw_link", "right_wrist_yaw_link"
    for label, name, color in (("l", left_name, LEFT_COLOR), ("r", right_name, RIGHT_COLOR)):
        point = frame.get(name)
        if point is None:
            continue
        projected = _project_camera(point, camera, width, height)
        if projected is None:
            continue
        _draw_circle(image, width, height, (projected[0], projected[1]), radius=12, color=(32, 38, 38))
        _draw_circle(image, width, height, (projected[0], projected[1]), radius=10, color=color)
        _draw_debug_text(image, width, height, projected[0] - 6, projected[1] + 5, label.upper(), (255, 255, 255))


def _draw_overlay_legend(
    image: bytearray,
    width: int,
    height: int,
    camera: Mapping[str, object],
) -> None:
    lines = [
        "red +x green +y blue +z",
        "orange front body facing",
        "purple l cyan r",
    ]
    y = height - 54
    for line in lines:
        _draw_debug_text(image, width, height, 14, y, line, (45, 54, 52), scale=0.45, thickness=1)
        y += 14
    forward = camera.get("forward")
    if isinstance(forward, tuple):
        _draw_debug_text(
            image,
            width,
            height,
            width - 250,
            height - 22,
            f"camera view {forward[0]:+.2f} {forward[1]:+.2f} {forward[2]:+.2f}",
            (75, 82, 78),
            scale=0.42,
            thickness=1,
        )


def _draw_debug_text(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    *,
    scale: float = 0.55,
    thickness: int = 2,
) -> None:
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        _draw_text(image, width, height, x, y, text, color)
        return
    arr = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 3))
    origin = (max(0, min(width - 1, x)), max(0, min(height - 1, y)))
    cv2.putText(arr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 30, 30), thickness + 2, cv2.LINE_AA)
    cv2.putText(arr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_arrow_2d(
    image: bytearray,
    width: int,
    height: int,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    radius: int,
) -> None:
    _draw_line(image, width, height, start, end, radius=radius, color=(35, 43, 43))
    _draw_line(image, width, height, start, end, radius=max(1, radius - 1), color=color)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head_len = 16
    for sign in (-1.0, 1.0):
        tip = end
        tail = (
            int(round(end[0] - ux * head_len + px * sign * head_len * 0.5)),
            int(round(end[1] - uy * head_len + py * sign * head_len * 0.5)),
        )
        _draw_line(image, width, height, tip, tail, radius=max(1, radius - 1), color=color)


def _body_origin(
    frame: Mapping[str, tuple[float, float, float]],
    fallback: tuple[float, float, float] | None = None,
) -> tuple[float, float, float] | None:
    for name in ("Hips", "pelvis"):
        if name in frame:
            return frame[name]
    return fallback


def _body_forward(
    frame: Mapping[str, tuple[float, float, float]],
    body_kind: str,
) -> tuple[float, float, float] | None:
    if body_kind == "source":
        left = frame.get("LeftShoulder")
        right = frame.get("RightShoulder")
        upper = frame.get("Chest")
        lower = frame.get("Hips")
    else:
        left = frame.get("left_shoulder_pitch_link")
        right = frame.get("right_shoulder_pitch_link")
        upper = frame.get("torso_link")
        lower = frame.get("pelvis")
    if left is None or right is None or upper is None or lower is None:
        return None
    left_axis = _normalize3(_sub3(left, right))
    up_axis = _normalize3(_sub3(upper, lower))
    forward = _normalize3(_cross3(left_axis, up_axis))
    if forward == (0.0, 0.0, 0.0):
        return None
    return forward


def _overlay_semantics() -> dict[str, str]:
    return {
        "+X": "red world +X after conversion to z-up render coordinates",
        "+Y": "green world +Y after conversion to z-up render coordinates",
        "+Z": "blue world +Z/up in render coordinates",
        "front": "orange body facing vector computed as cross(left_shoulder - right_shoulder, torso_up)",
        "L": "purple left hand/wrist label",
        "R": "cyan right hand/wrist label",
        "camera": "software perspective camera direction shown in bottom-right text; front/back appearance depends on this camera",
    }


def _read_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_rows(rows: Sequence[dict[str, str]], names: Sequence[str], limit: int) -> list[dict[str, str]]:
    by_name = {row["filename"]: row for row in rows}
    selected = [by_name[name] for name in names if name in by_name]
    if len(selected) < limit:
        for row in rows:
            if len(selected) >= limit:
                break
            if row in selected:
                continue
            if (
                row.get("merged_quality_action") == "keep"
                and row.get("curation_action") == "keep"
                and row.get("clean_bvh_exists") == "True"
                and row.get("bones_sonic_exists") == "True"
            ):
                selected.append(row)
    return selected[:limit]


def _bvh_fps(path: Path) -> float:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip().startswith("Frame Time:"):
                frame_time = float(line.split(":", 1)[1].strip())
                return 1.0 / frame_time
    return 120.0


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _readme(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Orientation Debug Visuals",
        "",
        "这些视频在 source SOMA BVH capsule 和 target G1 SONIC capsule 上叠加世界坐标系、body front、左右手标签。",
        "",
        "- red `+X`, green `+Y`, blue `+Z`: z-up render coordinates.",
        "- orange `front`: `cross(left_shoulder - right_shoulder, torso_up)`.",
        "- purple `L`, cyan `R`: left/right hand or wrist labels.",
        "",
        "| filename | source | target | metadata |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('filename', '')} | {row.get('source_video', '')} | "
            f"{row.get('target_video', '')} | {row.get('metadata', '')} |"
        )
    return "\n".join(lines) + "\n"


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("._")
    return safe[:120] or "clip"


def _require_numpy() -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError("numpy is required to read BONES-SONIC NPZ files.") from exc
    return np


if __name__ == "__main__":
    main()
