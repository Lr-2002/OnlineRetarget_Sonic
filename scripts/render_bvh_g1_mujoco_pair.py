#!/usr/bin/env python3
"""Render a source BVH and a target G1 motion as a reusable pair video.

The source panel uses the existing BVH capsule renderer.  The target panel uses
MuJoCo kinematic playback of the G1 MJCF and preserves target root XY motion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from online_retarget.data.bones_seed import G1_JOINT_COLUMNS  # noqa: E402
from online_retarget.data.review_clips import (  # noqa: E402
    ReviewClipExportConfig,
    _capsule_scene_bounds,
    _draw_capsule_3d_frame,
    _euler_xyz_to_quat_wxyz,
    _render_capsule_3d_video,
    _source_capsule_body_names,
    _source_capsule_edges_from_motion,
    _z_up_frames,
)
from online_retarget.web_pipeline import (  # noqa: E402
    DEFAULT_ROOT_HEIGHT,
    _normalize_quat_wxyz,
)
from online_retarget.web_pipeline import global_body_position_maps_from_bvh, parse_bvh_motion  # noqa: E402


SONIC_JOINT_NAMES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, required=True, help="Source BVH motion.")
    parser.add_argument(
        "--g1-motion",
        type=Path,
        required=True,
        help="Target G1 motion: SONIC motionlib .pkl, BONES/SONIC .npz, or G1 .csv.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output side-by-side MP4 path.")
    parser.add_argument("--model-xml", type=Path, default=None, help="G1 MJCF XML for MuJoCo rendering.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config to read model_xml/defaults from.")
    parser.add_argument("--format", choices=("auto", "motionlib", "npz", "csv"), default="auto")
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--source-position-scale", type=float, default=None)
    parser.add_argument("--root-position-scale", type=float, default=0.01, help="CSV root position scale only.")
    parser.add_argument("--angle-scale", type=float, default=math.pi / 180.0, help="CSV angle scale only.")
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Playback FPS override. Use this for generated CSVs because they do not store timing.",
    )
    parser.add_argument(
        "--root-rot-format",
        choices=("auto", "wxyz", "xyzw"),
        default="auto",
        help="Quaternion order for root_rot arrays. Auto treats motionlib/root_rot as xyzw and *_quat fields as wxyz.",
    )
    parser.add_argument(
        "--preserve-world-root",
        action="store_true",
        help="Do not subtract the first target root XY before rendering.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("trajectory", "follow", "fixed"),
        default="trajectory",
        help="MuJoCo camera behavior for the target panel.",
    )
    parser.add_argument("--render-frames", action="store_true", help="Ask MuJoCo to draw body frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_json(args.config) if args.config is not None else {}
    visual_cfg = config.get("visual_validation", {}) if isinstance(config.get("visual_validation", {}), Mapping) else {}
    model_xml = args.model_xml or _optional_path(visual_cfg.get("g1_model_xml"))
    if model_xml is None or not model_xml.exists():
        raise SystemExit(f"--model-xml is required and must exist: {model_xml}")

    source_scale = (
        float(args.source_position_scale)
        if args.source_position_scale is not None
        else float(visual_cfg.get("source_position_scale", 0.01))
    )

    g1_motion = load_g1_motion(
        args.g1_motion,
        fmt=args.format,
        max_frames=args.max_frames,
        duration_sec=args.duration_sec,
        root_position_scale=args.root_position_scale,
        angle_scale=args.angle_scale,
        root_rot_format=args.root_rot_format,
        target_fps=args.target_fps,
    )
    if not args.preserve_world_root:
        g1_motion = zero_initial_root_xy(g1_motion)

    frame_count = g1_motion["frame_count"]
    fps = float(g1_motion["fps"])
    panel_dir = args.output.with_suffix("")
    panel_dir.mkdir(parents=True, exist_ok=True)
    source_video = panel_dir / "source_bvh_capsules.mp4"
    target_video = panel_dir / "target_g1_mujoco.mp4"
    metadata_path = args.output.with_suffix(".json")

    source_report = render_source_bvh_panel(
        args.bvh,
        source_video,
        fps=fps,
        frame_count=frame_count,
        width=args.width,
        height=args.height,
        source_position_scale=source_scale,
    )
    target_report = render_g1_mujoco_panel(
        g1_motion,
        target_video,
        model_xml=model_xml,
        width=args.width,
        height=args.height,
        camera_mode=args.camera_mode,
        render_frames=args.render_frames,
    )
    combine_report = combine_two_videos((source_video, target_video), args.output, fps=int(round(fps)))

    metadata = {
        "status": "ok" if combine_report.get("status") == "ok" else "failed",
        "bvh": str(args.bvh),
        "g1_motion": str(args.g1_motion),
        "model_xml": str(model_xml),
        "output": str(args.output),
        "fps": fps,
        "target_fps_override": args.target_fps,
        "frames": frame_count,
        "duration_sec": frame_count / fps if fps > 0 else 0.0,
        "source_position_scale": source_scale,
        "root_xy_preserved": True,
        "initial_root_xy_zeroed": not args.preserve_world_root,
        "root_quat_format": g1_motion.get("root_quat_format"),
        "camera_mode": args.camera_mode,
        "source_xy_span_m": source_report.get("source_xy_span_m"),
        "target_root_xy_span_m": g1_motion["root_xy_span_m"],
        "source_render": source_report,
        "target_render": target_report,
        "combine": combine_report,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_path(value: object) -> Path | None:
    if value is None or str(value) == "":
        return None
    return Path(str(value))


def load_g1_motion(
    path: Path,
    *,
    fmt: str,
    max_frames: int,
    duration_sec: float,
    root_position_scale: float,
    angle_scale: float,
    root_rot_format: str,
    target_fps: float | None = None,
) -> dict[str, Any]:
    resolved = detect_g1_format(path, fmt)
    if resolved == "motionlib":
        motion = load_motionlib_g1(path, root_rot_format=root_rot_format)
    elif resolved == "npz":
        motion = load_npz_g1(path, root_rot_format=root_rot_format)
    elif resolved == "csv":
        motion = load_csv_g1(path, root_position_scale=root_position_scale, angle_scale=angle_scale)
    else:
        raise ValueError(f"unsupported G1 motion format: {resolved}")
    if target_fps is not None:
        motion = with_g1_motion_fps(motion, target_fps)
    return limit_g1_motion(motion, max_frames=max_frames, duration_sec=duration_sec)


def detect_g1_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower()
    if suffix in (".pkl", ".pickle"):
        return "motionlib"
    if suffix == ".npz":
        return "npz"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"cannot infer G1 motion format from suffix: {path}")


def load_motionlib_g1(path: Path, *, root_rot_format: str) -> dict[str, Any]:
    import joblib

    loaded = joblib.load(path)
    if not isinstance(loaded, Mapping) or not loaded:
        raise ValueError(f"motionlib file must contain a non-empty mapping: {path}")
    key = path.stem
    entry = loaded[key] if key in loaded else loaded[next(iter(loaded))]
    if not isinstance(entry, Mapping):
        raise ValueError(f"motionlib entry must be a mapping: {path}")
    joint_pos = np.asarray(entry["dof"], dtype=np.float32)
    root_quat, root_quat_format = root_rot_to_wxyz(
        np.asarray(entry["root_rot"], dtype=np.float32),
        requested_format=root_rot_format,
        auto_format="xyzw",
    )
    root_pos = np.asarray(
        entry.get("root_trans_offset", default_root_pos(joint_pos.shape[0])),
        dtype=np.float32,
    )
    fps = float(entry.get("fps") or 50.0)
    joint_names = tuple(str(name) for name in entry.get("joint_names", SONIC_JOINT_NAMES))
    return make_g1_motion(
        joint_pos=joint_pos,
        root_pos=root_pos,
        root_quat=root_quat,
        fps=fps,
        joint_names=joint_names,
        source_format="motionlib",
        root_quat_format=root_quat_format,
    )


def load_npz_g1(path: Path, *, root_rot_format: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0
        if "root_pos" in data:
            root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        elif "root_trans_offset" in data:
            root_pos = np.asarray(data["root_trans_offset"], dtype=np.float32)
        elif "body_pos_w" in data:
            root_pos = np.asarray(data["body_pos_w"][:, 0, :], dtype=np.float32)
        else:
            root_pos = default_root_pos(joint_pos.shape[0])
        if "root_quat" in data:
            root_quat = np.asarray(data["root_quat"], dtype=np.float32)
            root_quat_format = "wxyz"
        elif "root_rot" in data:
            root_quat, root_quat_format = root_rot_to_wxyz(
                np.asarray(data["root_rot"], dtype=np.float32),
                requested_format=root_rot_format,
                auto_format="xyzw",
            )
        elif "body_quat_w" in data:
            root_quat = np.asarray(data["body_quat_w"][:, 0, :], dtype=np.float32)
            root_quat_format = "wxyz"
        else:
            root_quat = identity_quat(joint_pos.shape[0])
            root_quat_format = "wxyz"
    return make_g1_motion(
        joint_pos=joint_pos,
        root_pos=root_pos,
        root_quat=normalize_quat_array(root_quat),
        fps=fps,
        joint_names=SONIC_JOINT_NAMES,
        source_format="npz",
        root_quat_format=root_quat_format,
    )


def load_csv_g1(path: Path, *, root_position_scale: float, angle_scale: float) -> dict[str, Any]:
    roots: list[list[float]] = []
    quats: list[list[float]] = []
    joints: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            root_euler = [
                float(row["root_rotateX"]) * angle_scale,
                float(row["root_rotateY"]) * angle_scale,
                float(row["root_rotateZ"]) * angle_scale,
            ]
            roots.append(
                [
                    float(row["root_translateX"]) * root_position_scale,
                    float(row["root_translateY"]) * root_position_scale,
                    float(row["root_translateZ"]) * root_position_scale,
                ]
            )
            quats.append(_euler_xyz_to_quat_wxyz(root_euler))
            joints.append([float(row[column]) * angle_scale for column in G1_JOINT_COLUMNS])
    joint_names = tuple(column[:-4] if column.endswith("_dof") else column for column in G1_JOINT_COLUMNS)
    return make_g1_motion(
        joint_pos=np.asarray(joints, dtype=np.float32),
        root_pos=np.asarray(roots, dtype=np.float32),
        root_quat=normalize_quat_array(np.asarray(quats, dtype=np.float32)),
        fps=50.0,
        joint_names=joint_names,
        source_format="csv",
        root_quat_format="wxyz",
    )


def make_g1_motion(
    *,
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    fps: float,
    joint_names: Sequence[str],
    source_format: str,
    root_quat_format: str,
) -> dict[str, Any]:
    target_len = min(joint_pos.shape[0], root_pos.shape[0], root_quat.shape[0])
    if target_len <= 0:
        raise ValueError("G1 motion has no frames")
    joint_pos = joint_pos[:target_len].astype(np.float32, copy=False)
    root_pos = root_pos[:target_len].astype(np.float32, copy=False)
    root_quat = root_quat[:target_len].astype(np.float32, copy=False)
    if len(joint_names) != joint_pos.shape[1]:
        raise ValueError(f"joint_names length {len(joint_names)} does not match joint_pos dim {joint_pos.shape[1]}")
    return {
        "joint_pos": joint_pos,
        "joint_vel": finite_difference_velocity(joint_pos, fps),
        "root_pos": root_pos,
        "root_quat": root_quat,
        "joint_names": tuple(joint_names),
        "fps": float(fps),
        "frame_count": int(target_len),
        "source_format": source_format,
        "root_quat_format": root_quat_format,
        "root_xy_span_m": root_xy_span(root_pos),
    }


def with_g1_motion_fps(motion: dict[str, Any], fps: float) -> dict[str, Any]:
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError(f"target_fps must be positive, got {fps}")
    updated = dict(motion)
    updated["fps"] = float(fps)
    updated["joint_vel"] = finite_difference_velocity(np.asarray(updated["joint_pos"]), float(fps))
    updated["fps_overridden"] = True
    return updated


def root_rot_to_wxyz(
    root_rot: np.ndarray,
    *,
    requested_format: str,
    auto_format: str,
) -> tuple[np.ndarray, str]:
    quat_format = auto_format if requested_format == "auto" else requested_format
    if quat_format == "xyzw":
        return normalize_quat_array(root_rot[..., [3, 0, 1, 2]]), quat_format
    if quat_format == "wxyz":
        return normalize_quat_array(root_rot), quat_format
    raise ValueError(f"unsupported root rotation format: {requested_format}")


def limit_g1_motion(motion: dict[str, Any], *, max_frames: int, duration_sec: float) -> dict[str, Any]:
    frame_limit = int(motion["frame_count"])
    if duration_sec > 0:
        frame_limit = min(frame_limit, max(1, int(round(float(motion["fps"]) * duration_sec))))
    if max_frames > 0:
        frame_limit = min(frame_limit, max_frames)
    limited = dict(motion)
    for key in ("joint_pos", "joint_vel", "root_pos", "root_quat"):
        limited[key] = np.asarray(motion[key])[:frame_limit]
    limited["frame_count"] = frame_limit
    limited["root_xy_span_m"] = root_xy_span(limited["root_pos"])
    return limited


def zero_initial_root_xy(motion: dict[str, Any]) -> dict[str, Any]:
    shifted = dict(motion)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float32).copy()
    if root_pos.shape[0] > 0:
        root_pos[:, :2] -= root_pos[0, :2]
    shifted["root_pos"] = root_pos
    shifted["root_xy_span_m"] = root_xy_span(root_pos)
    return shifted


def finite_difference_velocity(joint_pos: np.ndarray, fps: float) -> np.ndarray:
    if joint_pos.shape[0] <= 1:
        return np.zeros_like(joint_pos, dtype=np.float32)
    vel = np.zeros_like(joint_pos, dtype=np.float32)
    vel[1:] = (joint_pos[1:] - joint_pos[:-1]) * float(fps)
    vel[0] = vel[1]
    return vel.astype(np.float32, copy=False)


def default_root_pos(frame_count: int) -> np.ndarray:
    root = np.zeros((frame_count, 3), dtype=np.float32)
    root[:, 2] = DEFAULT_ROOT_HEIGHT
    return root


def identity_quat(frame_count: int) -> np.ndarray:
    quat = np.zeros((frame_count, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    return quat


def normalize_quat_array(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"quaternion array must have shape [T, 4], got {quat.shape}")
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    out = np.divide(quat, np.maximum(norm, 1e-8), out=np.zeros_like(quat), where=norm > 1e-8)
    bad = np.squeeze(norm <= 1e-8, axis=1)
    out[bad] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return out


def root_xy_span(root_pos: np.ndarray) -> float:
    if root_pos.shape[0] <= 1:
        return 0.0
    return float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))


def render_source_bvh_panel(
    bvh_path: Path,
    video_path: Path,
    *,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    source_position_scale: float,
) -> dict[str, object]:
    try:
        text = bvh_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
        motion = parse_bvh_motion(text)
        source_fps = 1.0 / float(motion.frame_time) if float(motion.frame_time) > 0 else fps
        body_names = _source_capsule_body_names(motion)
        edges = _source_capsule_edges_from_motion(motion, body_names)
        raw_frames = global_body_position_maps_from_bvh(
            motion,
            body_names=body_names,
            position_scale=source_position_scale,
        )
        frames = time_align_frames(raw_frames, source_fps=source_fps, target_fps=fps, frame_count=frame_count)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {"status": "failed", "message": f"Could not load BVH source motion: {exc}"}
    config = ReviewClipExportConfig(
        render_max_frames=frame_count,
        render_width=width,
        render_height=height,
        fps=fps,
        source_position_scale=source_position_scale,
    )
    report = _render_capsule_3d_video(
        frames=frames,
        edges=edges,
        video_path=video_path,
        config=config,
        label="source bvh",
        up_axis=1,
        capsule_color=(48, 132, 83),
        key_color=(132, 103, 34),
    )
    if report.get("status") == "blocked" and "ffmpeg" in str(report.get("message", "")).lower():
        report = render_capsule_3d_video_imageio(
            frames=frames,
            edges=edges,
            video_path=video_path,
            config=config,
            label="source bvh",
            up_axis=1,
            capsule_color=(48, 132, 83),
            key_color=(132, 103, 34),
        )
    report.update(
        {
            "source_fps": source_fps,
            "target_fps": fps,
            "source_frames": len(raw_frames),
            "aligned_frames": len(frames),
            "source_xy_span_m": source_xy_span_from_frames(frames, up_axis=1),
        }
    )
    return report


def render_capsule_3d_video_imageio(
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
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        return {"status": "blocked", "message": f"imageio is required when ffmpeg is unavailable: {exc}"}
    if not frames:
        return {"status": "blocked", "message": "No frames were available for 3D capsule rendering."}
    z_up_frames = _z_up_frames(frames, up_axis=up_axis)
    bounds = _capsule_scene_bounds(z_up_frames)
    width = int(config.render_width)
    height = int(config.render_height)
    fps = int(max(1, min(240, round(float(config.fps)))))
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame_sums: list[int] = []
    changed_frames = 0
    previous_frame: bytes | None = None
    try:
        with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
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
                array = np.frombuffer(bytes(image), dtype=np.uint8).reshape(height, width, 3)
                writer.append_data(array)
                frame_bytes = array.tobytes()
                frame_sums.append(int(array.sum()))
                if previous_frame is not None and frame_bytes != previous_frame:
                    changed_frames += 1
                previous_frame = frame_bytes
    except Exception as exc:
        return {"status": "failed", "message": f"imageio capsule render failed: {exc}"}
    if not video_path.exists() or video_path.stat().st_size == 0:
        return {"status": "failed", "message": "imageio capsule video was not written."}
    return {
        "status": "ok",
        "message": "Encoded 3D capsule review video with imageio.",
        "video_path": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frames": len(z_up_frames),
        "capsule_edges": len(edges),
        "changed_frames": changed_frames,
        "frame_sum_min": min(frame_sums),
        "frame_sum_max": max(frame_sums),
        "render_backend": "imageio_perspective_capsules",
        "up_axis": up_axis,
        "ground_z": 0.0,
    }


def time_align_frames(
    frames: Sequence[Mapping[str, tuple[float, float, float]]],
    *,
    source_fps: float,
    target_fps: float,
    frame_count: int,
) -> list[Mapping[str, tuple[float, float, float]]]:
    if not frames:
        return []
    aligned = []
    for index in range(frame_count):
        source_index = int(round((index / max(target_fps, 1e-6)) * source_fps))
        source_index = min(max(source_index, 0), len(frames) - 1)
        aligned.append(frames[source_index])
    return aligned


def source_xy_span_from_frames(
    frames: Sequence[Mapping[str, tuple[float, float, float]]],
    *,
    up_axis: int = 1,
) -> float:
    if len(frames) <= 1:
        return 0.0
    root_name = "Hips" if "Hips" in frames[0] else next(iter(frames[0]), "")
    if not root_name:
        return 0.0
    first = frames[0].get(root_name)
    last = frames[-1].get(root_name)
    if first is None or last is None:
        return 0.0
    horizontal_a, horizontal_b = (0, 2) if up_axis == 1 else (0, 1)
    return float(
        math.hypot(
            float(last[horizontal_a]) - float(first[horizontal_a]),
            float(last[horizontal_b]) - float(first[horizontal_b]),
        )
    )


def render_g1_mujoco_panel(
    motion: Mapping[str, Any],
    video_path: Path,
    *,
    model_xml: Path,
    width: int,
    height: int,
    camera_mode: str,
    render_frames: bool,
) -> dict[str, object]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco  # type: ignore[import-not-found]
    except Exception as exc:
        return {"status": "failed", "message": f"MuJoCo is unavailable: {exc}"}

    ffmpeg = shutil.which("ffmpeg")
    imageio_module = None
    encode_backend = "ffmpeg_rawvideo"
    if ffmpeg is None:
        try:
            import imageio.v2 as imageio_module  # type: ignore[import-not-found]
        except Exception as exc:
            return {"status": "blocked", "message": f"imageio is required when ffmpeg is unavailable: {exc}"}
        encode_backend = "imageio"

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    fps = int(max(1, min(240, round(float(motion["fps"])))))
    video_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = None
    process = None
    writer = None
    frame_sums: list[int] = []
    frame_stds: list[float] = []
    changed_frames = 0
    previous_frame: bytes | None = None
    missing_joints: set[str] = set()
    try:
        renderer = mujoco.Renderer(model, height=height, width=width)
        camera = make_mujoco_camera(mujoco, motion, mode=camera_mode)
        scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(scene_option)
        scene_option.frame = mujoco.mjtFrame.mjFRAME_BODY if render_frames else mujoco.mjtFrame.mjFRAME_NONE
        if ffmpeg is not None:
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
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            if process.stdin is None:
                return {"status": "failed", "message": "ffmpeg stdin was not available."}
        elif imageio_module is not None:
            writer = imageio_module.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
        root_quat = np.asarray(motion["root_quat"], dtype=np.float32)
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float32)
        joint_names = tuple(str(name) for name in motion["joint_names"])
        for index in range(int(motion["frame_count"])):
            missing_joints.update(
                set_mujoco_state_by_joint_names(
                    model,
                    data,
                    root=root_pos[index],
                    root_quat=root_quat[index],
                    joint_names=joint_names,
                    joint_values=joint_pos[index],
                )
            )
            mujoco.mj_forward(model, data)
            if camera_mode == "follow":
                follow_camera_to_pelvis(model, data, camera)
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            image = renderer.render()
            frame_bytes = image.tobytes()
            if process is not None and process.stdin is not None:
                process.stdin.write(frame_bytes)
            elif writer is not None:
                writer.append_data(image)
            frame_sums.append(int(image.sum()))
            frame_stds.append(float(image.std()))
            if previous_frame is not None and frame_bytes != previous_frame:
                changed_frames += 1
            previous_frame = frame_bytes
        if process is not None:
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            return_code = process.wait()
            if return_code != 0:
                return {"status": "failed", "message": "ffmpeg failed", "ffmpeg_tail": stderr[-800:]}
        if writer is not None:
            writer.close()
            writer = None
        if not video_path.exists() or video_path.stat().st_size == 0:
            return {"status": "failed", "message": "MuJoCo render video was not written."}
        if not frame_stds or max(frame_stds) <= 0.0:
            return {"status": "failed", "message": "MuJoCo renderer produced blank frames."}
        return {
            "status": "ok",
            "backend": "mujoco_kinematic_qpos",
            "encode_backend": encode_backend,
            "video_path": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frames": int(motion["frame_count"]),
            "changed_frames": changed_frames,
            "frame_sum_min": min(frame_sums),
            "frame_sum_max": max(frame_sums),
            "frame_std_max": round(max(frame_stds), 4),
            "root_xy_locked": False,
            "camera_mode": camera_mode,
            "missing_model_joints": sorted(missing_joints),
        }
    except Exception as exc:
        if process is not None and process.poll() is None:
            process.kill()
        if writer is not None:
            writer.close()
        return {"status": "failed", "message": f"MuJoCo G1 render failed: {exc}"}
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()


def make_mujoco_camera(mujoco: Any, motion: Mapping[str, Any], *, mode: str) -> Any:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    if mode == "fixed":
        camera.lookat = [0.0, 0.0, DEFAULT_ROOT_HEIGHT]
        camera.distance = 4.0
    else:
        min_xy = np.min(root_pos[:, :2], axis=0)
        max_xy = np.max(root_pos[:, :2], axis=0)
        center_xy = (min_xy + max_xy) * 0.5
        span_xy = float(np.linalg.norm(max_xy - min_xy))
        camera.lookat = [float(center_xy[0]), float(center_xy[1]), max(DEFAULT_ROOT_HEIGHT, float(np.mean(root_pos[:, 2])))]
        camera.distance = max(3.0, 2.0 + span_xy * 0.75)
    camera.elevation = -14
    camera.azimuth = 135
    return camera


def follow_camera_to_pelvis(model: Any, data: Any, camera: Any) -> None:
    try:
        body_id = model.body("pelvis").id
        pos = data.xpos[body_id]
        camera.lookat = [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception:
        pass


def set_mujoco_state_by_joint_names(
    model: Any,
    data: Any,
    *,
    root: Sequence[float],
    root_quat: Sequence[float],
    joint_names: Sequence[str],
    joint_values: Sequence[float],
) -> list[str]:
    qpos = data.qpos
    if len(qpos) >= 7:
        qpos[0:3] = [float(root[0]), float(root[1]), float(root[2])]
        qpos[3:7] = _normalize_quat_wxyz(root_quat)
    missing: list[str] = []
    for joint_name, value in zip(joint_names, joint_values):
        try:
            joint_id = model.joint(joint_name).id
            qpos_address = model.jnt_qposadr[joint_id]
        except Exception:
            missing.append(str(joint_name))
            continue
        if 0 <= qpos_address < len(qpos):
            qpos[qpos_address] = float(value)
    return missing


def combine_two_videos(inputs: Sequence[Path], output: Path, fps: int) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return combine_two_videos_imageio(inputs, output, fps=fps)
    missing = [str(path) for path in inputs if not path.exists() or path.stat().st_size == 0]
    if missing:
        return {"status": "failed", "message": "missing input panel", "missing": missing}
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y"]
    for path in inputs:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            f"[0:v][1:v]hstack=inputs=2,fps={max(1, fps)}[v]",
            "-map",
            "[v]",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return {"status": "failed", "message": "ffmpeg failed", "ffmpeg_tail": result.stderr[-800:]}
    return {"status": "ok", "video_path": str(output), "bytes": output.stat().st_size, "fps": max(1, fps)}


def combine_two_videos_imageio(inputs: Sequence[Path], output: Path, *, fps: int) -> dict[str, Any]:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        return {"status": "blocked", "message": f"imageio is required when ffmpeg is unavailable: {exc}"}
    missing = [str(path) for path in inputs if not path.exists() or path.stat().st_size == 0]
    if missing:
        return {"status": "failed", "message": "missing input panel", "missing": missing}
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    changed_frames = 0
    previous_frame: bytes | None = None
    try:
        readers = [imageio.get_reader(path) for path in inputs]
        try:
            with imageio.get_writer(output, fps=max(1, fps), codec="libx264", quality=8, macro_block_size=1) as writer:
                for left, right in zip(readers[0], readers[1]):
                    left_arr = np.asarray(left[..., :3], dtype=np.uint8)
                    right_arr = np.asarray(right[..., :3], dtype=np.uint8)
                    if left_arr.shape[0] != right_arr.shape[0]:
                        right_arr = resize_frame_to_height(right_arr, left_arr.shape[0])
                    if left_arr.shape[1] != right_arr.shape[1]:
                        right_arr = resize_frame_to_width(right_arr, left_arr.shape[1])
                    frame = np.concatenate([left_arr, right_arr], axis=1)
                    writer.append_data(frame)
                    frame_bytes = frame.tobytes()
                    if previous_frame is not None and frame_bytes != previous_frame:
                        changed_frames += 1
                    previous_frame = frame_bytes
                    frame_count += 1
        finally:
            for reader in readers:
                reader.close()
    except Exception as exc:
        return {"status": "failed", "message": f"imageio hstack failed: {exc}"}
    if not output.exists() or output.stat().st_size == 0:
        return {"status": "failed", "message": "combined video was not written"}
    return {
        "status": "ok",
        "video_path": str(output),
        "bytes": output.stat().st_size,
        "fps": max(1, fps),
        "frames": frame_count,
        "changed_frames": changed_frames,
        "backend": "imageio_hstack",
    }


def resize_frame_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    width = max(1, int(round(frame.shape[1] * (height / max(1, frame.shape[0])))))
    return resize_frame_nearest(frame, height=height, width=width)


def resize_frame_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    height = max(1, int(round(frame.shape[0] * (width / max(1, frame.shape[1])))))
    return resize_frame_nearest(frame, height=height, width=width)


def resize_frame_nearest(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    y_idx = np.linspace(0, frame.shape[0] - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).round().astype(np.int64)
    return frame[y_idx][:, x_idx]


if __name__ == "__main__":
    main()
