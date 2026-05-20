#!/usr/bin/env python3
"""Overlay orientation labels on existing IsaacLab G1 playback videos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_retarget.data.bones_sonic import SONIC_BODY_NAMES


DEFAULT_CLIPS = (
    "00_reach_jump_R_001__A419",
    "01_jump_over_obstacle_1_5m_L_001__A415",
    "02_point_laugh_head_arm_high_R_001__A550",
    "03_neutral_itching_head_R_001__A542",
    "04_drinking_bottle_throw_270_R_001__A550",
)

AXIS_COLORS = {
    "+X": (57, 57, 214),
    "+Y": (72, 137, 55),
    "+Z": (190, 90, 56),
}
FRONT_COLOR = (25, 91, 205)
LEFT_COLOR = (160, 53, 126)
RIGHT_COLOR = (150, 116, 20)
TEXT_COLOR = (25, 31, 35)
STROKE_COLOR = (245, 247, 250)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("runs/vis_pair_check/seed_mapping_5motion_vis_20260520"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/vis_pair_check"))
    parser.add_argument("--run-name", default="seed_isaac_orientation_debug_5motion_20260520")
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--camera-offset", type=float, nargs=3, default=(2.5, -3.0, 1.6))
    parser.add_argument("--axis-length", type=float, default=0.45)
    parser.add_argument("--front-length", type=float, default=0.55)
    args = parser.parse_args()

    clips = args.clip or list(DEFAULT_CLIPS)
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for clip in clips:
        input_dir = args.input_root / clip
        if not input_dir.exists():
            raise FileNotFoundError(f"missing input clip directory: {input_dir}")
        report = _read_json(input_dir / "target_g1_isaaclab.json")
        video_path = input_dir / "target_g1_isaaclab.mp4"
        npz_path = Path(str(report["npz"]))
        clip_dir = output_dir / clip
        clip_dir.mkdir(parents=True, exist_ok=True)
        output_video = clip_dir / "target_g1_isaaclab_orientation_debug.mp4"
        overlay_report = _overlay_video(
            video_path=video_path,
            npz_path=npz_path,
            output_video=output_video,
            camera_offset=np.asarray(args.camera_offset, dtype=np.float64),
            axis_length=args.axis_length,
            front_length=args.front_length,
        )
        combined_report = {
            "clip": clip,
            "source_isaac_video": str(video_path),
            "source_isaac_report": str(input_dir / "target_g1_isaaclab.json"),
            "npz": str(npz_path),
            "output_video": str(output_video),
            "overlay": overlay_report,
            "note": (
                "This is a post-process overlay on the existing IsaacLab kinematic "
                "playback. The underlying robot motion/video is unchanged."
            ),
        }
        report_path = clip_dir / "target_g1_isaaclab_orientation_debug.json"
        report_path.write_text(json.dumps(combined_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "clip": clip,
                "input_video": str(video_path),
                "npz": str(npz_path),
                "output_video": str(output_video),
                "status": overlay_report["status"],
                "frames": overlay_report["frames"],
                "fps": overlay_report["fps"],
                "width": overlay_report["width"],
                "height": overlay_report["height"],
                "projected_label_frames": overlay_report["projected_label_frames"],
                "report": str(report_path),
            }
        )

    _write_csv(output_dir / "summary.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "input_root": str(args.input_root),
                "output_dir": str(output_dir),
                "overlay_semantics": _overlay_semantics(),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_readme(rows), encoding="utf-8")
    _write_integrity(output_dir, rows)
    _write_thumbnails(output_dir, rows)
    print(json.dumps({"output_dir": str(output_dir), "rows": rows}, indent=2, sort_keys=True))


def _overlay_video(
    *,
    video_path: Path,
    npz_path: Path,
    output_video: Path,
    camera_offset: np.ndarray,
    axis_length: float,
    front_length: float,
) -> dict[str, object]:
    with np.load(npz_path) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = min(frame_count, int(body_pos.shape[0]))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(video_fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("failed to open ffmpeg stdin")

    root_offset = body_pos[0, 0].copy()
    projected_label_frames = 0
    changed_frames = 0
    previous_bytes: bytes | None = None
    frame_sums: list[int] = []

    for frame_index in range(frame_limit):
        ok, frame = cap.read()
        if not ok:
            break
        rendered_body = body_pos[frame_index].copy()
        rendered_body[:, :2] -= root_offset[:2]
        root_pos = rendered_body[0].copy()
        target = root_pos
        look_at = target + np.asarray((0.0, 0.0, 0.35), dtype=np.float64)
        eye = target + camera_offset
        projector = _Projector(width=width, height=height, eye=eye, look_at=look_at)

        if _draw_orientation_overlay(
            frame,
            rendered_body,
            projector,
            axis_length=axis_length,
            front_length=front_length,
            frame_index=frame_index,
            clip_label=video_path.parent.name,
        ):
            projected_label_frames += 1

        process.stdin.write(frame.tobytes())
        frame_bytes = frame.tobytes()
        frame_sums.append(int(frame.sum()))
        if previous_bytes is not None and previous_bytes != frame_bytes:
            changed_frames += 1
        previous_bytes = frame_bytes

    cap.release()
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {stderr[-2000:]}")

    return {
        "status": "ok",
        "input_video": str(video_path),
        "npz": str(npz_path),
        "output_video": str(output_video),
        "fps": video_fps,
        "motion_fps": fps,
        "frames": frame_limit,
        "input_video_frames": frame_count,
        "motion_frames": int(body_pos.shape[0]),
        "changed_frames": changed_frames,
        "projected_label_frames": projected_label_frames,
        "width": width,
        "height": height,
        "frame_sum_min": min(frame_sums) if frame_sums else None,
        "frame_sum_max": max(frame_sums) if frame_sums else None,
    }


class _Projector:
    def __init__(self, *, width: int, height: int, eye: np.ndarray, look_at: np.ndarray) -> None:
        self.width = width
        self.height = height
        self.eye = eye
        forward = _normalize(look_at - eye)
        world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        right = _normalize(np.cross(forward, world_up))
        if np.linalg.norm(right) < 1e-8:
            right = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        up = _normalize(np.cross(right, forward))
        self.forward = forward
        self.right = right
        self.up = up
        self.fx = width * 26.0 / 24.0
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0

    def project(self, point: np.ndarray) -> tuple[int, int] | None:
        rel = point - self.eye
        depth = float(np.dot(rel, self.forward))
        if depth <= 0.03:
            return None
        x = float(np.dot(rel, self.right))
        y = float(np.dot(rel, self.up))
        u = self.cx + self.fx * (x / depth)
        v = self.cy - self.fy * (y / depth)
        if not math.isfinite(u) or not math.isfinite(v):
            return None
        return int(round(u)), int(round(v))


def _draw_orientation_overlay(
    frame: np.ndarray,
    body_pos: np.ndarray,
    projector: _Projector,
    *,
    axis_length: float,
    front_length: float,
    frame_index: int,
    clip_label: str,
) -> bool:
    names = {name: idx for idx, name in enumerate(SONIC_BODY_NAMES)}
    pelvis = body_pos[names["pelvis"]]
    torso = body_pos[names["torso_link"]]
    left_shoulder = body_pos[names["left_shoulder_pitch_link"]]
    right_shoulder = body_pos[names["right_shoulder_pitch_link"]]
    left_wrist = body_pos[names["left_wrist_yaw_link"]]
    right_wrist = body_pos[names["right_wrist_yaw_link"]]

    origin = pelvis + np.asarray((0.0, 0.0, 0.05), dtype=np.float64)
    up = _normalize(torso - pelvis)
    shoulder_left = left_shoulder - right_shoulder
    front = _normalize(np.cross(shoulder_left, up))
    if np.linalg.norm(front) < 1e-8:
        front = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)

    projected = False
    for label, direction in (
        ("+X", np.asarray((1.0, 0.0, 0.0), dtype=np.float64)),
        ("+Y", np.asarray((0.0, 1.0, 0.0), dtype=np.float64)),
        ("+Z", np.asarray((0.0, 0.0, 1.0), dtype=np.float64)),
    ):
        projected |= _draw_arrow_3d(
            frame,
            projector,
            origin,
            origin + direction * axis_length,
            AXIS_COLORS[label],
            label,
        )

    projected |= _draw_arrow_3d(
        frame,
        projector,
        torso,
        torso + front * front_length,
        FRONT_COLOR,
        "FRONT",
        thickness=4,
    )
    projected |= _draw_label_3d(frame, projector, left_wrist, "L", LEFT_COLOR)
    projected |= _draw_label_3d(frame, projector, right_wrist, "R", RIGHT_COLOR)

    _draw_text(frame, "IsaacLab G1 playback + orientation overlay", (16, 30), 0.62, TEXT_COLOR)
    _draw_text(frame, f"{clip_label}  frame {frame_index:04d}", (16, 58), 0.50, TEXT_COLOR)
    _draw_text(
        frame,
        "red +X / green +Y / blue +Z / orange FRONT / purple L / cyan R",
        (16, frame.shape[0] - 22),
        0.48,
        TEXT_COLOR,
    )
    _draw_text(
        frame,
        "camera follows root from offset (2.5, -3.0, 1.6)",
        (frame.shape[1] - 390, frame.shape[0] - 22),
        0.43,
        TEXT_COLOR,
    )
    return projected


def _draw_arrow_3d(
    frame: np.ndarray,
    projector: _Projector,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    *,
    thickness: int = 3,
) -> bool:
    start_2d = projector.project(start)
    end_2d = projector.project(end)
    if start_2d is None or end_2d is None:
        return False
    cv2.arrowedLine(frame, start_2d, end_2d, color, thickness, cv2.LINE_AA, tipLength=0.16)
    _draw_text(frame, label, (end_2d[0] + 6, end_2d[1] - 6), 0.56, color)
    return True


def _draw_label_3d(
    frame: np.ndarray,
    projector: _Projector,
    point: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> bool:
    point_2d = projector.project(point)
    if point_2d is None:
        return False
    cv2.circle(frame, point_2d, 9, color, -1, cv2.LINE_AA)
    _draw_text(frame, label, (point_2d[0] + 10, point_2d[1] - 10), 0.72, color, thickness=2)
    return True


def _draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, STROKE_COLOR, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return vector * 0.0
    return vector / norm


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_integrity(output_dir: Path, rows: Iterable[dict[str, object]]) -> None:
    integrity: list[dict[str, object]] = []
    for row in rows:
        video_path = Path(str(row["output_video"]))
        payload = _ffprobe(video_path)
        payload["clip"] = row["clip"]
        payload["path"] = str(video_path)
        integrity.append(payload)
    _write_csv(output_dir / "isaac_orientation_video_integrity.csv", integrity)
    (output_dir / "isaac_orientation_video_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_thumbnails(output_dir: Path, rows: Iterable[dict[str, object]]) -> None:
    thumbs = output_dir / "thumbs"
    thumbs.mkdir(exist_ok=True)
    for row in rows:
        clip = str(row["clip"])
        video_path = Path(str(row["output_video"]))
        thumb_path = thumbs / f"{clip}_isaac_orientation_t1.png"
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            "1.0",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(thumb_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ffprobe(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "nb_frames": int(stream.get("nb_frames") or 0),
        "r_frame_rate": stream.get("r_frame_rate", ""),
        "duration_s": float(stream.get("duration") or 0.0),
        "size_bytes": path.stat().st_size,
    }


def _overlay_semantics() -> dict[str, str]:
    return {
        "+X": "red world +X projected from the robot pelvis",
        "+Y": "green world +Y projected from the robot pelvis",
        "+Z": "blue world +Z/up projected from the robot pelvis",
        "FRONT": "orange body-facing direction computed as cross(left_shoulder - right_shoulder, torso_up)",
        "L": "purple left_wrist_yaw_link",
        "R": "cyan right_wrist_yaw_link",
    }


def _readme(rows: list[dict[str, object]]) -> str:
    lines = [
        "# IsaacLab Orientation Debug Videos",
        "",
        "这些视频是对已验证 IsaacLab G1 kinematic playback 的后处理标注。底层机器人视频没有改变；标注来自同一份 BONES-SONIC `body_pos_w`，并用渲染时相同的 root-follow camera offset 近似投影到画面上。",
        "",
        "颜色含义：red `+X`，green `+Y`，blue `+Z`，orange `FRONT`，purple `L`，cyan `R`。",
        "",
        "| clip | annotated Isaac video | frames | fps |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['clip']}` | `{row['output_video']}` | {row['frames']} | {row['fps']} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
