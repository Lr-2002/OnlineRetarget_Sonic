"""A0 visual-validation data boundary.

The trainer owns model inference and sample selection. This module owns the
visualization-facing coordinate and backend contract for A0 frozen-Skeleton-AE
validation clips.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_G1_USD = Path("/home/user/project/OnlineRetarget/runs/isaaclab_urdf_cache/g1_main/main.usd")
PRIMARY_VISUAL_BACKEND = "somamesh_global_soma_plus_isaaclab_g1_kinematic_playback"
ACCEPTANCE_SOURCE_BACKEND = "accepted_somamesh_global_soma_display"
ACCEPTANCE_G1_BACKEND = "isaaclab_usd_g1_kinematic_playback"
DEBUG_CAPSULE_BACKEND = "software_capsule_debug_fallback"
SOMA_DISPLAY_TRANSFORM = "(x,y,z)_display=(x,-z,y)_soma"
ACCEPTANCE_OVERLAYS = ("world_axes", "root_axes", "semantic_left_right")


class A0VisualValidationRenderer:
    """Stable A0 visual-validation boundary used by the trainer.

    The primary acceptance backend is SomaMesh/global-SOMA source playback plus
    IsaacLab G1 kinematic playback. The legacy in-process capsule renderer is
    allowed only as debug fallback and is recorded as such in clip metadata.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        g1_usd_path: Path | str | None = None,
    ) -> None:
        self._config = config
        visual_cfg = config.get("visual_validation", {})
        if not isinstance(visual_cfg, Mapping):
            visual_cfg = {}
        configured_usd = visual_cfg.get("g1_robot_usd") or visual_cfg.get("g1_usd") or g1_usd_path
        self.g1_usd_path = Path(str(configured_usd)) if configured_usd else DEFAULT_G1_USD

    def compose_prediction_root(
        self,
        predicted_root_pos: np.ndarray,
        fallback_root_pos: np.ndarray,
    ) -> np.ndarray:
        """Return visualization-world root positions for model predictions."""

        root = np.asarray(predicted_root_pos, dtype=np.float32).copy()
        if self.should_compose_soma_root_xy:
            root[:, :2] += np.asarray(fallback_root_pos, dtype=np.float32)[:, :2]
        return root

    def compose_prediction_state(
        self,
        prediction: Mapping[str, np.ndarray],
        *,
        fallback_root_pos: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Copy prediction state and compose root XY at the visual boundary."""

        updated = {key: np.asarray(value, dtype=np.float32) for key, value in prediction.items()}
        if "root_pos" in updated:
            updated["root_pos"] = self.compose_prediction_root(updated["root_pos"], fallback_root_pos)
        return updated

    @property
    def should_compose_soma_root_xy(self) -> bool:
        input_data = self._config.get("input_data", {})
        return (
            isinstance(input_data, Mapping)
            and input_data.get("format") == "soma_motionlib"
            and _include_root_pos_target(self._config)
        )

    def root_composition_metadata(self) -> dict[str, Any]:
        return {
            "compose_soma_root_local_xy_to_world": bool(self.should_compose_soma_root_xy),
            "condition": 'input_data.format == "soma_motionlib" and include_root_pos_target == true',
            "xy_operation": "pred_root[:, :2] += fallback_root_pos[:, :2]",
            "z_semantics": "predicted_absolute_z",
            "root_rot6d_semantics": "predicted_absolute_rot6d",
        }

    def backend_manifest(self, *, active_backend: str) -> dict[str, Any]:
        active_is_acceptance = active_backend == PRIMARY_VISUAL_BACKEND
        return {
            "primary_backend": PRIMARY_VISUAL_BACKEND,
            "active_backend": active_backend,
            "debug_fallback_backend": DEBUG_CAPSULE_BACKEND,
            "source_human_backend": ACCEPTANCE_SOURCE_BACKEND,
            "source_display_transform": SOMA_DISPLAY_TRANSFORM,
            "g1_backend": ACCEPTANCE_G1_BACKEND,
            "g1_asset_usd": str(self.g1_usd_path),
            "required_overlays": list(ACCEPTANCE_OVERLAYS),
            "active_backend_is_acceptance_backend": active_is_acceptance,
            "debug_fallback_is_acceptance_backend": False,
        }

    def acceptance_backend_manifest(self) -> dict[str, Any]:
        return self.backend_manifest(active_backend=PRIMARY_VISUAL_BACKEND)

    def rerender_cli_command(
        self,
        *,
        config_path: Path | str,
        checkpoint_path: Path | str,
        output_dir: Path | str,
        rows_cache: Path | str | None = None,
        stats_path: Path | str | None = None,
        step: int | None = None,
        count: int | None = None,
        python_bin: Path | str = sys.executable,
        script_path: Path | str = "scripts/rerender_a0_visual_validation.py",
        isaac_python_bin: Path | str = "/workspace/isaaclab/_isaac_sim/python.sh",
    ) -> list[str]:
        """Build the stable acceptance rerender CLI command."""

        command = [
            str(python_bin),
            str(script_path),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--acceptance-backend",
            "--isaac-python-bin",
            str(isaac_python_bin),
        ]
        if rows_cache is not None:
            command.extend(["--rows-cache", str(rows_cache)])
        if stats_path is not None:
            command.extend(["--stats", str(stats_path)])
        if step is not None:
            command.extend(["--step", str(int(step))])
        if count is not None:
            command.extend(["--num-videos", str(int(count))])
        return command

    def isaaclab_g1_render_command(
        self,
        *,
        python_bin: Path | str,
        script_path: Path | str,
        motion_path: Path | str,
        output_path: Path | str,
        duration_sec: float,
        width: int,
        height: int,
    ) -> list[str]:
        """Build the stable G1 IsaacLab playback command for rerender handoff."""

        return [
            str(python_bin),
            str(script_path),
            "--g1-motion",
            str(motion_path),
            "--format",
            "npz",
            "--output",
            str(output_path),
            "--duration-sec",
            f"{float(duration_sec):g}",
            "--robot-usd",
            str(self.g1_usd_path),
            "--preserve-world-root",
            "--width",
            str(int(width)),
            "--height",
            str(int(height)),
            "--overlay-world-root-axes",
            "--overlay-semantic-lr",
        ]

    def write_g1_motion_npz(
        self,
        *,
        path: Path | str,
        joint_pos: np.ndarray,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        fps: float,
        joint_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Write the kinematic G1 motion asset consumed by the IsaacLab renderer."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        root_pos = np.asarray(root_pos, dtype=np.float32)
        root_quat = np.asarray(root_quat, dtype=np.float32)
        frame_count = min(joint_pos.shape[0], root_pos.shape[0], root_quat.shape[0])
        if frame_count <= 0:
            raise ValueError("cannot write G1 playback with zero frames")
        joint_pos = joint_pos[:frame_count]
        root_pos = root_pos[:frame_count]
        root_quat = _normalize_quat_array(root_quat[:frame_count])
        joint_vel = _finite_difference_velocity(joint_pos, float(fps))
        payload: dict[str, Any] = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "root_pos": root_pos,
            "root_quat": root_quat,
            "fps": np.asarray([float(fps)], dtype=np.float32),
        }
        if joint_names is not None:
            payload["joint_names"] = np.asarray([str(name) for name in joint_names])
        np.savez(out, **payload)
        return {
            "status": "ok",
            "path": str(out),
            "format": "npz",
            "frames": int(frame_count),
            "fps": float(fps),
            "root_xy_preserved": True,
            "root_xy_span_m": _root_xy_span(root_pos),
        }

    def render_g1_isaaclab_playback(
        self,
        *,
        python_bin: Path | str,
        script_path: Path | str,
        motion_path: Path | str,
        output_path: Path | str,
        duration_sec: float,
        width: int,
        height: int,
        execute: bool = True,
        cwd: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run or record the real IsaacLab USD kinematic playback command."""

        output = Path(output_path)
        command = self.isaaclab_g1_render_command(
            python_bin=python_bin,
            script_path=script_path,
            motion_path=motion_path,
            output_path=output,
            duration_sec=duration_sec,
            width=width,
            height=height,
        )
        command_record = output.with_suffix(output.suffix + ".command.json")
        command_record.parent.mkdir(parents=True, exist_ok=True)
        command_payload = {
            "backend": ACCEPTANCE_G1_BACKEND,
            "command": command,
            "robot_usd": str(self.g1_usd_path),
            "motion_path": str(motion_path),
            "output_path": str(output),
            "overlays": list(ACCEPTANCE_OVERLAYS),
            "preserve_world_root": True,
            "execute": bool(execute),
        }
        command_record.write_text(json.dumps(command_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not execute:
            return {
                "status": "ready",
                "backend": ACCEPTANCE_G1_BACKEND,
                "command": command,
                "command_record": str(command_record),
                "robot_usd": str(self.g1_usd_path),
                "output": str(output),
                "overlays": list(ACCEPTANCE_OVERLAYS),
                "preserve_world_root": True,
            }
        result = subprocess.run(command, cwd=str(cwd) if cwd is not None else None, capture_output=True, text=True)
        report_path = output.with_suffix(".json")
        report: dict[str, Any] = {}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                report = {"report_parse_status": "failed"}
        report_status = report.get("status")
        status = (
            "ok"
            if result.returncode == 0
            and output.exists()
            and output.stat().st_size > 0
            and (report_status in (None, "ok"))
            else "failed"
        )
        return {
            "status": status,
            "backend": ACCEPTANCE_G1_BACKEND,
            "command": command,
            "command_record": str(command_record),
            "robot_usd": str(self.g1_usd_path),
            "output": str(output),
            "report": str(report_path) if report_path.exists() else "",
            "returncode": int(result.returncode),
            "stdout_tail": result.stdout[-1000:],
            "stderr_tail": result.stderr[-1000:],
            "overlays": list(ACCEPTANCE_OVERLAYS),
            "preserve_world_root": True,
            "isaaclab_report": report,
        }

    def render_somamesh_global_source_video(
        self,
        *,
        frames: Sequence[Mapping[str, Sequence[float]]],
        edges: Sequence[tuple[str, str]],
        video_path: Path | str,
        fps: float,
        width: int,
        height: int,
        label: str,
    ) -> dict[str, Any]:
        """Render accepted global-SOMA source playback without using the capsule fallback."""

        ffmpeg = _ffmpeg_executable()
        if ffmpeg is None:
            return {"status": "blocked", "message": "ffmpeg is required for accepted SomaMesh source rendering"}
        clean_frames = [_finite_frame(frame) for frame in frames]
        if not clean_frames:
            return {"status": "blocked", "message": "no source frames were available for accepted SomaMesh rendering"}
        output = Path(video_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fps_int = max(1, min(240, int(round(float(fps)))))
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
            f"{int(width)}x{int(height)}",
            "-r",
            str(fps_int),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        bounds = _source_scene_bounds(clean_frames)
        frame_sums: list[int] = []
        changed_frames = 0
        previous_frame: bytes | None = None
        process = None
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            if process.stdin is None:
                return {"status": "failed", "message": "ffmpeg stdin was unavailable"}
            for frame_index, frame in enumerate(clean_frames):
                image = _draw_somamesh_frame(
                    frame,
                    edges,
                    bounds,
                    int(width),
                    int(height),
                    label=f"{label} frame {frame_index:04d}",
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
                    "message": "ffmpeg failed while encoding accepted SomaMesh source video",
                    "ffmpeg_tail": stderr[-800:],
                }
            if not output.exists() or output.stat().st_size == 0:
                return {"status": "failed", "message": "accepted SomaMesh source video was not written"}
            return {
                "status": "ok",
                "backend": ACCEPTANCE_SOURCE_BACKEND,
                "render_backend": ACCEPTANCE_SOURCE_BACKEND,
                "video_path": str(output),
                "frames": len(clean_frames),
                "fps": fps_int,
                "width": int(width),
                "height": int(height),
                "source_display_transform": SOMA_DISPLAY_TRANSFORM,
                "overlays": list(ACCEPTANCE_OVERLAYS),
                "edge_pairs": [f"{start}->{end}" for start, end in edges],
                "changed_frames": changed_frames,
                "frame_sum_min": min(frame_sums),
                "frame_sum_max": max(frame_sums),
            }
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()
            return {"status": "failed", "message": f"accepted SomaMesh source render failed: {exc}"}

    @staticmethod
    def soma_point_to_display(point: Sequence[float]) -> tuple[float, float, float]:
        x, y, z = (float(point[index]) if index < len(point) else 0.0 for index in range(3))
        return (x, -z, y)

    @classmethod
    def soma_frame_maps_to_display(
        cls,
        frames: Sequence[Mapping[str, Sequence[float]]],
    ) -> list[dict[str, tuple[float, float, float]]]:
        return [
            {name: cls.soma_point_to_display(point) for name, point in frame.items()}
            for frame in frames
        ]

    @classmethod
    def soma_motionlib_source_frames(
        cls,
        soma_joints: np.ndarray,
        joint_names: Sequence[str],
    ) -> list[dict[str, tuple[float, float, float]]]:
        joints = np.asarray(soma_joints, dtype=np.float32)
        usable = min(len(joint_names), joints.shape[1])
        return [
            {
                joint_names[index]: cls.soma_point_to_display(frame[index])
                for index in range(usable)
            }
            for frame in joints
        ]


def _include_root_pos_target(config: Mapping[str, Any]) -> bool:
    features = config.get("features", {})
    if not isinstance(features, Mapping):
        return False
    explicit = features.get("include_root_pos_target")
    if explicit is not None:
        return bool(explicit)
    target_text = " ".join(
        str(features.get(key, ""))
        for key in ("target_feature", "target_features", "target_pose_feature")
    )
    return "root_pos" in target_text


def _finite_difference_velocity(joint_pos: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(joint_pos, dtype=np.float32)
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, axis=0) * float(fps)


def _ffmpeg_executable() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    try:
        import imageio_ffmpeg

        imageio_ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None
    if imageio_ffmpeg_path.exists():
        return str(imageio_ffmpeg_path)
    return None


def _normalize_quat_array(quat: np.ndarray) -> np.ndarray:
    values = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.where(norm < 1e-8, 1.0, norm)


def _root_xy_span(root_pos: np.ndarray) -> float:
    values = np.asarray(root_pos, dtype=np.float32)
    if values.shape[0] <= 1:
        return 0.0
    return float(np.linalg.norm(values[-1, :2] - values[0, :2]))


def _finite_frame(frame: Mapping[str, Sequence[float]]) -> dict[str, tuple[float, float, float]]:
    clean: dict[str, tuple[float, float, float]] = {}
    for name, point in frame.items():
        values = tuple(float(point[index]) if index < len(point) else 0.0 for index in range(3))
        if all(math.isfinite(value) for value in values):
            clean[str(name)] = values
    return clean


def _source_scene_bounds(frames: Sequence[Mapping[str, tuple[float, float, float]]]) -> dict[str, float]:
    projected = [_source_project_raw(point) for frame in frames for point in frame.values()]
    if not projected:
        return {"min_x": -1.0, "max_x": 1.0, "min_y": -1.0, "max_y": 1.0}
    xs = [point[0] for point in projected]
    ys = [point[1] for point in projected]
    span_x = max(max(xs) - min(xs), 0.5)
    span_y = max(max(ys) - min(ys), 0.5)
    pad_x = span_x * 0.15
    pad_y = span_y * 0.15
    return {
        "min_x": min(xs) - pad_x,
        "max_x": max(xs) + pad_x,
        "min_y": min(ys) - pad_y,
        "max_y": max(ys) + pad_y,
    }


def _source_project_raw(point: Sequence[float]) -> tuple[float, float]:
    x, y, z = (float(point[index]) if index < len(point) else 0.0 for index in range(3))
    return (x + 0.24 * y, z + 0.18 * y)


def _source_project(
    point: Sequence[float],
    bounds: Mapping[str, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    x, y = _source_project_raw(point)
    span_x = max(float(bounds["max_x"]) - float(bounds["min_x"]), 1e-6)
    span_y = max(float(bounds["max_y"]) - float(bounds["min_y"]), 1e-6)
    px = 28 + (x - float(bounds["min_x"])) / span_x * max(1, width - 56)
    py = height - 34 - (y - float(bounds["min_y"])) / span_y * max(1, height - 68)
    return (int(round(px)), int(round(py)))


def _draw_somamesh_frame(
    frame: Mapping[str, tuple[float, float, float]],
    edges: Sequence[tuple[str, str]],
    bounds: Mapping[str, float],
    width: int,
    height: int,
    *,
    label: str,
) -> bytearray:
    image = bytearray((238, 241, 237) * width * height)
    _draw_source_grid(image, width, height)
    _draw_source_axes(image, width, height)
    projected = {name: _source_project(point, bounds, width, height) for name, point in frame.items()}
    for start, end in edges:
        if start not in projected or end not in projected:
            continue
        color = _semantic_color(start, end)
        _draw_line(image, width, height, projected[start], projected[end], radius=3, color=(36, 48, 48))
        _draw_line(image, width, height, projected[start], projected[end], radius=2, color=color)
    for name, center in projected.items():
        color = _semantic_color(name)
        radius = 7 if _key_source_point(name) else 5
        _draw_circle(image, width, height, center, radius=radius + 1, color=(34, 47, 47))
        _draw_circle(image, width, height, center, radius=radius, color=color)
    _draw_label_bars(image, width, height, label)
    return image


def _draw_source_grid(image: bytearray, width: int, height: int) -> None:
    for x in range(28, width, max(32, width // 8)):
        _draw_line(image, width, height, (x, 28), (x, height - 28), radius=0, color=(215, 220, 214))
    for y in range(28, height, max(28, height // 6)):
        _draw_line(image, width, height, (28, y), (width - 28, y), radius=0, color=(215, 220, 214))


def _draw_source_axes(image: bytearray, width: int, height: int) -> None:
    origin = (38, height - 38)
    _draw_line(image, width, height, origin, (92, height - 38), radius=3, color=(190, 55, 45))
    _draw_line(image, width, height, origin, (38, height - 92), radius=3, color=(55, 120, 65))
    _draw_line(image, width, height, origin, (72, height - 72), radius=3, color=(55, 80, 170))


def _draw_label_bars(image: bytearray, width: int, height: int, label: str) -> None:
    del label
    _draw_rect(image, width, height, 10, 10, min(width - 20, 330), 22, color=(221, 228, 222))
    _draw_rect(image, width, height, width - 116, 10, 44, 16, color=(48, 132, 83))
    _draw_rect(image, width, height, width - 62, 10, 44, 16, color=(154, 66, 91))


def _semantic_color(*names: str) -> tuple[int, int, int]:
    text = " ".join(names).lower()
    if "left" in text:
        return (48, 132, 83)
    if "right" in text:
        return (154, 66, 91)
    return (75, 96, 150)


def _key_source_point(name: str) -> bool:
    lowered = name.lower()
    return lowered in {"hips", "pelvis", "head"} or "hand" in lowered or "foot" in lowered


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
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if radius <= 0:
            _set_pixel(image, width, height, x0, y0, color)
        else:
            _draw_circle(image, width, height, (x0, y0), radius=radius, color=color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


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
    r2 = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r2:
                _set_pixel(image, width, height, x, y, color)


def _draw_rect(
    image: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    rect_width: int,
    rect_height: int,
    *,
    color: tuple[int, int, int],
) -> None:
    for yy in range(y, y + rect_height):
        for xx in range(x, x + rect_width):
            _set_pixel(image, width, height, xx, yy, color)


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
    index = (y * width + x) * 3
    image[index : index + 3] = bytes(color)
