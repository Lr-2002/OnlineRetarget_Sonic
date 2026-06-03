"""A0 visual-validation data boundary.

The trainer owns model inference and sample selection. This module owns the
visualization-facing coordinate and backend contract for A0 frozen-Skeleton-AE
validation clips.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


G1_USD_RELATIVE_PATH = Path("runs/isaaclab_urdf_cache/g1_main/main.usd")
DEFAULT_G1_USD = Path.cwd() / G1_USD_RELATIVE_PATH
DEFAULT_ISAACLAB_RENDER_TIMEOUT_SEC = 900.0
DEFAULT_ISAACLAB_SUCCESS_ARTIFACT_GRACE_SEC = 2.0
DEFAULT_ISAACLAB_TERMINATE_GRACE_SEC = 10.0
ONLINE_RETARGET_ROOT_ENV_KEYS = (
    "ONLINE_RETARGET_RUNTIME_ROOT",
    "ONLINERETARGET_RUNTIME_ROOT",
    "ONLINE_RETARGET_ROOT",
)
PRIMARY_VISUAL_BACKEND = "somamesh_global_soma_plus_isaaclab_g1_target_and_kinematic_playback"
ACCEPTANCE_SOURCE_BACKEND = "accepted_somamesh_global_soma_display"
ACCEPTANCE_G1_BACKEND = "isaaclab_usd_g1_kinematic_playback"
DEBUG_CAPSULE_BACKEND = "software_capsule_debug_fallback"
SOMA_DISPLAY_TRANSFORM = "(x,y,z)_display=(x,-z,y)_soma"
ACCEPTANCE_OVERLAYS = ("world_axes", "root_axes", "semantic_left_right")


def resolve_g1_usd_path(
    config: Mapping[str, Any],
    explicit_path: Path | str | None = None,
    *,
    explicit_source: str = "",
) -> tuple[Path, dict[str, Any]]:
    """Resolve the acceptance G1 USD without assuming a fixed host checkout."""

    if explicit_path:
        path = Path(str(explicit_path)).expanduser()
        exists = path.exists()
        return path, {
            "status": "ok" if exists else "missing_explicit",
            "source": explicit_source or "explicit",
            "path": str(path),
            "exists": bool(exists),
            "failure_reasons": [] if exists else ["robot_usd_missing"],
        }

    roots = _online_retarget_runtime_roots(config)
    candidate_records: list[dict[str, str]] = []
    for source, root in roots:
        candidate = root / G1_USD_RELATIVE_PATH
        candidate_records.append({"source": source, "path": str(candidate)})
        if candidate.exists():
            return candidate, {
                "status": "ok",
                "source": source,
                "runtime_root": str(root),
                "path": str(candidate),
                "exists": True,
                "candidate_paths": candidate_records,
                "failure_reasons": [],
            }

    if candidate_records:
        path = Path(candidate_records[0]["path"])
        return path, {
            "status": "missing_derived",
            "source": candidate_records[0]["source"],
            "runtime_root": str(roots[0][1]),
            "path": str(path),
            "exists": False,
            "candidate_paths": candidate_records,
            "failure_reasons": ["robot_usd_missing"],
        }

    return DEFAULT_G1_USD, {
        "status": "missing_runtime_root",
        "source": "cwd",
        "runtime_root": str(Path.cwd()),
        "path": str(DEFAULT_G1_USD),
        "exists": bool(DEFAULT_G1_USD.exists()),
        "candidate_paths": [{"source": "cwd", "path": str(DEFAULT_G1_USD)}],
        "failure_reasons": [] if DEFAULT_G1_USD.exists() else ["robot_usd_missing"],
    }


def _online_retarget_runtime_roots(config: Mapping[str, Any]) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    visual_cfg = config.get("visual_validation", {})
    if isinstance(visual_cfg, Mapping):
        for key in ("online_retarget_root", "runtime_root"):
            if visual_cfg.get(key):
                candidates.append((f"visual_validation.{key}", Path(str(visual_cfg[key])).expanduser()))

    for key in ONLINE_RETARGET_ROOT_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            candidates.append((f"env.{key}", Path(value).expanduser()))

    runtime_cfg = config.get("runtime", {})
    if isinstance(runtime_cfg, Mapping) and runtime_cfg.get("write_root"):
        write_root = Path(str(runtime_cfg["write_root"])).expanduser()
        candidates.append(("runtime.write_root", write_root.parent if write_root.name == "outputs" else write_root))

    for source, value in _config_path_values(config):
        root = _path_before_outputs(Path(str(value)).expanduser())
        if root is not None:
            candidates.append((source, root))

    candidates.append(("cwd", Path.cwd()))
    return _dedupe_paths(candidates)


def _config_path_values(config: Mapping[str, Any]) -> list[tuple[str, object]]:
    values: list[tuple[str, object]] = []
    for key in ("output_dir",):
        if config.get(key):
            values.append((key, config[key]))
    for section_name in ("input_data", "wandb"):
        section = config.get(section_name, {})
        if not isinstance(section, Mapping):
            continue
        for key, value in section.items():
            if value and ("dir" in key or "root" in key or "cache" in key):
                values.append((f"{section_name}.{key}", value))
    return values


def _path_before_outputs(path: Path) -> Path | None:
    parts = path.parts
    if "outputs" not in parts:
        return None
    index = parts.index("outputs")
    if index <= 0:
        return None
    return Path(*parts[:index])


def _dedupe_paths(candidates: Sequence[tuple[str, Path]]) -> list[tuple[str, Path]]:
    deduped: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for source, path in candidates:
        key = str(path)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((source, path))
    return deduped


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
        configured_usd: Path | str | None = None
        configured_source = ""
        for key in ("g1_robot_usd", "g1_usd"):
            value = visual_cfg.get(key)
            if value:
                configured_usd = value
                configured_source = f"visual_validation.{key}"
                break
        if configured_usd is None and g1_usd_path is not None:
            configured_usd = g1_usd_path
            configured_source = "constructor.g1_usd_path"
        self.g1_usd_path, self.g1_usd_resolution = resolve_g1_usd_path(
            config,
            configured_usd,
            explicit_source=configured_source,
        )

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
            "g1_asset_usd_resolution": self.g1_usd_resolution,
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
        command.extend(["--g1-robot-usd", str(self.g1_usd_path)])
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
            "sha256": _file_sha256(out),
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
        timeout_sec: float | None = None,
        success_artifact_grace_sec: float | None = None,
        terminate_grace_sec: float | None = None,
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
            "robot_usd_resolution": self.g1_usd_resolution,
            "motion_path": str(motion_path),
            "output_path": str(output),
            "expected_output_path": str(output),
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
                "robot_usd_resolution": self.g1_usd_resolution,
                "output": str(output),
                "expected_output_path": str(output),
                "overlays": list(ACCEPTANCE_OVERLAYS),
                "preserve_world_root": True,
            }
        report_path = output.with_suffix(".json")
        timeout = _positive_float(
            timeout_sec,
            self._visual_validation_float("isaaclab_render_timeout_sec", DEFAULT_ISAACLAB_RENDER_TIMEOUT_SEC),
        )
        success_grace = _positive_float(
            success_artifact_grace_sec,
            self._visual_validation_float(
                "isaaclab_success_artifact_grace_sec",
                DEFAULT_ISAACLAB_SUCCESS_ARTIFACT_GRACE_SEC,
            ),
        )
        terminate_grace = _positive_float(
            terminate_grace_sec,
            self._visual_validation_float("isaaclab_terminate_grace_sec", DEFAULT_ISAACLAB_TERMINATE_GRACE_SEC),
        )
        result = _run_isaaclab_renderer_bounded(
            command,
            cwd=Path(cwd) if cwd is not None else None,
            output_path=output,
            report_path=report_path,
            timeout_sec=timeout,
            success_artifact_grace_sec=success_grace,
            terminate_grace_sec=terminate_grace,
        )
        report, artifact_failure_reasons = _load_isaaclab_success_report(
            report_path,
            output,
            min_mtime=float(result["started_wall_time"]),
        )
        output_exists = output.exists()
        output_bytes = output.stat().st_size if output_exists else 0
        artifacts_ok = not artifact_failure_reasons
        lifecycle_ok = (
            result["returncode"] == 0
            or (artifacts_ok and bool(result["terminated_after_success_artifacts"]))
            or (artifacts_ok and bool(result["timed_out"]))
        )
        status = "ok" if artifacts_ok and lifecycle_ok else "failed"
        failure_reasons: list[str] = []
        if not lifecycle_ok:
            if result["timed_out"]:
                failure_reasons.append("subprocess_timeout")
            else:
                failure_reasons.append(f"subprocess_returncode={result['returncode']}")
        failure_reasons.extend(artifact_failure_reasons)
        return {
            "status": status,
            "backend": ACCEPTANCE_G1_BACKEND,
            "command": command,
            "command_record": str(command_record),
            "robot_usd": str(self.g1_usd_path),
            "robot_usd_resolution": self.g1_usd_resolution,
            "output": str(output),
            "expected_output_path": str(output),
            "output_exists": bool(output_exists),
            "output_bytes": int(output_bytes),
            "missing_output": bool(not output_exists or output_bytes <= 0),
            "report": str(report_path) if report_path.exists() else "",
            "returncode": int(result["returncode"]),
            "stdout_tail": str(result["stdout_tail"])[-1000:],
            "stderr_tail": str(result["stderr_tail"])[-1000:],
            "subprocess_timed_out": bool(result["timed_out"]),
            "subprocess_elapsed_sec": float(result["elapsed_sec"]),
            "subprocess_timeout_sec": float(timeout),
            "terminated_after_success_artifacts": bool(result["terminated_after_success_artifacts"]),
            "overlays": list(ACCEPTANCE_OVERLAYS),
            "preserve_world_root": True,
            "isaaclab_report": report,
            "failure_reasons": failure_reasons,
            "message": "" if status == "ok" else "; ".join(failure_reasons),
        }

    def _visual_validation_float(self, key: str, default: float) -> float:
        visual_cfg = self._config.get("visual_validation", {})
        if not isinstance(visual_cfg, Mapping):
            return float(default)
        try:
            return float(visual_cfg.get(key, default))
        except (TypeError, ValueError):
            return float(default)

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


def _positive_float(value: float | None, default: float) -> float:
    try:
        parsed = float(default if value is None else value)
    except (TypeError, ValueError):
        parsed = float(default)
    return parsed if parsed > 0 else float(default)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_isaaclab_renderer_bounded(
    command: Sequence[str],
    *,
    cwd: Path | None,
    output_path: Path,
    report_path: Path,
    timeout_sec: float,
    success_artifact_grace_sec: float,
    terminate_grace_sec: float,
) -> dict[str, Any]:
    started = time.monotonic()
    started_wall = time.time()
    success_artifacts_seen_at: float | None = None
    timed_out = False
    terminated_after_success_artifacts = False
    returncode = -1
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stdout_file, tempfile.TemporaryFile(
        mode="w+",
        encoding="utf-8",
        errors="replace",
    ) as stderr_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
        )
        while True:
            polled = process.poll()
            if polled is not None:
                returncode = int(polled)
                break

            now = time.monotonic()
            if _isaaclab_success_artifacts_exist(report_path, output_path, min_mtime=started_wall):
                if success_artifacts_seen_at is None:
                    success_artifacts_seen_at = now
                elif now - success_artifacts_seen_at >= success_artifact_grace_sec:
                    terminated_after_success_artifacts = True
                    returncode = _terminate_process_group(process, terminate_grace_sec)
                    break

            if now - started >= timeout_sec:
                timed_out = True
                returncode = _terminate_process_group(process, terminate_grace_sec)
                break
            time.sleep(min(0.25, max(0.01, timeout_sec - (now - started))))

        elapsed = time.monotonic() - started
        stdout_tail = _text_file_tail(stdout_file, limit=1000)
        stderr_tail = _text_file_tail(stderr_file, limit=1000)
    return {
        "returncode": int(returncode),
        "timed_out": bool(timed_out),
        "terminated_after_success_artifacts": bool(terminated_after_success_artifacts),
        "elapsed_sec": float(elapsed),
        "started_wall_time": float(started_wall),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def _terminate_process_group(process: subprocess.Popen[str], terminate_grace_sec: float) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(process.wait())
    except OSError:
        process.terminate()
    try:
        return int(process.wait(timeout=terminate_grace_sec))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
        return int(process.wait(timeout=terminate_grace_sec))


def _text_file_tail(handle: Any, *, limit: int) -> str:
    handle.flush()
    handle.seek(0)
    return str(handle.read())[-limit:]


def _isaaclab_success_artifacts_exist(report_path: Path, output_path: Path, *, min_mtime: float | None = None) -> bool:
    _, failure_reasons = _load_isaaclab_success_report(report_path, output_path, min_mtime=min_mtime)
    return not failure_reasons


def _load_isaaclab_success_report(
    report_path: Path,
    output_path: Path,
    *,
    min_mtime: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    output_exists = output_path.exists()
    output_stat = output_path.stat() if output_exists else None
    output_bytes = output_stat.st_size if output_stat is not None else 0
    failure_reasons: list[str] = []
    if not output_exists or output_bytes <= 0:
        failure_reasons.append("expected_output_mp4_missing")
    elif min_mtime is not None and output_stat is not None and output_stat.st_mtime < min_mtime - 1.0:
        failure_reasons.append("expected_output_mp4_stale")
    if not report_path.exists():
        failure_reasons.append("renderer_report_missing")
        return {}, failure_reasons
    report_stat = report_path.stat()
    if min_mtime is not None and report_stat.st_mtime < min_mtime - 1.0:
        failure_reasons.append("renderer_report_stale")
    try:
        raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"report_parse_status": "failed", "message": str(exc)}, failure_reasons + ["renderer_report_bad_json"]
    if not isinstance(raw_report, dict):
        return {"report_parse_status": "failed"}, failure_reasons + ["renderer_report_bad_json"]
    report: dict[str, Any] = raw_report
    report_status = report.get("status")
    if report_status != "ok":
        failure_reasons.append(f"renderer_report_status={report_status}")
    renderer_failure_reasons = report.get("failure_reasons")
    if renderer_failure_reasons:
        failure_reasons.append("renderer_report_failure_reasons_present")
    return report, failure_reasons


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
