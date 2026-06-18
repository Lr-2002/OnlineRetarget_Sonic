"""Native-fps visual validation helpers for temporal DP periodic eval."""

from __future__ import annotations

import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data.g1_quality import g1_fk_body_positions, load_g1_kinematic_model
from .data.schema import MORPHOLOGY_NUMERIC_COLUMNS
from .data.sonic_windowed_builder import (
    SonicWindowedBuildConfig,
    _source_body_tokens,
    _source_features_from_bvh,
)
from .data.windowed_builder import DEFAULT_SOURCE_BODY_NAMES, global_body_position_maps_from_bvh, parse_bvh_motion
from .sonic_validation_callback import DEFAULT_LOG_PREFIX, _render_triplet_video
from .sonic_validation_export import (
    TRACKING_BODY_NAMES,
    clip_index_selected,
    load_raw_validation_trajectory,
    native_fps_review_evidence,
    parse_clip_indices,
    render_readable_validation_video,
    save_raw_validation_trajectory,
)


@dataclass(frozen=True)
class TemporalVisualValidationConfig:
    enabled: bool = False
    num_videos: int = 1
    duration_sec: float = 4.0
    output_dir: str = DEFAULT_LOG_PREFIX
    wandb_upload: bool = True
    readable_render: bool = True
    readable_clip_indices: tuple[int, ...] = (0, 6)
    model_xml: str | None = None
    source_bvh_tar: str | None = None
    source_bvh_cache: str | None = None
    source_bvh_roots: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "num_videos": self.num_videos,
            "duration_sec": self.duration_sec,
            "output_dir": self.output_dir,
            "wandb_upload": self.wandb_upload,
            "readable_render": self.readable_render,
            "readable_clip_indices": list(self.readable_clip_indices),
            "model_xml": self.model_xml or "",
            "source_bvh_tar": self.source_bvh_tar or "",
            "source_bvh_cache": self.source_bvh_cache or "",
            "source_bvh_roots": list(self.source_bvh_roots),
        }


def run_temporal_native_fps_visual_validation(
    *,
    torch,
    model: Any,
    config: Mapping[str, Any],
    visual_validation: TemporalVisualValidationConfig,
    samples: Sequence[Mapping[str, Any]],
    output_dir: Path,
    step: int,
    device: Any,
    wandb_run: Any = None,
) -> dict[str, Any]:
    """Roll out temporal-DP sequentially on the original frame timeline."""

    step_dir = _step_output_dir(output_dir, visual_validation=visual_validation, step=step)
    rank_dir = step_dir / "rank_000"
    rank_dir.mkdir(parents=True, exist_ok=True)
    summary_path = step_dir / "summary.json"

    if not visual_validation.enabled:
        summary = {
            "enabled": False,
            "status": "disabled",
            "step": int(step),
            "output_dir": str(step_dir),
            "summary_json": str(summary_path),
        }
        _write_json(summary_path, summary)
        return summary

    clip_limit = max(0, int(visual_validation.num_videos))
    selected_samples = [dict(sample) for sample in samples[:clip_limit]]
    if not selected_samples:
        summary = {
            "enabled": True,
            "status": "blocked",
            "message": "periodic eval provided no samples for native-fps visual validation",
            "step": int(step),
            "output_dir": str(step_dir),
            "summary_json": str(summary_path),
            "clips": [],
            "videos_ok": 0,
            "videos_failed": 0,
            "config": visual_validation.to_dict(),
        }
        _write_json(summary_path, summary)
        return summary

    model_cfg = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), Mapping) else {}
    target_horizon_frames = int(data_cfg.get("target_horizon_frames", model_cfg.get("target_horizon_frames", 1)) or 1)
    target_future_step = int(data_cfg.get("target_future_step", 1) or 1)
    source_cfg = SonicWindowedBuildConfig(
        history_frames=int(data_cfg.get("history_frames", target_horizon_frames) or target_horizon_frames),
        target_horizon_frames=target_horizon_frames,
        target_future_step=target_future_step,
        include_source_angular_velocity=bool(
            data_cfg.get("include_source_angular_velocity", data_cfg.get("build", {}).get("include_source_angular_velocity", True))
            if isinstance(data_cfg.get("build"), Mapping)
            else data_cfg.get("include_source_angular_velocity", True)
        ),
        position_scale=float(data_cfg.get("build", {}).get("position_scale", 0.01) if isinstance(data_cfg.get("build"), Mapping) else 0.01),
        root_body=str(data_cfg.get("build", {}).get("root_body", "Hips") if isinstance(data_cfg.get("build"), Mapping) else "Hips"),
        source_body_names=tuple(
            str(name)
            for name in data_cfg.get("source_body_names", DEFAULT_SOURCE_BODY_NAMES)
        ),
    )
    model_xml = _resolve_model_xml(config, visual_validation)
    g1_model = load_g1_kinematic_model(model_xml) if model_xml is not None else None
    if g1_model is None:
        summary = {
            "enabled": True,
            "status": "blocked",
            "message": "visual_validation.model_xml or visualization.capsule.model_xml is required",
            "step": int(step),
            "output_dir": str(step_dir),
            "summary_json": str(summary_path),
            "clips": [],
            "videos_ok": 0,
            "videos_failed": 0,
            "config": visual_validation.to_dict(),
        }
        _write_json(summary_path, summary)
        return summary

    clip_reports: list[dict[str, Any]] = []
    videos_ok = 0
    videos_failed = 0
    for clip_index, sample in enumerate(selected_samples):
        try:
            report = _rollout_one_clip(
                torch=torch,
                model=model,
                device=device,
                config=config,
                sample=sample,
                clip_index=clip_index,
                visual_validation=visual_validation,
                source_cfg=source_cfg,
                rank_dir=rank_dir,
                g1_model=g1_model,
            )
            if report.get("status") == "ok":
                videos_ok += 1
            else:
                videos_failed += 1
        except Exception as exc:  # noqa: BLE001
            report = {
                "clip_index": int(clip_index),
                "sample_id": str(sample.get("sample_id", "")),
                "status": "failed",
                "message": str(exc),
            }
            videos_failed += 1
        clip_reports.append(report)
        _write_json(rank_dir / f"clip_{clip_index:02d}_report.json", report)

    status = "ok" if clip_reports and videos_failed == 0 else "partial"
    if not clip_reports:
        status = "blocked"
    rank_report = {
        "status": status,
        "step": int(step),
        "rank": 0,
        "world_size": 1,
        "videos_ok": int(videos_ok),
        "videos_failed": int(videos_failed),
        "clips": clip_reports,
    }
    _write_json(rank_dir / "rank_report.json", rank_report)

    wandb_report = _upload_wandb(
        run=wandb_run,
        step=step,
        step_dir=step_dir,
        visual_validation=visual_validation,
    )
    summary = {
        "enabled": True,
        "status": rank_report["status"],
        "step": int(step),
        "output_dir": str(step_dir),
        "summary_json": str(summary_path),
        "rank_report_json": str(rank_dir / "rank_report.json"),
        "videos_ok": int(videos_ok),
        "videos_failed": int(videos_failed),
        "clips": clip_reports,
        "config": visual_validation.to_dict(),
        "wandb": wandb_report,
    }
    _write_json(summary_path, summary)
    return summary


def _rollout_one_clip(
    *,
    torch,
    model: Any,
    device: Any,
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
    clip_index: int,
    visual_validation: TemporalVisualValidationConfig,
    source_cfg: SonicWindowedBuildConfig,
    rank_dir: Path,
    g1_model: Any,
) -> dict[str, Any]:
    sample_id = _safe_name(str(sample.get("sample_id", f"clip_{clip_index:02d}")))
    target_path = _resolve_target_g1_path(sample, config)
    target_arrays = _load_npz_arrays(target_path)
    source_bvh = _resolve_source_bvh(sample, config, visual_validation, rank_dir)
    source_motion = parse_bvh_motion(source_bvh.read_text(encoding="utf-8"))
    source_features = _source_features_from_bvh(source_motion, config=source_cfg)
    source_global_maps = global_body_position_maps_from_bvh(
        source_motion,
        body_names=source_cfg.source_body_names,
        position_scale=source_cfg.position_scale,
    )
    source_fps = 1.0 / float(source_motion.frame_time) if source_motion.frame_time > 0 else float(sample.get("fps", 50.0))
    target_fps = float(np.asarray(target_arrays.get("fps", np.asarray([sample.get("fps", 50.0)]))).reshape(-1)[0])
    if target_fps <= 0:
        raise ValueError(f"target fps must be positive, got {target_fps}")
    frame_budget = max(1, int(round(float(visual_validation.duration_sec) * target_fps)))
    target_joint_pos = np.asarray(target_arrays["joint_pos"], dtype=np.float32)
    target_body_pos = np.asarray(target_arrays["body_pos_w"], dtype=np.float32)
    target_body_quat = np.asarray(target_arrays["body_quat_w"], dtype=np.float32)
    if target_joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must be [frames, joints], got {target_joint_pos.shape}")
    if target_body_pos.ndim != 3 or target_body_quat.ndim != 3:
        raise ValueError("body_pos_w/body_quat_w must be rank-3 arrays")

    start_target_index = int(sample.get("target_frame", 0) or 0)
    horizon = max(1, int(sample.get("target_horizon_frames", source_cfg.target_horizon_frames) or source_cfg.target_horizon_frames))
    future_step = max(1, int(sample.get("target_future_step", source_cfg.target_future_step) or source_cfg.target_future_step))
    morphology = _float_vector(sample.get("morphology"), width=len(MORPHOLOGY_NUMERIC_COLUMNS))
    robot_state_width = int(config.get("model", {}).get("robot_state_dim", 0) or 0) if isinstance(config.get("model"), Mapping) else 0
    robot_state = _float_vector(sample.get("robot_state"), width=robot_state_width) if robot_state_width > 0 else []
    action_dim = int(config.get("model", {}).get("action_dim", target_joint_pos.shape[1]) or target_joint_pos.shape[1]) if isinstance(config.get("model"), Mapping) else target_joint_pos.shape[1]
    prev_action = _float_vector(sample.get("prev_target_joints"), width=action_dim)
    if not prev_action:
        prev_action = [0.0] * action_dim

    target_indices: list[int] = []
    source_indices: list[int] = []
    predicted_joint_frames: list[np.ndarray] = []
    target_joint_frames: list[np.ndarray] = []
    source_soma_frames: list[np.ndarray] = []
    target_root_pos_frames: list[np.ndarray] = []
    target_root_quat_frames: list[np.ndarray] = []

    with torch.no_grad():
        for frame_offset in range(frame_budget):
            target_index = start_target_index + frame_offset
            future_source_indices = _future_source_indices(
                target_index=target_index,
                horizon=horizon,
                future_step=future_step,
                source_fps=source_fps,
                target_fps=target_fps,
            )
            if target_index >= target_joint_pos.shape[0]:
                break
            if not future_source_indices:
                break
            if future_source_indices[-1] >= len(source_features.positions):
                break
            current_source_index = future_source_indices[0]
            if current_source_index >= len(source_global_maps):
                break

            condition = {
                "source_body_tokens": torch.as_tensor(
                    [_source_body_tokens(source_features, future_source_indices)],
                    device=device,
                    dtype=torch.float32,
                ),
                "source_skeleton": torch.as_tensor([source_features.skeleton], device=device, dtype=torch.float32),
                "morphology": torch.as_tensor([morphology], device=device, dtype=torch.float32),
                "prev_action": torch.as_tensor([prev_action], device=device, dtype=torch.float32),
            }
            if robot_state:
                condition["robot_state"] = torch.as_tensor([robot_state], device=device, dtype=torch.float32)

            predicted = model.sample(
                condition["source_body_tokens"],
                source_skeleton=condition.get("source_skeleton"),
                morphology=condition.get("morphology"),
                robot_state=condition.get("robot_state"),
                prev_action=condition.get("prev_action"),
                steps=int(config.get("model", {}).get("inference_steps", config.get("model", {}).get("diffusion_steps", 32)) if isinstance(config.get("model"), Mapping) else 32),
                start=str(config.get("model", {}).get("inference_start", "zeros") if isinstance(config.get("model"), Mapping) else "zeros"),
            )
            current_prediction = np.asarray(predicted[0, 0].detach().cpu().numpy(), dtype=np.float32)
            prev_action = [float(value) for value in current_prediction[:action_dim]]

            target_indices.append(int(target_index))
            source_indices.append(int(current_source_index))
            predicted_joint_frames.append(current_prediction[:action_dim])
            target_joint_frames.append(target_joint_pos[target_index, :action_dim].astype(np.float32, copy=False))
            source_frame_map = source_global_maps[current_source_index]
            source_soma_frames.append(
                np.asarray(
                    [
                        source_frame_map.get(name, (0.0, 0.0, 0.0))
                        for name in source_cfg.source_body_names
                    ],
                    dtype=np.float32,
                )
            )
            target_root_pos_frames.append(target_body_pos[target_index, 0, :].astype(np.float32, copy=False))
            target_root_quat_frames.append(target_body_quat[target_index, 0, :].astype(np.float32, copy=False))

    if not predicted_joint_frames:
        raise RuntimeError("native-fps rollout produced no frames")

    pred_joint_pos = np.asarray(predicted_joint_frames, dtype=np.float32)
    target_joint_frames_arr = np.asarray(target_joint_frames, dtype=np.float32)
    source_soma = np.asarray(source_soma_frames, dtype=np.float32)
    target_root_pos = np.asarray(target_root_pos_frames, dtype=np.float32)
    target_root_quat = np.asarray(target_root_quat_frames, dtype=np.float32)
    pred_root_pos = target_root_pos.copy()
    pred_root_quat = target_root_quat.copy()

    target_g1 = target_body_pos[np.asarray(target_indices, dtype=np.int64), :, :].astype(np.float32, copy=False)
    inferred_g1 = np.asarray(
        _g1_tracking_body_frames(
            pred_joint_pos,
            root_pos=pred_root_pos,
            root_quat=pred_root_quat,
            g1_model=g1_model,
        ),
        dtype=np.float32,
    )

    trajectory = {
        "clip_index": int(clip_index),
        "local_env_index": int(clip_index),
        "motion_id": str(sample.get("sample_id") or sample_id),
        "motion_key": str(sample.get("target_g1_path") or sample.get("sample_id") or sample_id),
        "source_soma": source_soma,
        "target_g1": _tracking_subset(target_g1),
        "inferred_g1": inferred_g1,
        "target_root_pos_w": target_root_pos,
        "target_root_rot_w": target_root_quat,
        "pred_root_pos_w": pred_root_pos,
        "pred_root_rot_w": pred_root_quat,
        "source_frame_indices": [int(index) for index in source_indices],
        "encoder_routes": [],
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
        "physical_time_aligned": True,
        "root_rot_format": "wxyz",
        "initial_root_xy_zeroed": False,
        "source_soma_names": tuple(str(name) for name in source_cfg.source_body_names),
        "g1_body_names": TRACKING_BODY_NAMES,
    }

    triplet_path = rank_dir / f"clip_{clip_index:02d}_{sample_id}.mp4"
    readable_path = rank_dir / f"clip_{clip_index:02d}_{sample_id}_readable.mp4"
    raw_path = rank_dir / f"clip_{clip_index:02d}_{sample_id}_trajectory.npz"
    raw_report = save_raw_validation_trajectory(
        trajectory=trajectory,
        output_path=raw_path,
        target_fps=target_fps,
        duration_sec=len(predicted_joint_frames) / target_fps,
    )
    _render_triplet_video(
        trajectory=trajectory,
        video_path=triplet_path,
        target_fps=target_fps,
        duration_sec=len(predicted_joint_frames) / target_fps,
    )
    readable_report: dict[str, Any] = {"status": "skipped"}
    if visual_validation.readable_render and clip_index_selected(clip_index, visual_validation.readable_clip_indices):
        readable_report = render_readable_validation_video(
            trajectory=trajectory,
            video_path=readable_path,
            target_fps=target_fps,
            duration_sec=len(predicted_joint_frames) / target_fps,
        )
    loaded = load_raw_validation_trajectory(raw_path)
    review_contract = native_fps_review_evidence(loaded)
    return {
        "clip_index": int(clip_index),
        "sample_id": str(sample.get("sample_id", "")),
        "status": "ok",
        "message": "native-fps temporal rollout validation clip rendered",
        "target_g1_path": str(target_path),
        "source_bvh": str(source_bvh),
        "triplet_video_path": str(triplet_path),
        "readable_video_path": str(readable_path) if readable_path.exists() else "",
        "raw_trajectory_path": str(raw_path),
        "raw_trajectory_metadata_path": str(raw_path.with_suffix(".json")),
        "target_frame_range": [int(target_indices[0]), int(target_indices[-1])],
        "prediction_root_pose_source": "target_root_pose_reused",
        "raw_report": raw_report,
        "readable_report": readable_report,
        "review_contract": review_contract,
        "fps": review_contract["fps"],
        "frame_count": review_contract["frame_count"],
        "source_frame_range": review_contract["source_frame_range"],
        "physical_time_aligned": review_contract["physical_time_aligned"],
    }


def _future_source_indices(
    *,
    target_index: int,
    horizon: int,
    future_step: int,
    source_fps: float,
    target_fps: float,
) -> list[int]:
    indices: list[int] = []
    for offset in range(max(1, int(horizon))):
        future_target_index = int(target_index) + offset * int(future_step)
        time_sec = future_target_index / float(target_fps)
        indices.append(max(0, int(math.floor(time_sec * float(source_fps) + 1e-6))))
    return indices


def _resolve_model_xml(
    config: Mapping[str, Any],
    visual_validation: TemporalVisualValidationConfig,
) -> Path | None:
    if visual_validation.model_xml:
        path = Path(str(visual_validation.model_xml)).expanduser()
        return path if path.exists() else None
    visualization = config.get("visualization", {}) if isinstance(config.get("visualization"), Mapping) else {}
    capsule = visualization.get("capsule", {}) if isinstance(visualization.get("capsule"), Mapping) else {}
    model_xml = capsule.get("model_xml")
    if not model_xml:
        return None
    path = Path(str(model_xml)).expanduser()
    return path if path.exists() else None


def _resolve_target_g1_path(sample: Mapping[str, Any], config: Mapping[str, Any]) -> Path:
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), Mapping) else {}
    raw = str(sample.get("target_g1_path") or sample.get("sonic_relative_path") or "")
    if not raw:
        raise ValueError("sample lacks target_g1_path")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate
    sonic_root = data_cfg.get("sonic_npz_root")
    if sonic_root:
        rooted = Path(str(sonic_root)).expanduser() / raw
        if rooted.exists():
            return rooted
    prefix_from = data_cfg.get("sonic_path_prefix_from")
    prefix_to = data_cfg.get("sonic_path_prefix_to")
    if prefix_from and prefix_to and raw.startswith(str(prefix_from)):
        rewritten = Path(str(prefix_to) + raw[len(str(prefix_from)) :]).expanduser()
        if rewritten.exists():
            return rewritten
    raise FileNotFoundError(f"could not resolve target_g1_path={raw!r}")


def _resolve_source_bvh(
    sample: Mapping[str, Any],
    config: Mapping[str, Any],
    visual_validation: TemporalVisualValidationConfig,
    rank_dir: Path,
) -> Path:
    raw = str(sample.get("source_motion_path") or "")
    if not raw:
        raise ValueError("sample lacks source_motion_path")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    roots = [Path(text).expanduser() for text in visual_validation.source_bvh_roots]
    for root in roots:
        rooted = root / raw
        if rooted.exists():
            return rooted
        if raw.startswith("soma_proportional/"):
            stripped = root / raw[len("soma_proportional/") :]
            if stripped.exists():
                return stripped
    tar_path_text = visual_validation.source_bvh_tar or str(
        config.get("data", {}).get("source_bvh_tar", "") if isinstance(config.get("data"), Mapping) else ""
    )
    if not tar_path_text:
        raise FileNotFoundError(f"could not resolve source BVH and no tar configured for {raw!r}")
    tar_path = Path(str(tar_path_text)).expanduser()
    if not tar_path.exists():
        raise FileNotFoundError(f"source_bvh_tar does not exist: {tar_path}")
    cache_root = Path(
        str(visual_validation.source_bvh_cache or (rank_dir.parent.parent / "source_bvh_cache"))
    ).expanduser()
    return _extract_source_bvh_from_tar(tar_path, raw, cache_root)


def _extract_source_bvh_from_tar(tar_path: Path, member_name: str, cache_root: Path) -> Path:
    safe_member = Path(member_name)
    if safe_member.is_absolute() or ".." in safe_member.parts:
        raise ValueError(f"unsafe source_motion_path inside tar: {member_name!r}")
    out_path = cache_root / safe_member
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        extracted = tar.extractfile(member_name)
        if extracted is None:
            if member_name.startswith("soma_proportional/"):
                extracted = tar.extractfile(member_name[len("soma_proportional/") :])
            if extracted is None:
                raise FileNotFoundError(f"{member_name!r} not present in {tar_path}")
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with extracted, tmp_path.open("wb") as handle:
            handle.write(extracted.read())
        tmp_path.replace(out_path)
    return out_path


def _tracking_subset(body_pos_w: np.ndarray) -> np.ndarray:
    indices = []
    for name in TRACKING_BODY_NAMES:
        if name not in DEFAULT_TRACKING_BODY_INDEX:
            raise ValueError(f"tracking body name is unknown: {name}")
        indices.append(DEFAULT_TRACKING_BODY_INDEX[name])
    if body_pos_w.shape[1] <= max(indices):
        raise ValueError(f"body_pos_w body axis {body_pos_w.shape[1]} is too small for tracking subset")
    return body_pos_w[:, indices, :]


def _g1_tracking_body_frames(
    joint_pos: np.ndarray,
    *,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    g1_model: Any,
) -> list[list[list[float]]]:
    frames: list[list[list[float]]] = []
    for joints, root, quat in zip(joint_pos, root_pos, root_quat, strict=False):
        body_points = g1_fk_body_positions(
            g1_model,
            [float(value) for value in joints],
            root_position=[float(value) for value in root],
            root_euler=_quat_wxyz_to_euler_xyz(quat),
            include_empty_body_origin=True,
        )
        frame: list[list[float]] = []
        for name in TRACKING_BODY_NAMES:
            points = body_points.get(name, ())
            if not points:
                raise ValueError(f"G1 FK output is missing tracking body {name!r}")
            frame.append([float(value) for value in _centroid(points)])
        frames.append(frame)
    return frames


def _quat_wxyz_to_euler_xyz(quat_wxyz: Sequence[float]) -> list[float]:
    w, x, y, z = (float(quat_wxyz[index]) if index < len(quat_wxyz) else 0.0 for index in range(4))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def _centroid(points: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    count = max(1, len(points))
    return tuple(
        sum(float(point[axis]) for point in points) / count
        for axis in range(3)
    )


def _float_vector(value: Any, *, width: int) -> list[float]:
    if width <= 0:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        values = [float(item) for item in value[:width]]
        if len(values) < width:
            values.extend([0.0] * (width - len(values)))
        return values
    return [0.0] * width


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def _step_output_dir(
    output_dir: Path,
    *,
    visual_validation: TemporalVisualValidationConfig,
    step: int,
) -> Path:
    root = Path(str(visual_validation.output_dir)).expanduser()
    if not root.is_absolute():
        root = output_dir / root
    return root / f"step_{int(step):08d}"


def _upload_wandb(
    *,
    run: Any,
    step: int,
    step_dir: Path,
    visual_validation: TemporalVisualValidationConfig,
) -> dict[str, Any]:
    if run is None or not visual_validation.wandb_upload:
        return {"status": "skipped", "message": "wandb disabled or no active run"}
    try:
        import wandb
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "message": f"wandb import failed: {exc}"}

    payload: dict[str, Any] = {}
    triplet_videos = [path for path in sorted(step_dir.glob("rank_*/clip_*.mp4")) if not path.stem.endswith("_readable")]
    readable_videos = [path for path in sorted(step_dir.glob("rank_*/clip_*_readable.mp4"))]
    for video_path in triplet_videos:
        key = f"{DEFAULT_LOG_PREFIX}/{video_path.parent.name}_{video_path.stem}"
        payload[key] = wandb.Video(str(video_path), fps=50, format="mp4")
    for video_path in readable_videos:
        key = f"{DEFAULT_LOG_PREFIX}_readable/{video_path.parent.name}_{video_path.stem}"
        payload[key] = wandb.Video(str(video_path), fps=50, format="mp4")
    summary_path = step_dir / "summary.json"
    if summary_path.exists():
        try:
            payload[f"{DEFAULT_LOG_PREFIX}/summary"] = wandb.Html(
                f"<pre>{summary_path.read_text(encoding='utf-8')}</pre>"
            )
        except Exception:
            pass
    if not payload:
        return {"status": "skipped", "message": "no validation videos found"}
    payload[f"{DEFAULT_LOG_PREFIX}/videos_uploaded"] = len(triplet_videos)
    payload[f"{DEFAULT_LOG_PREFIX}/readable_videos_uploaded"] = len(readable_videos)
    run.log(payload, step=step)
    for path in [summary_path, *step_dir.glob("rank_*/clip_*_trajectory.npz"), *step_dir.glob("rank_*/clip_*_trajectory.json")]:
        if path.exists():
            run.save(str(path))
    return {
        "status": "ok",
        "videos_uploaded": len(triplet_videos),
        "readable_videos_uploaded": len(readable_videos),
    }


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"_", "-", "."} else "_" for character in value)[:120] or "sample"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


DEFAULT_TRACKING_BODY_INDEX = {
    "pelvis": 0,
    "left_hip_roll_link": 4,
    "left_knee_link": 10,
    "left_ankle_roll_link": 18,
    "right_hip_roll_link": 5,
    "right_knee_link": 11,
    "right_ankle_roll_link": 19,
    "torso_link": 9,
    "left_shoulder_roll_link": 16,
    "left_elbow_link": 22,
    "left_wrist_yaw_link": 28,
    "right_shoulder_roll_link": 17,
    "right_elbow_link": 23,
    "right_wrist_yaw_link": 29,
}
