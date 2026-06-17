#!/usr/bin/env python3
"""Bridge LR-310 DP predictions into the accepted_vertical_v2 visual backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from online_retarget.a0_visual_validation import (  # noqa: E402
    ACCEPTANCE_G1_BACKEND,
    ACCEPTANCE_ROW2_DATA_SOURCE,
    ACCEPTANCE_ROW2_ROLE,
    ACCEPTANCE_ROW3_DATA_SOURCE,
    ACCEPTANCE_ROW3_ROLE,
    A0VisualValidationRenderer,
    accepted_vertical_v2_artifact_paths,
    build_accepted_vertical_v2_metadata,
)
from online_retarget.data.bones_sonic import SONIC_JOINT_NAMES  # noqa: E402
from online_retarget.data.windowed_builder import (  # noqa: E402
    DEFAULT_SOURCE_BODY_NAMES,
    global_body_position_maps_from_bvh,
    parse_bvh_motion,
)
from online_retarget.sonic_validation_export import (  # noqa: E402
    TRACKING_BODY_NAMES,
    native_fps_review_evidence,
    save_raw_validation_trajectory,
)


SourceBvhResolver = Callable[[Mapping[str, Any], dict[str, Any], Path], Path | None]
SourceRenderer = Callable[..., dict[str, Any]]
PanelCombiner = Callable[..., dict[str, Any]]


def read_prediction_rows(path: Path, *, count: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
            if count is not None and len(rows) >= count:
                break
    return rows


def normalize_prediction_row_for_source_bvh(row: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt DP prediction schema to the LR-290 source-BVH resolver input."""

    normalized = dict(row)
    if not normalized.get("source_soma_proportional_path"):
        source_motion = str(normalized.get("source_motion_path") or "")
        if source_motion:
            normalized["source_soma_proportional_path"] = source_motion
    return normalized


def resolve_target_g1_path(row: Mapping[str, Any], roots: Sequence[Path]) -> Path:
    raw = str(row.get("target_g1_path") or "")
    if not raw:
        raise ValueError("prediction row lacks target_g1_path")
    candidate = Path(raw)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"target_g1_path does not exist: {candidate}")
    if candidate.exists():
        return candidate
    for root in roots:
        rooted = root / raw
        if rooted.exists():
            return rooted
    roots_text = ", ".join(str(root) for root in roots) or "<none>"
    raise FileNotFoundError(f"could not resolve relative target_g1_path={raw!r}; roots={roots_text}")


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def prediction_joint_sequence(row: Mapping[str, Any], key: str) -> np.ndarray:
    values = row.get(key)
    if values is None:
        raise ValueError(f"prediction row lacks {key}")
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{key} must be a non-empty 2D joint sequence")
    if not np.isfinite(array).all():
        raise ValueError(f"{key} contains non-finite values")
    return array


def target_joint_sequence(
    row: Mapping[str, Any],
    target_arrays: Mapping[str, np.ndarray],
    *,
    frame_indices: np.ndarray,
) -> np.ndarray:
    if "joint_pos" in target_arrays:
        array = _select_frames(
            np.asarray(target_arrays["joint_pos"], dtype=np.float32),
            frame_indices,
            name="joint_pos",
        )
        if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError("target_g1_path joint_pos must be a non-empty 2D array")
        return array
    if row.get("target_joints") is None:
        raise ValueError("target_joints missing from row and joint_pos missing from target_g1_path NPZ")
    array = prediction_joint_sequence(row, "target_joints")
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("target_joints must be a non-empty 2D joint sequence")
    if array.shape[0] != len(frame_indices):
        raise ValueError(
            "target_joints length must match target frame indices "
            f"({array.shape[0]} vs {len(frame_indices)})"
        )
    return array


def fps_from_row_or_npz(row: Mapping[str, Any], target_arrays: Mapping[str, np.ndarray], *, default: float = 50.0) -> float:
    value = row.get("fps", None)
    if value is None and "fps" in target_arrays:
        value = np.asarray(target_arrays["fps"]).reshape(-1)[0]
    fps = float(default if value is None else value)
    if fps <= 0.0 or not math.isfinite(fps):
        raise ValueError(f"fps must be a positive finite value, got {fps!r}")
    return fps


def joint_names_from_row(row: Mapping[str, Any], joint_dim: int) -> list[str]:
    names = row.get("target_joint_names")
    if isinstance(names, list) and len(names) == joint_dim:
        return [str(name) for name in names]
    if joint_dim == len(SONIC_JOINT_NAMES):
        return list(SONIC_JOINT_NAMES)
    return [f"joint_{index}" for index in range(joint_dim)]


def target_frame_indices(row: Mapping[str, Any], *, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    explicit = row.get("target_frame_indices")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
            raise ValueError("target_frame_indices must be a sequence of integer frame indices")
        indices = [_coerce_frame_index(value, key="target_frame_indices") for value in explicit]
        if len(indices) != horizon:
            raise ValueError(
                "target_frame_indices length must match predicted_joints horizon "
                f"({len(indices)} vs {horizon})"
            )
    else:
        start = _coerce_frame_index(row.get("target_frame", 0), key="target_frame")
        indices = [start + offset for offset in range(horizon)]
    result = np.asarray(indices, dtype=np.int64)
    if np.any(result < 0):
        raise ValueError(f"target frame indices must be non-negative, got {indices}")
    return result


def _coerce_frame_index(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} must contain integer frame indices, got bool")
    try:
        number = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{key} must contain integer frame indices, got {value!r}") from None
    if not math.isfinite(numeric) or number != numeric:
        raise ValueError(f"{key} must contain integer frame indices, got {value!r}")
    return number


def root_pose_from_prediction_row(
    row: Mapping[str, Any],
    *,
    frame_count: int,
    pos_key: str,
    quat_key: str,
    root_quat_format: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if row.get(pos_key) is None or row.get(quat_key) is None:
        return None
    root_pos = _as_2d_float(row[pos_key], width=3, name=pos_key)
    root_quat = _quat_to_wxyz(_as_2d_float(row[quat_key], width=4, name=quat_key), root_quat_format)
    if min(root_pos.shape[0], root_quat.shape[0]) <= 0:
        return None
    return root_pos[:frame_count], root_quat[:frame_count]


def root_pose_from_target_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    frame_indices: np.ndarray,
    root_source: str,
    root_body_index: int,
    root_body_name: str,
    root_quat_format: str,
    allow_root_fixed_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source = root_source
    tried: list[str] = []
    if source == "auto":
        candidates = (
            ("target_npz_root", ("root_pos", "root_quat")),
            ("motionlib_root", ("root_trans_offset", "root_rot")),
            ("body_root", ("body_pos_w", "body_quat_w")),
        )
        for candidate, required_keys in candidates:
            missing = [key for key in required_keys if key not in arrays]
            if missing:
                tried.append(f"{candidate}: missing {', '.join(missing)}")
                continue
            return root_pose_from_target_arrays(
                arrays,
                frame_indices=frame_indices,
                root_source=candidate,
                root_body_index=root_body_index,
                root_body_name=root_body_name,
                root_quat_format=root_quat_format,
                allow_root_fixed_fallback=False,
            )
        if allow_root_fixed_fallback:
            source = "root_fixed"
        else:
            raise ValueError(
                "target motion lacks explicit root pose; pass --root-source root_fixed or "
                "--allow-root-fixed-fallback to mark a fixed-root artifact. Tried: "
                + "; ".join(tried)
            )

    if source == "target_npz_root":
        if "root_pos" not in arrays or "root_quat" not in arrays:
            raise ValueError("target_npz_root requires root_pos and root_quat keys")
        root_pos = _select_frames(
            _as_2d_float(arrays["root_pos"], width=3, name="root_pos"),
            frame_indices,
            name="root_pos",
        )
        root_quat = _quat_to_wxyz(
            _select_frames(
                _as_2d_float(arrays["root_quat"], width=4, name="root_quat"),
                frame_indices,
                name="root_quat",
            ),
            root_quat_format,
        )
        semantics = {
            "root_pose_source": "target_npz_root",
            "root_pos_key": "root_pos",
            "root_quat_key": "root_quat",
            "root_quat_format": "wxyz",
            "root_fixed_fallback": False,
            "root_semantics": "world-space root_pos/root_quat selected from target_g1_path NPZ by target frame indices",
        }
        return root_pos, root_quat, semantics

    if source == "motionlib_root":
        if "root_trans_offset" not in arrays or "root_rot" not in arrays:
            raise ValueError("motionlib_root requires root_trans_offset and root_rot keys")
        root_pos = _select_frames(
            _as_2d_float(arrays["root_trans_offset"], width=3, name="root_trans_offset"),
            frame_indices,
            name="root_trans_offset",
        )
        root_quat = _quat_to_wxyz(
            _select_frames(
                _as_2d_float(arrays["root_rot"], width=4, name="root_rot"),
                frame_indices,
                name="root_rot",
            ),
            root_quat_format,
        )
        semantics = {
            "root_pose_source": "motionlib_root",
            "root_pos_key": "root_trans_offset",
            "root_quat_key": "root_rot",
            "root_quat_format": root_quat_format,
            "root_fixed_fallback": False,
            "root_semantics": "world-space motionlib root_trans_offset/root_rot selected from target_g1_path NPZ by target frame indices",
        }
        return root_pos, root_quat, semantics

    if source == "body_root":
        if "body_pos_w" not in arrays or "body_quat_w" not in arrays:
            raise ValueError("body_root requires body_pos_w and body_quat_w keys")
        body_pos = np.asarray(arrays["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(arrays["body_quat_w"], dtype=np.float32)
        if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
            raise ValueError("body_pos_w must have shape [frames, bodies, 3]")
        if body_quat.ndim != 3 or body_quat.shape[-1] != 4:
            raise ValueError("body_quat_w must have shape [frames, bodies, 4]")
        if root_body_index < 0 or root_body_index >= body_pos.shape[1] or root_body_index >= body_quat.shape[1]:
            raise ValueError(f"root_body_index {root_body_index} outside body_pos_w/body_quat_w body axis")
        selected_body_pos = _select_frames(body_pos, frame_indices, name="body_pos_w")
        selected_body_quat = _select_frames(body_quat, frame_indices, name="body_quat_w")
        root_pos = selected_body_pos[:, root_body_index, :]
        root_quat = _quat_to_wxyz(selected_body_quat[:, root_body_index, :], root_quat_format)
        semantics = {
            "root_pose_source": "body_root",
            "root_pos_key": f"body_pos_w[:, {root_body_index}, :]",
            "root_quat_key": f"body_quat_w[:, {root_body_index}, :]",
            "root_body_index": int(root_body_index),
            "root_body_name": str(root_body_name),
            "root_quat_format": root_quat_format,
            "root_fixed_fallback": False,
            "root_semantics": "world-space G1 root body pose selected from target_g1_path body_pos_w/body_quat_w by target frame indices",
        }
        return root_pos, root_quat, semantics

    if source == "root_fixed":
        root_pos = np.zeros((int(len(frame_indices)), 3), dtype=np.float32)
        root_quat = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (int(len(frame_indices)), 1))
        semantics = {
            "root_pose_source": "root_fixed",
            "root_pos_key": "",
            "root_quat_key": "",
            "root_fixed_fallback": True,
            "root_semantics": "explicit fixed-root fallback; not LR-290 world-root-preserved isomorphism",
        }
        return root_pos, root_quat, semantics

    raise ValueError(f"unsupported root_source: {root_source}")


def write_dp_motion_assets(
    *,
    renderer: A0VisualValidationRenderer,
    paths: Mapping[str, Path],
    row: Mapping[str, Any],
    target_arrays: Mapping[str, np.ndarray],
    fps: float,
    root_source: str,
    root_body_index: int,
    root_body_name: str,
    root_quat_format: str,
    allow_root_fixed_fallback: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    predicted_joints = prediction_joint_sequence(row, "predicted_joints")
    frame_indices = target_frame_indices(row, horizon=predicted_joints.shape[0])
    target_joints = target_joint_sequence(row, target_arrays, frame_indices=frame_indices)
    frame_count = min(predicted_joints.shape[0], target_joints.shape[0], len(frame_indices))
    if frame_count <= 0:
        raise ValueError("cannot write zero-frame DP visual validation assets")
    target_root_pos, target_root_quat, root_semantics = root_pose_from_target_arrays(
        target_arrays,
        frame_indices=frame_indices[:frame_count],
        root_source=root_source,
        root_body_index=root_body_index,
        root_body_name=root_body_name,
        root_quat_format=root_quat_format,
        allow_root_fixed_fallback=allow_root_fixed_fallback,
    )
    frame_count = min(frame_count, target_root_pos.shape[0], target_root_quat.shape[0])
    predicted_root = root_pose_from_prediction_row(
        row,
        frame_count=frame_count,
        pos_key="pred_root_pos_w",
        quat_key="pred_root_quat_w",
        root_quat_format=root_quat_format,
    )
    if predicted_root is None:
        predicted_root_pos = target_root_pos[:frame_count]
        predicted_root_quat = target_root_quat[:frame_count]
        prediction_root_semantics = {
            "prediction_root_pose_source": "target_root_pose_reused",
            "prediction_root_fixed_fallback": bool(root_semantics["root_fixed_fallback"]),
            "prediction_root_semantics": (
                "DP predictions.jsonl row has joint trajectories only; row3 root_pos/root_quat "
                "reuse row2 target root pose and this is recorded as bridge metadata"
            ),
        }
    else:
        predicted_root_pos, predicted_root_quat = predicted_root
        frame_count = min(frame_count, predicted_root_pos.shape[0], predicted_root_quat.shape[0])
        prediction_root_semantics = {
            "prediction_root_pose_source": "predictions_jsonl",
            "prediction_root_pos_key": "pred_root_pos_w",
            "prediction_root_quat_key": "pred_root_quat_w",
            "prediction_root_fixed_fallback": False,
            "prediction_root_semantics": "world-space prediction root pose supplied by predictions.jsonl row",
        }

    joint_names = joint_names_from_row(row, predicted_joints.shape[1])
    target_report = renderer.write_g1_motion_npz(
        path=paths["row2_motion_npz"],
        joint_pos=target_joints[:frame_count],
        root_pos=target_root_pos[:frame_count],
        root_quat=target_root_quat[:frame_count],
        fps=fps,
        joint_names=joint_names_from_row(row, target_joints.shape[1]),
    )
    target_report.update(
        {
            "data_source": ACCEPTANCE_ROW2_DATA_SOURCE,
            "row_role": ACCEPTANCE_ROW2_ROLE,
            **root_semantics,
        }
    )
    prediction_report = renderer.write_g1_motion_npz(
        path=paths["row3_motion_npz"],
        joint_pos=predicted_joints[:frame_count],
        root_pos=predicted_root_pos[:frame_count],
        root_quat=predicted_root_quat[:frame_count],
        fps=fps,
        joint_names=joint_names,
    )
    prediction_report.update(
        {
            "data_source": ACCEPTANCE_ROW3_DATA_SOURCE,
            "row_role": ACCEPTANCE_ROW3_ROLE,
            **root_semantics,
            **prediction_root_semantics,
        }
    )
    bridge_metadata = {
        "schema": "lr310_dp_predictions_jsonl_to_accepted_vertical_v2",
        "target_frames": int(frame_count),
        "target_frame_indices": [int(index) for index in frame_indices[:frame_count].tolist()],
        "predicted_joint_dim": int(predicted_joints.shape[1]),
        "target_joint_dim": int(target_joints.shape[1]),
        "root_pose": dict(root_semantics),
        "prediction_root_pose": dict(prediction_root_semantics),
        "root_fixed_fallback": bool(
            root_semantics["root_fixed_fallback"] or prediction_root_semantics["prediction_root_fixed_fallback"]
        ),
    }
    return target_report, prediction_report, bridge_metadata


def _load_source_soma_tracking_frames(
    *,
    source_bvh: Path | None,
    frame_indices: Sequence[int],
) -> tuple[np.ndarray | None, list[str] | None]:
    if source_bvh is None:
        return None, None
    text = source_bvh.read_text(encoding="utf-8")
    max_frame = max(int(index) for index in frame_indices) if frame_indices else -1
    try:
        motion = parse_bvh_motion(text, max_frames=max_frame + 1 if max_frame >= 0 else 1)
    except ValueError:
        return None, None
    maps = global_body_position_maps_from_bvh(
        motion,
        body_names=DEFAULT_SOURCE_BODY_NAMES,
        position_scale=0.01,
    )
    selected: list[np.ndarray] = []
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= len(maps):
            raise ValueError(
                f"source BVH lacks frame {frame_index} needed by target_frame_indices={list(frame_indices)}"
            )
        frame_map = maps[frame_index]
        frame_points: list[list[float]] = []
        for name in DEFAULT_SOURCE_BODY_NAMES:
            point = frame_map.get(name, (0.0, 0.0, 0.0))
            frame_points.append([float(point[0]), float(point[1]), float(point[2])])
        selected.append(np.asarray(frame_points, dtype=np.float32))
    if not selected:
        return np.zeros((0, len(DEFAULT_SOURCE_BODY_NAMES), 3), dtype=np.float32), list(DEFAULT_SOURCE_BODY_NAMES)
    return np.stack(selected, axis=0), list(DEFAULT_SOURCE_BODY_NAMES)


def _g1_tracking_frames_from_joint_root(
    *,
    joint_pos: np.ndarray,
    root_pos: np.ndarray,
) -> np.ndarray:
    frames = int(min(len(joint_pos), len(root_pos)))
    body_count = len(TRACKING_BODY_NAMES)
    tracked = np.zeros((frames, body_count, 3), dtype=np.float32)
    tracked[:, 0, :] = np.asarray(root_pos[:frames], dtype=np.float32)
    usable_joint_dim = min(joint_pos.shape[1], body_count - 1) if joint_pos.ndim == 2 else 0
    if usable_joint_dim > 0:
        tracked[:, 1 : usable_joint_dim + 1, 0] = np.asarray(joint_pos[:frames, :usable_joint_dim], dtype=np.float32)
    return tracked


def _load_motion_asset_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        joint_pos = np.asarray(loaded["joint_pos"], dtype=np.float32)
        root_pos = np.asarray(loaded["root_pos"], dtype=np.float32)
        root_quat = np.asarray(loaded["root_quat"], dtype=np.float32)
    return joint_pos, root_pos, root_quat


def rerender_prediction_row(
    *,
    row: Mapping[str, Any],
    index: int,
    predictions_jsonl: Path,
    output_dir: Path,
    config: dict[str, Any],
    target_g1_roots: Sequence[Path],
    step: int,
    execute_renderers: bool,
    root_source: str,
    root_body_index: int,
    root_body_name: str,
    root_quat_format: str,
    allow_root_fixed_fallback: bool,
    checkpoint_path: Path | None = None,
    checkpoint_step: int | None = None,
    skip_source_bvh_resolve: bool = False,
    source_bvh_resolver: SourceBvhResolver | None = None,
    source_renderer: SourceRenderer | None = None,
    panel_combiner: PanelCombiner | None = None,
) -> dict[str, Any]:
    step_dir = output_dir / f"step_{int(step):08d}"
    sample_id = safe_sample_id(row, index)
    paths = accepted_vertical_v2_artifact_paths(step_dir, sample_id=sample_id, step=step)
    paths["artifact_dir"].mkdir(parents=True, exist_ok=True)

    target_path = resolve_target_g1_path(row, target_g1_roots)
    target_arrays = load_npz_arrays(target_path)
    fps = fps_from_row_or_npz(row, target_arrays)
    renderer = A0VisualValidationRenderer(config)
    target_motion_report, prediction_motion_report, bridge_metadata = write_dp_motion_assets(
        renderer=renderer,
        paths=paths,
        row=row,
        target_arrays=target_arrays,
        fps=fps,
        root_source=root_source,
        root_body_index=root_body_index,
        root_body_name=root_body_name,
        root_quat_format=root_quat_format,
        allow_root_fixed_fallback=allow_root_fixed_fallback,
    )
    frame_count = int(min(target_motion_report["frames"], prediction_motion_report["frames"]))

    if skip_source_bvh_resolve:
        source_bvh = None
    else:
        resolver = source_bvh_resolver or _lazy_source_bvh_resolver()
        source_bvh = resolver(normalize_prediction_row_for_source_bvh(row), config, step_dir)
    frame_indices = bridge_metadata.get("target_frame_indices", [])
    source_tracking_frames, source_tracking_names = _load_source_soma_tracking_frames(
        source_bvh=source_bvh,
        frame_indices=frame_indices,
    )
    visual_cfg = config.get("visual_validation", {})
    if not isinstance(visual_cfg, Mapping):
        visual_cfg = {}
    width = int(visual_cfg.get("width", 640))
    height = int(visual_cfg.get("height", 360))
    duration_sec = frame_count / fps if fps > 0 else float(visual_cfg.get("duration_sec", 4.0))

    if execute_renderers:
        source_render = source_renderer or _lazy_source_renderer()
        source_report = source_render(
            cfg=visual_cfg,
            source_bvh=source_bvh,
            video_path=paths["row1_video"],
            report_path=paths["row1_video"].with_suffix(".json"),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            sample_id=sample_id,
        )
    else:
        source_report = _ready_source_report(source_bvh=source_bvh, sample_id=sample_id, execute_renderers=False)

    isaac_python = str(visual_cfg.get("isaac_python_bin") or sys.executable)
    isaac_script = Path(str(visual_cfg.get("isaac_render_script") or ROOT / "scripts" / "render_g1_isaac_pair.py"))
    target_render_report = renderer.render_g1_isaaclab_playback(
        python_bin=isaac_python,
        script_path=isaac_script,
        motion_path=paths["row2_motion_npz"],
        output_path=paths["row2_video"],
        duration_sec=duration_sec,
        width=width,
        height=height,
        execute=execute_renderers,
        cwd=ROOT,
    )
    target_render_report.update(
        {
            "panel": "G1 Target Playback",
            "sample_id": sample_id,
            "backend": "IsaacLab",
            "render_backend": ACCEPTANCE_G1_BACKEND,
            "data_source": ACCEPTANCE_ROW2_DATA_SOURCE,
            "target_motion_path": target_motion_report["path"],
            "target_motion_sha256": target_motion_report["sha256"],
            "capsule_renderer_used": False,
        }
    )
    prediction_render_report = renderer.render_g1_isaaclab_playback(
        python_bin=isaac_python,
        script_path=isaac_script,
        motion_path=paths["row3_motion_npz"],
        output_path=paths["row3_video"],
        duration_sec=duration_sec,
        width=width,
        height=height,
        execute=execute_renderers,
        cwd=ROOT,
    )
    prediction_render_report.update(
        {
            "panel": "G1 Kinematics Playback",
            "sample_id": sample_id,
            "backend": "IsaacLab",
            "render_backend": ACCEPTANCE_G1_BACKEND,
            "data_source": ACCEPTANCE_ROW3_DATA_SOURCE,
            "prediction_motion_path": prediction_motion_report["path"],
            "prediction_motion_sha256": prediction_motion_report["sha256"],
            "checkpoint": str(checkpoint_path or ""),
            "checkpoint_step": int(step if checkpoint_step is None else checkpoint_step),
            "capsule_renderer_used": False,
        }
    )

    if execute_renderers:
        combiner = panel_combiner or _lazy_panel_combiner()
        combine_report = combiner(
            (paths["row1_video"], paths["row2_video"], paths["row3_video"]),
            paths["combined_video"],
            fps=int(round(fps)),
            layout="vertical",
        )
    else:
        combine_report = {
            "status": "ready",
            "message": "renderers were not executed; pass --execute-renderers to create videos and combine panel",
            "layout": "vertical",
            "panel_count": 3,
        }

    metadata, acceptance_ok, failure_reasons = build_accepted_vertical_v2_metadata(
        visual_renderer=renderer,
        step=step,
        index=index,
        row=row,
        sample_id=sample_id,
        source_bvh=source_bvh,
        fps=fps,
        frame_count=frame_count,
        clip_dir=paths["artifact_dir"],
        source_video=paths["row1_video"],
        target_video=paths["row2_video"],
        inference_video=paths["row3_video"],
        combined_video=paths["combined_video"],
        source_report=source_report,
        target_report=target_render_report,
        inference_report=prediction_render_report,
        target_motion_asset_report=target_motion_report,
        motion_asset_report=prediction_motion_report,
        combine_report=combine_report,
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
    )
    target_joint_pos_asset, target_root_pos_asset, target_root_quat_asset = _load_motion_asset_arrays(
        paths["row2_motion_npz"]
    )
    pred_joint_pos_asset, pred_root_pos_asset, pred_root_quat_asset = _load_motion_asset_arrays(
        paths["row3_motion_npz"]
    )
    raw_trajectory_report = {
        "status": "blocked",
        "message": "source BVH unresolved; raw trajectory pack not persisted",
        "raw_trajectory_path": "",
        "raw_trajectory_metadata_path": "",
        "raw_trajectory_frames": 0,
        "raw_trajectory_fields": [],
    }
    review_contract = {
        "mode": "metric_horizon_bridge_only",
        "fps": float(fps),
        "frame_count": int(frame_count),
        "source_frame_range": None,
        "duration_sec": float(frame_count / fps) if fps > 0 else 0.0,
        "source_frame_indices_count": int(len(frame_indices)),
        "source_frame_indices_covered": 0,
        "physical_time_aligned": False,
        "final_review_eligible": False,
        "blocked_reason": (
            "DP accepted-v2 bridge reuses target_frame_indices from the metric horizon; "
            "final native-fps review requires the non-predict visual validation rollout/export path"
        ),
    }
    if source_tracking_frames is not None and source_tracking_names is not None:
        raw_path = paths["artifact_dir"] / f"clip_{int(index):02d}_{sample_id}_trajectory.npz"
        raw_trajectory_report = save_raw_validation_trajectory(
            trajectory={
                "clip_index": int(index),
                "local_env_index": int(index),
                "motion_id": row.get("motion_id"),
                "motion_key": row.get("sequence_id") or row.get("target_g1_path") or sample_id,
                "source_soma": source_tracking_frames[:frame_count],
                "target_g1": _g1_tracking_frames_from_joint_root(
                    joint_pos=target_joint_pos_asset,
                    root_pos=target_root_pos_asset,
                ),
                "inferred_g1": _g1_tracking_frames_from_joint_root(
                    joint_pos=pred_joint_pos_asset,
                    root_pos=pred_root_pos_asset,
                ),
                "target_root_pos_w": target_root_pos_asset,
                "target_root_rot_w": target_root_quat_asset,
                "pred_root_pos_w": pred_root_pos_asset,
                "pred_root_rot_w": pred_root_quat_asset,
                "source_frame_indices": list(frame_indices[:frame_count]),
                "encoder_routes": [],
                "source_fps": float(fps),
                "target_fps": float(fps),
                "physical_time_aligned": False,
                "root_rot_format": "wxyz",
                "initial_root_xy_zeroed": False,
                "source_soma_names": source_tracking_names,
                "g1_body_names": TRACKING_BODY_NAMES,
            },
            output_path=raw_path,
            target_fps=fps,
            duration_sec=frame_count / fps if fps > 0 else float(visual_cfg.get("duration_sec", 4.0)),
        )
        loaded_contract = native_fps_review_evidence(
            {
                "source_soma": source_tracking_frames[:frame_count],
                "target_g1": _g1_tracking_frames_from_joint_root(
                    joint_pos=target_joint_pos_asset,
                    root_pos=target_root_pos_asset,
                ),
                "inferred_g1": _g1_tracking_frames_from_joint_root(
                    joint_pos=pred_joint_pos_asset,
                    root_pos=pred_root_pos_asset,
                ),
                "source_frame_indices": list(frame_indices[:frame_count]),
                "target_fps": float(fps),
                "physical_time_aligned": False,
            }
        )
        review_contract.update(loaded_contract)
        review_contract["mode"] = "metric_horizon_bridge_only"
        review_contract["final_review_eligible"] = False
        review_contract["physical_time_aligned"] = False
        blocked_reason = review_contract.get("blocked_reason")
        review_contract["blocked_reason"] = (
            (blocked_reason + "; " if blocked_reason else "")
            + "DP bridge target_frame_indices are metric-horizon sparse/windows, not original native-fps contiguous rollout frames"
        )
    metadata["lr310_dp_prediction_bridge"] = {
        **bridge_metadata,
        "predictions_jsonl": str(predictions_jsonl),
        "target_g1_path": str(row.get("target_g1_path", "")),
        "resolved_target_g1_path": str(target_path),
        "execute_renderers": bool(execute_renderers),
        "source_bvh_resolution_skipped": bool(skip_source_bvh_resolve),
        "lr290_backend_reuse": [
            "accepted_vertical_v2_artifact_paths",
            "A0VisualValidationRenderer.write_g1_motion_npz",
            "A0VisualValidationRenderer.render_g1_isaaclab_playback",
            "build_accepted_vertical_v2_metadata",
            "scripts.train_sonic_kin_skeleton_ae._resolve_source_bvh",
        ],
        "lr290_contract_parity_note": (
            "accepted_vertical_v2 backend is reused; row3 root pose is model-predicted only when "
            "pred_root_pos_w/pred_root_quat_w exist, otherwise metadata marks target-root reuse or fixed-root fallback"
        ),
        "raw_trajectory": raw_trajectory_report,
        "review_contract": review_contract,
    }
    paths["manifest_json"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "index": int(index),
        "sample_id": sample_id,
        "fps": float(fps),
        "frames": frame_count,
        "target_motion_npz": str(paths["row2_motion_npz"]),
        "prediction_motion_npz": str(paths["row3_motion_npz"]),
        "metadata": str(paths["manifest_json"]),
        "combined_video": str(paths["combined_video"]),
        "row1_soma_somamesh_video": str(paths["row1_video"]),
        "row2_g1_target_isaaclab_video": str(paths["row2_video"]),
        "row3_g1_kinematics_isaaclab_video": str(paths["row3_video"]),
        "acceptance_ok": bool(acceptance_ok),
        "accepted_vertical_v2_status": metadata["visual_backend"].get("accepted_vertical_v2_status"),
        "acceptance_failure_reasons": failure_reasons,
        "root_fixed_fallback": metadata["lr310_dp_prediction_bridge"]["root_fixed_fallback"],
        "execute_renderers": bool(execute_renderers),
        "raw_trajectory_path": raw_trajectory_report.get("raw_trajectory_path", ""),
        "raw_trajectory_metadata_path": raw_trajectory_report.get("raw_trajectory_metadata_path", ""),
        "review_contract": review_contract,
    }


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("--config must contain a JSON object")
    else:
        config = {}
    visual_cfg = dict(config.get("visual_validation", {}))
    if args.source_bvh_root:
        visual_cfg["source_bvh_roots"] = [str(Path(value)) for value in args.source_bvh_root]
    if args.source_bvh_tar:
        visual_cfg["source_bvh_tar"] = str(Path(args.source_bvh_tar))
    if args.source_bvh_cache:
        visual_cfg["source_bvh_cache"] = str(Path(args.source_bvh_cache))
    if args.g1_robot_usd:
        visual_cfg["g1_robot_usd"] = str(Path(args.g1_robot_usd))
    if args.isaac_python_bin:
        visual_cfg["isaac_python_bin"] = str(args.isaac_python_bin)
    if args.isaac_render_script:
        visual_cfg["isaac_render_script"] = str(Path(args.isaac_render_script))
    visual_cfg["width"] = int(args.width)
    visual_cfg["height"] = int(args.height)
    config["visual_validation"] = visual_cfg
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Optional JSON config; CLI values override visual_validation keys")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--target-g1-root", action="append", default=[], help="Root for relative target_g1_path values")
    parser.add_argument("--source-bvh-root", action="append", default=[], help="Root for relative source_motion_path BVH")
    parser.add_argument("--source-bvh-tar", type=Path, help="Optional BVH tar used by _resolve_source_bvh")
    parser.add_argument("--source-bvh-cache", type=Path, help="Extraction cache for --source-bvh-tar")
    parser.add_argument("--g1-robot-usd", type=Path, help="G1 USD path; no host-specific default is required")
    parser.add_argument("--isaac-python-bin", help="IsaacLab python executable used only with --execute-renderers")
    parser.add_argument("--isaac-render-script", type=Path, default=ROOT / "scripts" / "render_g1_isaac_pair.py")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument(
        "--root-source",
        choices=("auto", "target_npz_root", "motionlib_root", "body_root", "root_fixed"),
        default="auto",
        help="How to populate root_pos/root_quat in exported G1 motion NPZ assets",
    )
    parser.add_argument("--root-body-index", type=int, default=0)
    parser.add_argument("--root-body-name", default="pelvis")
    parser.add_argument("--root-quat-format", choices=("wxyz", "xyzw"), default="wxyz")
    parser.add_argument(
        "--allow-root-fixed-fallback",
        action="store_true",
        help="Allow auto mode to create explicit fixed-root fallback metadata when no target root exists",
    )
    parser.add_argument(
        "--execute-renderers",
        action="store_true",
        help="Run SomaMesh/IsaacLab/ffmpeg renderers; without this the CLI writes NPZ assets, commands, and metadata only",
    )
    parser.add_argument(
        "--skip-source-bvh-resolve",
        action="store_true",
        help="Skip _resolve_source_bvh for local NPZ/command dry-runs; default runtime path reuses it",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    if args.count is not None and args.count <= 0:
        raise ValueError("--count must be positive")
    config = build_config(args)
    rows = read_prediction_rows(args.predictions_jsonl, count=args.count)
    target_roots = [Path(value) for value in args.target_g1_root]
    summaries = []
    errors = []
    for index, row in enumerate(rows):
        try:
            summaries.append(
                rerender_prediction_row(
                    row=row,
                    index=index,
                    predictions_jsonl=args.predictions_jsonl,
                    output_dir=args.output_dir,
                    config=config,
                    target_g1_roots=target_roots,
                    step=args.step,
                    execute_renderers=args.execute_renderers,
                    root_source=args.root_source,
                    root_body_index=args.root_body_index,
                    root_body_name=args.root_body_name,
                    root_quat_format=args.root_quat_format,
                    allow_root_fixed_fallback=args.allow_root_fixed_fallback,
                    checkpoint_path=args.checkpoint,
                    checkpoint_step=args.checkpoint_step,
                    skip_source_bvh_resolve=args.skip_source_bvh_resolve,
                )
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            errors.append({"index": index, "error": str(exc)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ok" if not errors else "partial",
        "predictions_jsonl": str(args.predictions_jsonl),
        "output_dir": str(args.output_dir),
        "step": int(args.step),
        "execute_renderers": bool(args.execute_renderers),
        "clips": summaries,
        "errors": errors,
    }
    summary_path = args.output_dir / "lr310_dp_visual_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


def safe_sample_id(row: Mapping[str, Any], index: int) -> str:
    raw = str(row.get("sample_id") or row.get("sequence_id") or row.get("target_g1_path") or f"sample_{index:06d}")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(raw).stem if "/" in raw else raw).strip("._")
    return safe[:96] or f"sample_{index:06d}"


def _as_2d_float(value: Any, *, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [frames, {width}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _select_frames(array: np.ndarray, frame_indices: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim < 1:
        raise ValueError(f"{name} must have a frame axis")
    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.shape[0] <= 0:
        raise ValueError("target frame indices must be a non-empty 1D sequence")
    if np.any(indices < 0):
        raise ValueError(f"target frame indices for {name} must be non-negative")
    if int(indices.max()) >= values.shape[0]:
        raise ValueError(
            f"target frame index {int(indices.max())} out of range for {name} with {values.shape[0]} frames"
        )
    return values[indices]


def _quat_to_wxyz(quat: np.ndarray, fmt: str) -> np.ndarray:
    array = np.asarray(quat, dtype=np.float32)
    if fmt == "xyzw":
        array = array[..., [3, 0, 1, 2]]
    elif fmt != "wxyz":
        raise ValueError(f"unsupported quaternion format: {fmt}")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-8)


def _file_sha256(path: Path | None) -> str:
    if path is None or not Path(path).exists():
        return ""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ready_source_report(*, source_bvh: Path | None, sample_id: str, execute_renderers: bool) -> dict[str, Any]:
    return {
        "status": "ready",
        "backend": "accepted_somamesh_shapes_lbs_source",
        "render_backend": "accepted_somamesh_shapes_lbs_source",
        "source_renderer": "SomaMesh LBS",
        "soma_backend": "SomaMeshShapes",
        "sample_id": sample_id,
        "source_bvh": str(source_bvh or ""),
        "source_bvh_sha256": _file_sha256(source_bvh),
        "source_provenance": {
            "source_type": "source_bvh" if source_bvh is not None else "",
            "source_bvh": str(source_bvh or ""),
            "source_bvh_sha256": _file_sha256(source_bvh),
        },
        "execute_renderers": bool(execute_renderers),
        "message": "source renderer not executed; pass --execute-renderers to create row1 video",
    }


def _lazy_source_bvh_resolver() -> SourceBvhResolver:
    from scripts.train_sonic_kin_skeleton_ae import _resolve_source_bvh

    return _resolve_source_bvh


def _lazy_source_renderer() -> SourceRenderer:
    from scripts.train_sonic_kin_skeleton_ae import _render_somamesh_shapes_source_video

    return _render_somamesh_shapes_source_video


def _lazy_panel_combiner() -> PanelCombiner:
    from scripts.train_sonic_kin_skeleton_ae import _combine_panel_videos

    return _combine_panel_videos


if __name__ == "__main__":
    raise SystemExit(main())
