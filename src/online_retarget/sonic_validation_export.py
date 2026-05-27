"""Raw trajectory export and readable renders for SONIC visual validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


DEFAULT_READABLE_CLIP_INDICES = (0, 6)
DEFAULT_READABLE_WIDTH = 1920
DEFAULT_READABLE_HEIGHT = 720

TRACKING_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

TRACKING_BODY_EDGES = (
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
)

VARIANT_NAMES = ("A1_concat", "A2_film_contact", "B1_adapter", "B2_expert")


@dataclass(frozen=True)
class PackExportResult:
    """Summary for a readable validation pack export."""

    output_dir: Path
    manifest_path: Path
    status: str
    videos_ok: int
    videos_failed: int
    missing: tuple[str, ...]


def parse_clip_indices(value: object) -> tuple[int, ...]:
    """Parse Hydra/CLI clip index values into a stable tuple."""

    if value is None:
        return DEFAULT_READABLE_CLIP_INDICES
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(int(item) for item in value)
    return (int(value),)


def clip_index_selected(clip_index: int, selected: object) -> bool:
    """Return whether ``clip_index`` should get the readable render."""

    return int(clip_index) in set(parse_clip_indices(selected))


def save_raw_validation_trajectory(
    *,
    trajectory: Mapping[str, Any],
    output_path: Path,
    target_fps: float,
    duration_sec: float,
) -> dict[str, Any]:
    """Persist validation trajectory arrays for later readable re-rendering."""

    import numpy as np

    source = _trajectory_array(trajectory, "source_soma")
    target = _trajectory_array(trajectory, "target_g1")
    inferred = _trajectory_array(trajectory, "inferred_g1")
    frame_count = min(len(source), len(target), len(inferred))
    if frame_count <= 0:
        raise RuntimeError("no validation trajectory frames available to persist")

    source_frame_indices = np.asarray(
        list(trajectory.get("source_frame_indices") or []),
        dtype=np.int64,
    )
    encoder_routes = np.asarray(list(trajectory.get("encoder_routes") or []), dtype=np.int64)
    metadata = _trajectory_metadata(
        trajectory=trajectory,
        target_fps=target_fps,
        duration_sec=duration_sec,
        frame_count=frame_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")
    np.savez_compressed(
        output_path,
        source_soma=source[:frame_count],
        target_g1=target[:frame_count],
        inferred_g1=inferred[:frame_count],
        source_frame_indices=source_frame_indices[:frame_count],
        encoder_routes=encoder_routes[:frame_count],
        source_soma_names=np.asarray(
            _body_names(trajectory.get("source_soma_names"), source.shape[1]),
            dtype="U64",
        ),
        g1_body_names=np.asarray(
            _body_names(trajectory.get("g1_body_names"), target.shape[1]),
            dtype="U64",
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype="U4096"),
    )
    _write_json(metadata_path, metadata)
    return {
        "status": "ok",
        "raw_trajectory_path": str(output_path),
        "raw_trajectory_metadata_path": str(metadata_path),
        "raw_trajectory_frames": frame_count,
        "raw_trajectory_fields": [
            "source_soma",
            "target_g1",
            "inferred_g1",
            "source_frame_indices",
            "encoder_routes",
            "source_soma_names",
            "g1_body_names",
        ],
    }


def load_raw_validation_trajectory(path: Path) -> dict[str, Any]:
    """Load a validation trajectory written by ``save_raw_validation_trajectory``."""

    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        result: dict[str, Any] = {
            "source_soma": np.asarray(payload["source_soma"], dtype=float),
            "target_g1": np.asarray(payload["target_g1"], dtype=float),
            "inferred_g1": np.asarray(payload["inferred_g1"], dtype=float),
            "source_frame_indices": [int(value) for value in payload["source_frame_indices"]],
            "encoder_routes": [int(value) for value in payload["encoder_routes"]],
            "source_soma_names": [str(value) for value in payload["source_soma_names"]],
            "g1_body_names": [str(value) for value in payload["g1_body_names"]],
        }
        metadata_text = str(payload["metadata_json"].item()) if "metadata_json" in payload else "{}"
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = json.loads(metadata_text)
    result.update(metadata)
    return result


def render_readable_validation_video(
    *,
    trajectory: Mapping[str, Any],
    video_path: Path,
    target_fps: float,
    duration_sec: float,
    width: int = DEFAULT_READABLE_WIDTH,
    height: int = DEFAULT_READABLE_HEIGHT,
) -> dict[str, Any]:
    """Render a zoomed, labeled source/target/inferred validation video."""

    import imageio.v2 as imageio
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = _trajectory_array(trajectory, "source_soma")
    target = _trajectory_array(trajectory, "target_g1")
    inferred = _trajectory_array(trajectory, "inferred_g1")
    frame_count = min(
        len(source),
        len(target),
        len(inferred),
        int(round(duration_sec * target_fps)),
    )
    if frame_count <= 0:
        raise RuntimeError("no validation trajectory frames available to render")

    source_names = _body_names(trajectory.get("source_soma_names"), source.shape[1])
    g1_names = _body_names(trajectory.get("g1_body_names"), target.shape[1])
    source_indices = [int(value) for value in trajectory.get("source_frame_indices") or []]
    panels = (
        ("Source SOMA skeleton", source[:frame_count], source_names, "#236192", "#134f3d"),
        ("Dataset G1 target skeleton", target[:frame_count], g1_names, "#2f3338", "#8a5a00"),
        ("Inferred G1 skeleton", inferred[:frame_count], g1_names, "#b23a48", "#7a1f31"),
    )
    bounds = tuple(_readable_axis_bounds(array) for _title, array, _names, _color, _key in panels)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    frame_sums: list[int] = []
    changed_frames = 0
    previous_frame: bytes | None = None
    with imageio.get_writer(
        video_path,
        fps=int(round(target_fps)),
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
    ) as writer:
        for frame_idx in range(frame_count):
            fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
            fig.patch.set_facecolor("#f5f6f2")
            for panel_index, (title, array, names, color, key_color) in enumerate(panels, start=1):
                ax = fig.add_subplot(1, 3, panel_index, projection="3d")
                _plot_readable_panel(
                    ax=ax,
                    points=array[frame_idx],
                    names=names,
                    title=title,
                    color=color,
                    key_color=key_color,
                    bounds=bounds[panel_index - 1],
                    frame_idx=frame_idx,
                    source_frame_idx=(
                        source_indices[frame_idx] if frame_idx < len(source_indices) else None
                    ),
                )
            fig.tight_layout(pad=0.8)
            image = _figure_canvas_rgb(fig)
            plt.close(fig)
            writer.append_data(image)
            frame_bytes = image.tobytes()
            frame_sums.append(int(np.asarray(image, dtype=np.uint8).sum()))
            if previous_frame is not None and frame_bytes != previous_frame:
                changed_frames += 1
            previous_frame = frame_bytes

    report = {
        "status": "ok",
        "message": "Encoded readable validation review video.",
        "video_path": str(video_path),
        "width": int(width),
        "height": int(height),
        "fps": float(target_fps),
        "frames": int(frame_count),
        "changed_frames": int(changed_frames),
        "frame_sum_min": min(frame_sums) if frame_sums else None,
        "frame_sum_max": max(frame_sums) if frame_sums else None,
        "render_backend": "matplotlib_readable_skeleton",
        "readable_features": [
            "zoomed_side_by_side_panels",
            "source_soma_skeleton",
            "g1_readable_skeleton",
            "floor_contact_grid",
            "root_axes",
            "world_xyz_axes",
            "left_right_labels",
            "source_target_frame_counters",
        ],
    }
    _write_json(video_path.with_suffix(".json"), report)
    return report


def export_readable_validation_pack(
    *,
    search_root: Path,
    run_group: str,
    output_dir: Path,
    clips: Sequence[int] = DEFAULT_READABLE_CLIP_INDICES,
    variants: Sequence[str] = VARIANT_NAMES,
    width: int = DEFAULT_READABLE_WIDTH,
    height: int = DEFAULT_READABLE_HEIGHT,
    allow_missing: bool = False,
) -> PackExportResult:
    """Render a four-variant pack from persisted validation trajectory NPZ files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_by_key = _find_latest_raw_trajectories(
        search_root=search_root,
        run_group=run_group,
        clips=clips,
        variants=variants,
    )
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    videos_ok = 0
    videos_failed = 0
    for variant in variants:
        for clip_index in clips:
            key = (variant, int(clip_index))
            raw_path = raw_by_key.get(key)
            if raw_path is None:
                missing.append(f"{variant}:clip_{clip_index:02d}")
                continue
            try:
                trajectory = load_raw_validation_trajectory(raw_path)
                target_fps = float(trajectory.get("target_fps", 50.0))
                duration_sec = float(trajectory.get("duration_sec", 4.0))
                video_path = output_dir / variant / f"clip_{clip_index:02d}_readable.mp4"
                render_report = render_readable_validation_video(
                    trajectory=trajectory,
                    video_path=video_path,
                    target_fps=target_fps,
                    duration_sec=duration_sec,
                    width=width,
                    height=height,
                )
                videos_ok += 1
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                render_report = {"status": "failed", "message": str(exc)}
                videos_failed += 1
                status = "failed"
            results.append(
                {
                    "variant": variant,
                    "clip_index": int(clip_index),
                    "status": status,
                    "raw_trajectory_path": str(raw_path),
                    "render": render_report,
                }
            )

    if missing and not allow_missing:
        videos_failed += len(missing)
    status = "ok" if not missing and videos_failed == 0 else "partial"
    manifest = {
        "status": status,
        "run_group": run_group,
        "search_root": str(search_root),
        "output_dir": str(output_dir),
        "clips": [int(item) for item in clips],
        "variants": list(variants),
        "videos_ok": videos_ok,
        "videos_failed": videos_failed,
        "missing": missing,
        "results": results,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return PackExportResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        status=status,
        videos_ok=videos_ok,
        videos_failed=videos_failed,
        missing=tuple(missing),
    )


def _find_latest_raw_trajectories(
    *,
    search_root: Path,
    run_group: str,
    clips: Sequence[int],
    variants: Sequence[str],
) -> dict[tuple[str, int], Path]:
    candidates: dict[tuple[str, int], list[Path]] = {}
    for clip_index in clips:
        pattern = (
            f"*{run_group}*/online_retarget_visual_validation/step_*/rank_*/"
            f"clip_{int(clip_index):02d}_*_trajectory.npz"
        )
        for path in search_root.glob(pattern):
            variant = _variant_from_path(path, variants)
            if variant is None:
                continue
            candidates.setdefault((variant, int(clip_index)), []).append(path)
    latest: dict[tuple[str, int], Path] = {}
    for key, paths in candidates.items():
        latest[key] = max(paths, key=lambda item: (_step_number(item), item.stat().st_mtime))
    return latest


def _variant_from_path(path: Path, variants: Sequence[str]) -> str | None:
    text = str(path)
    for variant in variants:
        if variant in text:
            return variant
    return None


def _step_number(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"step_(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def _trajectory_array(trajectory: Mapping[str, Any], key: str) -> Any:
    import numpy as np

    value = trajectory.get(key)
    if value is None:
        raise RuntimeError(f"missing validation trajectory field: {key}")
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise RuntimeError(f"empty validation trajectory field: {key}")
    if array.ndim == 2 and array.shape[-1] == 3:
        array = array.reshape(1, array.shape[0], 3)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise RuntimeError(f"{key} must have shape [frames, points, 3], got {array.shape}")
    return array


def _trajectory_metadata(
    *,
    trajectory: Mapping[str, Any],
    target_fps: float,
    duration_sec: float,
    frame_count: int,
) -> dict[str, Any]:
    return {
        "clip_index": _json_value(trajectory.get("clip_index")),
        "local_env_index": _json_value(trajectory.get("local_env_index")),
        "motion_id": _json_value(trajectory.get("motion_id")),
        "motion_key": _json_value(trajectory.get("motion_key")),
        "source_fps": float(trajectory.get("source_fps", target_fps)),
        "target_fps": float(target_fps),
        "duration_sec": float(duration_sec),
        "target_frame_count": int(frame_count),
        "physical_time_aligned": bool(trajectory.get("physical_time_aligned", False)),
    }


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _body_names(value: object, count: int) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        names = tuple(str(item) for item in value)
        if len(names) == count:
            return names
    if count == len(TRACKING_BODY_NAMES):
        return TRACKING_BODY_NAMES
    return tuple(f"point_{index:02d}" for index in range(count))


def _readable_axis_bounds(
    array: Any,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    import numpy as np

    points = np.asarray(array, dtype=float).reshape(-1, 3)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    floor_z = min(float(lo[2]), 0.0)
    hi[2] = max(float(hi[2]), floor_z + 1.0)
    center = (lo + hi) * 0.5
    span = max(float((hi - lo).max()), 0.9)
    half = span * 0.6
    return (
        (float(center[0] - half), float(center[0] + half)),
        (float(center[1] - half), float(center[1] + half)),
        (float(floor_z), float(max(floor_z + span * 1.1, center[2] + half))),
    )


def _plot_readable_panel(
    *,
    ax: Any,
    points: Any,
    names: Sequence[str],
    title: str,
    color: str,
    key_color: str,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    frame_idx: int,
    source_frame_idx: int | None,
) -> None:
    import numpy as np

    arr = np.asarray(points, dtype=float).reshape(-1, 3)
    edges = _edges_for_names(names, len(arr))
    _plot_floor(ax, bounds)
    for start, end in edges:
        if start >= len(arr) or end >= len(arr):
            continue
        segment = arr[[start, end]]
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=3.0,
            alpha=0.92,
        )
    key_indices = set(_label_indices(names))
    point_colors = [key_color if index in key_indices else color for index in range(len(arr))]
    point_sizes = [42 if index in key_indices else 26 for index in range(len(arr))]
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=point_sizes, c=point_colors, depthshade=True)
    _plot_root_axes(ax, arr, names, bounds)
    _plot_left_right_labels(ax, arr, names)
    _plot_contact_markers(ax, arr, names)
    source_label = "n/a" if source_frame_idx is None else f"{source_frame_idx:04d}"
    ax.set_title(f"{title}\nsrc {source_label}  tgt {frame_idx:04d}", fontsize=9)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.set_xlabel("world X", fontsize=8)
    ax.set_ylabel("world Y", fontsize=8)
    ax.set_zlabel("world Z", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.view_init(elev=18, azim=-60)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 0.8))


def _plot_floor(
    ax: Any,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    import numpy as np

    x0, x1 = bounds[0]
    y0, y1 = bounds[1]
    z = bounds[2][0]
    xs = np.linspace(x0, x1, 5)
    ys = np.linspace(y0, y1, 5)
    for x in xs:
        ax.plot([x, x], [y0, y1], [z, z], color="#d1d5ce", linewidth=0.8, alpha=0.8)
    for y in ys:
        ax.plot([x0, x1], [y, y], [z, z], color="#d1d5ce", linewidth=0.8, alpha=0.8)
    ax.text(x0, y0, z, "floor/contact", color="#5b635d", fontsize=8)


def _plot_root_axes(
    ax: Any,
    arr: Any,
    names: Sequence[str],
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> None:
    root_index = _name_index(names, "pelvis")
    root = arr[root_index if root_index is not None else 0]
    span = max(bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0], 1.0)
    scale = span * 0.18
    ax.quiver(root[0], root[1], root[2], scale, 0, 0, color="#c9372c", linewidth=2)
    ax.quiver(root[0], root[1], root[2], 0, scale, 0, color="#2f7d32", linewidth=2)
    ax.quiver(root[0], root[1], root[2], 0, 0, scale, color="#2f5fb3", linewidth=2)
    ax.text(root[0] + scale, root[1], root[2], "X", color="#c9372c", fontsize=8)
    ax.text(root[0], root[1] + scale, root[2], "Y", color="#2f7d32", fontsize=8)
    ax.text(root[0], root[1], root[2] + scale, "Z", color="#2f5fb3", fontsize=8)
    ax.text(root[0], root[1], root[2], "root", color="#323b3c", fontsize=8)


def _plot_left_right_labels(ax: Any, arr: Any, names: Sequence[str]) -> None:
    for label, keywords in (
        ("L ankle", ("left_ankle", "LeftFoot", "left_foot")),
        ("R ankle", ("right_ankle", "RightFoot", "right_foot")),
        ("L wrist", ("left_wrist", "LeftHand", "left_hand")),
        ("R wrist", ("right_wrist", "RightHand", "right_hand")),
    ):
        index = _first_name_index(names, keywords)
        if index is not None and index < len(arr):
            point = arr[index]
            ax.text(point[0], point[1], point[2], label, color="#111111", fontsize=8)


def _plot_contact_markers(ax: Any, arr: Any, names: Sequence[str]) -> None:
    for keyword, color in (("left_ankle", "#2b6cb0"), ("right_ankle", "#b83280")):
        index = _first_name_index(names, (keyword,))
        if index is None or index >= len(arr):
            continue
        point = arr[index]
        ax.scatter([point[0]], [point[1]], [0.0], s=60, c=color, marker="x", depthshade=False)
        ax.plot(
            [point[0], point[0]],
            [point[1], point[1]],
            [0.0, point[2]],
            color=color,
            linewidth=1.0,
            alpha=0.35,
        )


def _edges_for_names(names: Sequence[str], count: int) -> tuple[tuple[int, int], ...]:
    lookup = {name: index for index, name in enumerate(names)}
    edges = [
        (lookup[start], lookup[end])
        for start, end in TRACKING_BODY_EDGES
        if start in lookup and end in lookup
    ]
    if edges:
        return tuple(edges)
    if count >= len(TRACKING_BODY_NAMES):
        return tuple((start, end) for start, end in _tracking_index_edges() if end < count)
    return tuple((index, index + 1) for index in range(max(0, count - 1)))


def _tracking_index_edges() -> tuple[tuple[int, int], ...]:
    lookup = {name: index for index, name in enumerate(TRACKING_BODY_NAMES)}
    return tuple((lookup[start], lookup[end]) for start, end in TRACKING_BODY_EDGES)


def _label_indices(names: Sequence[str]) -> tuple[int, ...]:
    labels = []
    for index, name in enumerate(names):
        lowered = name.lower()
        if any(token in lowered for token in ("ankle", "wrist", "head", "toe", "hand", "foot")):
            labels.append(index)
    return tuple(labels)


def _name_index(names: Sequence[str], name: str) -> int | None:
    try:
        return tuple(names).index(name)
    except ValueError:
        return None


def _first_name_index(names: Sequence[str], keywords: Sequence[str]) -> int | None:
    for index, name in enumerate(names):
        lowered = name.lower()
        for keyword in keywords:
            if keyword.lower() in lowered:
                return index
    return None


def _figure_canvas_rgb(fig: Any) -> Any:
    import numpy as np

    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return rgba[..., :3].copy()
    if hasattr(fig.canvas, "tostring_rgb"):
        width, height = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return image.reshape(height, width, 3)
    raise RuntimeError("Matplotlib canvas cannot export RGB image data")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
