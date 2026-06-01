#!/usr/bin/env python3
"""Render SOMA BVH source motion as a skinned SomaMesh video panel.

This is the stable source-panel renderer for accepted OnlineRetarget visual
validation.  It intentionally bypasses capsule/BVH stick rendering when SOMA
mesh data is available.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


DEFAULT_RETARGETER_ROOT = Path("/home/user/project/ContextRetarget/third_party/soma-retargeter")
DEFAULT_SOMA_USD = Path("/home/user/data/motion_data/soma_shapes/soma_base_rig/soma_base_skel_minimal.usd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--retargeter-root", type=Path, default=DEFAULT_RETARGETER_ROOT)
    parser.add_argument("--soma-usd", type=Path, default=DEFAULT_SOMA_USD)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--frame-count", type=int, default=200)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--stride-triangles", type=int, default=3)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def load_soma_retargeter(root: Path) -> None:
    sys.path.insert(0, str(root))


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transforms_to_matrices(transforms: np.ndarray) -> np.ndarray:
    mats = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], transforms.shape[0], axis=0)
    mats[:, :3, 3] = transforms[:, :3]
    for idx, transform in enumerate(transforms):
        mats[idx, :3, :3] = quat_xyzw_to_matrix(transform[3:7])
    return mats


def skin_vertices(
    *,
    bind_points: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
    global_transforms: np.ndarray,
    bind_transforms: np.ndarray,
) -> np.ndarray:
    global_mats = transforms_to_matrices(global_transforms)
    bind_mats = transforms_to_matrices(bind_transforms)
    skin_mats = global_mats @ np.linalg.inv(bind_mats)

    hom = np.concatenate([bind_points, np.ones((bind_points.shape[0], 1), dtype=np.float64)], axis=1)
    out = np.zeros((bind_points.shape[0], 3), dtype=np.float64)
    for influence_index in range(joint_indices.shape[1]):
        idx = joint_indices[:, influence_index]
        weighted = np.einsum("nij,nj->ni", skin_mats[idx, :3, :], hom)
        out += weighted * joint_weights[:, influence_index : influence_index + 1]
    return out.astype(np.float32)


def soma_y_up_to_display_z_up(points: np.ndarray) -> np.ndarray:
    """Convert SOMA/BVH Y-up coordinates to the accepted Z-up display frame."""

    points = np.asarray(points, dtype=np.float32)
    converted = np.empty_like(points)
    converted[:, 0] = points[:, 0]
    converted[:, 1] = -points[:, 2]
    converted[:, 2] = points[:, 1]
    return converted


def soma_y_up_vector_to_display_z_up(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    return np.asarray([vector[0], -vector[2], vector[1]], dtype=np.float32)


def smooth_xy(positions: np.ndarray, radius: int = 4) -> np.ndarray:
    if len(positions) == 0:
        return positions
    smoothed = positions.astype(np.float64, copy=True)
    xy = positions[:, :2].astype(np.float64)
    for frame_idx in range(len(positions)):
        start = max(0, frame_idx - radius)
        stop = min(len(positions), frame_idx + radius + 1)
        smoothed[frame_idx, :2] = xy[start:stop].mean(axis=0)
    return smoothed.astype(np.float32)


def find_pelvis_index(skeleton: Any) -> int:
    names = list(getattr(skeleton, "joint_names", []) or [])
    for candidate in ("Hips", "Pelvis", "pelvis", "hips"):
        if candidate in names:
            return names.index(candidate)
    return 0


def look_at_rotation(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera_pos
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return np.stack([right, up, forward], axis=0)


def project(
    vertices: np.ndarray,
    camera_pos: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    rot = look_at_rotation(camera_pos, target)
    cam = (vertices.astype(np.float64) - camera_pos) @ rot.T
    z = np.clip(cam[:, 2], 0.05, None)
    focal = 0.95 * min(width, height)
    xy = np.empty((vertices.shape[0], 2), dtype=np.float64)
    xy[:, 0] = width * 0.5 + focal * cam[:, 0] / z
    xy[:, 1] = height * 0.54 - focal * cam[:, 1] / z
    return xy, cam[:, 2]


def draw_axes(draw: Any, origin: np.ndarray, axes: dict[str, np.ndarray], project_fn: Any, label_prefix: str) -> None:
    colors = {"+X": (220, 60, 60), "+Y": (50, 170, 70), "+Z": (55, 95, 220)}
    o2, _ = project_fn(origin[None, :])
    ox, oy = o2[0]
    for label, vec in axes.items():
        p2, _ = project_fn((origin + vec)[None, :])
        x, y = p2[0]
        draw.line((ox, oy, x, y), fill=colors[label], width=3)
        draw.text((x + 4, y + 4), f"{label_prefix} {label}", fill=colors[label])


def draw_frame(
    *,
    vertices: np.ndarray,
    triangles: np.ndarray,
    frame_idx: int,
    total_frames: int,
    fps: float,
    title: str,
    width: int,
    height: int,
    stride_triangles: int,
    camera_target: np.ndarray,
    root_position: np.ndarray,
    root_axes: dict[str, np.ndarray],
    camera_span: float,
) -> Any:
    from PIL import Image, ImageDraw

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    span = max(float(np.linalg.norm(maxs - mins)), float(camera_span), 1.0)
    target = camera_target.astype(np.float64, copy=True)
    camera_pos = target + np.array([2.4, -3.2, 1.35], dtype=np.float64) * span

    image = Image.new("RGB", (width, height), (248, 248, 244))
    draw = ImageDraw.Draw(image)

    def project_fn(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return project(points, camera_pos, target, width, height)

    xy, depth = project_fn(vertices)
    tris = triangles[:: max(1, stride_triangles)]
    valid = np.all(depth[tris] > 0.05, axis=1)
    tris = tris[valid]
    order = np.argsort(depth[tris].mean(axis=1))[::-1]
    light = np.array([0.25, -0.45, 0.86], dtype=np.float64)
    light = light / np.linalg.norm(light)
    for tri in tris[order]:
        pts3 = vertices[tri].astype(np.float64)
        normal = np.cross(pts3[1] - pts3[0], pts3[2] - pts3[0])
        normal_norm = np.linalg.norm(normal)
        shade = 0.65 if normal_norm < 1e-9 else 0.45 + 0.45 * max(0.0, float(np.dot(normal / normal_norm, light)))
        base = np.array([100, 151, 186], dtype=np.float64)
        color = tuple(np.clip(base * shade + np.array([40, 38, 32]), 0, 255).astype(np.uint8))
        draw.polygon([tuple(xy[index]) for index in tri], fill=color)

    axes_len = max(0.35, 0.18 * span)
    draw_axes(
        draw,
        np.asarray([mins[0], mins[1], mins[2]], dtype=np.float32),
        {
            "+X": np.array([axes_len, 0.0, 0.0]),
            "+Y": np.array([0.0, axes_len, 0.0]),
            "+Z": np.array([0.0, 0.0, axes_len]),
        },
        project_fn,
        "SOMA world",
    )
    draw_axes(
        draw,
        root_position.astype(np.float32),
        {label: axis * axes_len for label, axis in root_axes.items()},
        project_fn,
        "root",
    )

    draw.rectangle((0, 0, width, 62), fill=(250, 250, 247))
    draw.text((14, 10), title, fill=(28, 32, 36))
    draw.text(
        (14, 34),
        f"SomaMesh LBS | frame {frame_idx + 1}/{total_frames} | camera=root XY, fixed height",
        fill=(64, 68, 72),
    )
    return image


def ffmpeg_writer(output_path: Path, width: int, height: int, fps: float) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
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
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def render_somamesh_source_panel(
    *,
    bvh_path: Path,
    output_path: Path,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    stride_triangles: int = 3,
    title: str | None = None,
    retargeter_root: Path | None = None,
    usd_path: Path | None = None,
    mesh_loader: Any | None = None,
) -> dict[str, Any]:
    """Render one SOMA BVH as a skinned SomaMesh panel."""

    if retargeter_root is not None:
        load_soma_retargeter(retargeter_root)

    from soma_retargeter.assets.bvh import load_bvh

    if mesh_loader is None:
        if usd_path is None:
            usd_path = DEFAULT_SOMA_USD
        from soma_retargeter.assets.usd import load_skeletal_mesh_from_usd

        def mesh_loader(skeleton: Any) -> Any:
            return load_skeletal_mesh_from_usd(
                str(usd_path),
                skeleton,
                "/OUTPUT/c_geometry_grp/Mesh",
                "/OUTPUT/c_skeleton_grp/Root",
                "soma_base_skel_minimal",
            )

    skeleton, animation = load_bvh(str(bvh_path))
    mesh = mesh_loader(skeleton)
    skinned = mesh.skinned_meshes[0]
    bind_points = skinned.points.numpy()
    triangles = skinned.indices.numpy().reshape(-1, 3)
    joint_indices = skinned.joint_indices.numpy().reshape(bind_points.shape[0], -1)
    joint_weights = skinned.joint_weights.numpy().reshape(bind_points.shape[0], -1)
    bind_transforms = mesh.bind_transforms.numpy()
    pelvis_index = find_pelvis_index(skeleton)
    pelvis_name = (getattr(skeleton, "joint_names", []) or ["Root"])[pelvis_index]

    available_frames = int(math.floor(animation.num_frames * fps / animation.sample_rate))
    total_frames = min(int(frame_count), available_frames)
    globals_by_frame: list[np.ndarray] = []
    vertices_by_frame: list[np.ndarray] = []
    root_positions: list[np.ndarray] = []
    mesh_median_positions: list[np.ndarray] = []
    root_heights: list[float] = []
    mesh_median_z: list[float] = []
    spans: list[float] = []

    for out_idx in range(total_frames):
        local = animation.sample(out_idx / fps)
        global_tf = skeleton.compute_global_transforms(local)
        vertices = skin_vertices(
            bind_points=bind_points,
            joint_indices=joint_indices,
            joint_weights=joint_weights,
            global_transforms=global_tf,
            bind_transforms=bind_transforms,
        )
        vertices = soma_y_up_to_display_z_up(vertices)
        root_position = soma_y_up_to_display_z_up(global_tf[pelvis_index : pelvis_index + 1, :3])[0]
        globals_by_frame.append(global_tf)
        vertices_by_frame.append(vertices)
        root_positions.append(root_position)
        mesh_median = np.median(vertices, axis=0)
        mesh_median_positions.append(mesh_median)
        root_heights.append(float(root_position[2]))
        mesh_median_z.append(float(mesh_median[2]))
        spans.append(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))))

    root_positions_np = np.asarray(root_positions, dtype=np.float32)
    smoothed_root = smooth_xy(root_positions_np)
    stable_root_height = float(root_positions_np[0, 2]) if len(root_positions_np) else 0.0
    camera_targets = smoothed_root.copy()
    if len(camera_targets):
        camera_targets[:, 2] = stable_root_height + 0.35
    camera_span = float(np.percentile(spans, 75)) if spans else 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    process = ffmpeg_writer(output_path, width, height, fps)
    if process.stdin is None:
        raise RuntimeError("failed to open ffmpeg stdin")
    changed_frames = 0
    last_frame: np.ndarray | None = None
    for out_idx in range(total_frames):
        root_rotation = quat_xyzw_to_matrix(globals_by_frame[out_idx][pelvis_index, 3:7])
        root_axes = {
            "+X": soma_y_up_vector_to_display_z_up(root_rotation[:, 0]),
            "+Y": soma_y_up_vector_to_display_z_up(root_rotation[:, 1]),
            "+Z": soma_y_up_vector_to_display_z_up(root_rotation[:, 2]),
        }
        image = draw_frame(
            vertices=vertices_by_frame[out_idx],
            triangles=triangles,
            frame_idx=out_idx,
            total_frames=total_frames,
            fps=fps,
            title=title or bvh_path.stem,
            width=width,
            height=height,
            stride_triangles=stride_triangles,
            camera_target=camera_targets[out_idx],
            root_position=root_positions_np[out_idx],
            root_axes=root_axes,
            camera_span=camera_span,
        )
        arr = np.asarray(image, dtype=np.uint8)
        if last_frame is not None and not np.array_equal(arr, last_frame):
            changed_frames += 1
        last_frame = arr.copy()
        process.stdin.write(arr.tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path}: {stderr[-1000:]}")

    return {
        "status": "ok",
        "stem": title or bvh_path.stem,
        "bvh_path": str(bvh_path),
        "video": str(output_path),
        "frames": total_frames,
        "fps": fps,
        "duration_sec": total_frames / fps,
        "changed_frames": changed_frames,
        "vertices": int(bind_points.shape[0]),
        "triangles_loaded": int(triangles.shape[0]),
        "triangle_stride": int(max(1, stride_triangles)),
        "triangles_drawn_per_frame": int(math.ceil(triangles.shape[0] / max(1, stride_triangles))),
        "renderer": "NVIDIA soma-retargeter USD SkeletalMesh + CPU LBS + software triangle rasterizer",
        "not_capsule_bvh_visualizer": True,
        "source_display_conversion": "(x, y, z)_display = (x, -z, y)_soma",
        "source_coordinate_convention": "SOMA/BVH native Y-up LBS; Z-up conversion is display-only",
        "camera_reference_joint": pelvis_name,
        "camera_reference_joint_index": int(pelvis_index),
        "camera_policy": (
            "camera follows smoothed Hips/pelvis XY only; look-at height is locked to "
            "the first-frame pelvis display height plus 0.35m"
        ),
        "forbidden_camera_sources": "mesh median, bbox center, hand height, and per-frame camera-Z updates are not used",
        "root_display_height_range_m": [
            float(min(root_heights)) if root_heights else 0.0,
            float(max(root_heights)) if root_heights else 0.0,
        ],
        "mesh_median_display_z_range_m": [
            float(min(mesh_median_z)) if mesh_median_z else 0.0,
            float(max(mesh_median_z)) if mesh_median_z else 0.0,
        ],
        "camera_target_display_z_range_m": [
            float(camera_targets[:, 2].min()) if len(camera_targets) else 0.0,
            float(camera_targets[:, 2].max()) if len(camera_targets) else 0.0,
        ],
        "mesh_median_xy_span_not_used_for_camera_m": [
            float(np.ptp(np.asarray(mesh_median_positions, dtype=np.float32)[:, 0])) if mesh_median_positions else 0.0,
            float(np.ptp(np.asarray(mesh_median_positions, dtype=np.float32)[:, 1])) if mesh_median_positions else 0.0,
        ],
    }


def main() -> None:
    args = parse_args()
    report = render_somamesh_source_panel(
        bvh_path=args.bvh,
        output_path=args.output,
        fps=args.fps,
        frame_count=args.frame_count,
        width=args.width,
        height=args.height,
        stride_triangles=args.stride_triangles,
        title=args.title or args.bvh.stem,
        retargeter_root=args.retargeter_root,
        usd_path=args.soma_usd,
    )
    report_path = args.report or args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
