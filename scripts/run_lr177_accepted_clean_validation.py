#!/usr/bin/env python3
"""Rerender LR-177 clean supervised validation with the accepted LR-117 view.

This runner exports model inference from an existing supervised checkpoint to a
G1 motion NPZ, then renders the source SOMA BVH as SomaMesh and the inferred G1
motion as IsaacLab kinematic playback. It does not launch training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_sonic_kin_skeleton_ae as kin  # noqa: E402


DEFAULT_ISAAC_PYTHON = Path("/home/user/venvs/isaaclab-210/bin/python")
DEFAULT_SOMA_PYTHON = Path("/home/user/project/ContextRetarget/third_party/soma-retargeter/.venv/bin/python")
DEFAULT_SOMA_RETARGETER = Path("/home/user/project/ContextRetarget/third_party/soma-retargeter")
DEFAULT_SOMA_USD = Path("/home/user/data/motion_data/soma_shapes/soma_base_rig/soma_base_skel_minimal.usd")
DEFAULT_ROBOT_USD = Path("/home/user/project/OnlineRetarget/runs/isaaclab_urdf_cache/g1_main/main.usd")
DEFAULT_GROUND_COLOR = (0.08, 0.20, 0.72)
DEFAULT_GROUND_SIZE = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing stats/ and checkpoints/.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to <run-dir>/checkpoints/latest.pt.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--sample-name", default="", help="Exact row filename/stem to render.")
    parser.add_argument("--sample-index", type=int, default=0, help="Index within the stable visual row selection.")
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true", help="Run IsaacLab render after exporting NPZ.")
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--soma-python", type=Path, default=DEFAULT_SOMA_PYTHON)
    parser.add_argument("--soma-retargeter-root", type=Path, default=DEFAULT_SOMA_RETARGETER)
    parser.add_argument("--somamesh-usd", type=Path, default=DEFAULT_SOMA_USD)
    parser.add_argument("--somamesh-triangle-stride", type=int, default=3)
    parser.add_argument("--robot-usd", type=Path, default=DEFAULT_ROBOT_USD)
    parser.add_argument("--preserve-world-root", action="store_true", default=False)
    parser.add_argument("--no-preserve-world-root", dest="preserve_world_root", action="store_false")
    parser.add_argument("--draw-orientation-labels", action="store_true", default=True)
    parser.add_argument("--no-draw-orientation-labels", dest="draw_orientation_labels", action="store_false")
    parser.add_argument("--ground-size", type=float, default=DEFAULT_GROUND_SIZE)
    parser.add_argument("--ground-color", type=float, nargs=3, default=DEFAULT_GROUND_COLOR)
    parser.add_argument("--camera-mode", choices=("trajectory", "follow", "fixed"), default="follow")
    parser.add_argument("--camera-offset", type=float, nargs=3, default=(3.4, -4.4, 2.2))
    parser.add_argument("--camera-follow-smoothing", type=int, default=4)
    parser.add_argument("--camera-framing-margin", type=float, default=1.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = kin.read_config(args.config)
    visual_cfg = config.get("visual_validation", {})
    run_name = args.run_name or time.strftime("lr177_accepted_clean_validation_%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (Path(config.get("runtime", {}).get("write_root", ROOT / "outputs")) / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint or (args.run_dir / "checkpoints" / "latest.pt")
    stats_path = args.run_dir / "stats" / "normalization.pt"
    rows = select_rows(config, args)
    if not rows:
        raise SystemExit("no rows selected for accepted clean validation")

    device = torch.device(args.device)
    stats = load_stats(stats_path, device)
    model, joint_dim = load_model(config, checkpoint, stats, device)
    reports: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        report = export_and_render(
            args=args,
            config=config,
            row=row,
            model=model,
            stats=stats,
            device=device,
            joint_dim=joint_dim,
            output_dir=output_dir,
            index=index,
        )
        reports.append(report)
        print(json.dumps({"event": "clip", **report}, sort_keys=True), flush=True)

    summary = {
        "status": "ok" if all(row.get("status") == "ok" for row in reports) else "partial",
        "run_name": run_name,
        "output_dir": str(output_dir),
        "config": str(args.config),
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint),
        "accepted_visualization_contract": accepted_contract(args),
        "reports": reports,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "accepted_clean_validation_report.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({"event": "summary", **summary}, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 2


def select_rows(config: dict[str, Any], args: argparse.Namespace) -> list[Mapping[str, Any]]:
    data_root = Path(config["input_data"].get("data_root", config["input_data"].get("robot_motion_dir", ".")))
    rows, _ = kin.rows_from_index(config, data_root)
    if args.sample_name:
        stem = Path(args.sample_name).stem
        selected = [
            row
            for row in rows
            if Path(str(row.get("filename") or row.get("relative_path", ""))).stem == stem
            or Path(str(row.get("robot_relative_path", ""))).stem == stem
            or Path(str(row.get("relative_path", ""))).stem == stem
        ]
    else:
        visual_cfg = config.get("visual_validation", {})
        selected = kin._select_visual_rows(
            rows,
            count=max(args.sample_limit, args.sample_index + 1),
            salt=f"{config['variant']['name']}:{config['training']['seed']}",
        )[args.sample_index :]
    return list(selected[: max(1, args.sample_limit)])


def load_stats(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=device, weights_only=False)
    return {key: value.to(device) for key, value in payload.items() if torch.is_tensor(value)}


def load_model(
    config: dict[str, Any],
    checkpoint: Path,
    stats: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.nn.Module, int]:
    motion_dim = int(stats["motion_mean"].numel())
    skeleton_dim = int(stats["skeleton_mean"].numel())
    target_dim = int(stats["target_mean"].numel())
    window = int(config["features"]["future_window_frames"])
    root_pose_dim = kin.root_pose_target_dim(config, window)
    command_dim = kin.target_command_dim(target_dim, window, config)
    if command_dim <= 0 or command_dim % (window * 2) != 0:
        raise ValueError(f"target_dim={target_dim} is incompatible with window={window}")
    joint_dim = command_dim // (window * 2)
    model = kin.make_model(motion_dim, skeleton_dim, target_dim, config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, joint_dim


def export_and_render(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    row: Mapping[str, Any],
    model: torch.nn.Module,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    joint_dim: int,
    output_dir: Path,
    index: int,
) -> dict[str, Any]:
    visual_cfg = config.get("visual_validation", {})
    clip_name = kin._safe_filename(str(row.get("filename") or Path(str(row["relative_path"])).stem))
    clip_dir = output_dir / f"{index:02d}_{clip_name}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    arrays = kin.load_soma_motionlib_arrays(row, config)
    robot_root = kin._load_motionlib_robot_root(row, config)
    fps = float(arrays["fps"])
    duration_sec = float(args.duration_sec if args.duration_sec is not None else visual_cfg.get("duration_sec", 4.0))
    frame_count = min(
        arrays["joint_pos"].shape[0],
        robot_root["root_pos"].shape[0],
        max(1, int(round(duration_sec * fps))),
    )
    prediction = kin._predict_motionlib_visual_g1_state(
        model=model,
        arrays=arrays,
        frame_count=frame_count,
        stats=stats,
        device=device,
        config=config,
        joint_dim=joint_dim,
        fallback_root_pos=robot_root["root_pos"][:frame_count],
        fallback_root_quat=robot_root["root_quat"][:frame_count],
    )
    root_quat = prediction.get("root_quat")
    if root_quat is None:
        root_quat = euler_xyz_to_quat_wxyz_batch(np.asarray(prediction["root_euler"], dtype=np.float32))

    inference_npz = clip_dir / "inference_g1_world_composed.npz"
    np.savez(
        inference_npz,
        joint_pos=np.asarray(prediction["joint_pos"][:frame_count], dtype=np.float32),
        root_pos=np.asarray(prediction["root_pos"][:frame_count], dtype=np.float32),
        root_quat=np.asarray(root_quat[:frame_count], dtype=np.float32),
        fps=np.asarray(fps, dtype=np.float32),
    )
    source_bvh = kin._resolve_source_bvh(row, config, output_dir)
    report: dict[str, Any] = {
        "index": index,
        "filename": row.get("filename", ""),
        "relative_path": row.get("relative_path", ""),
        "status": "ok",
        "render_status": "not_requested",
        "fps": fps,
        "frames": frame_count,
        "source_bvh": str(source_bvh) if source_bvh is not None else "",
        "inference_npz": str(inference_npz),
        "root_xy_span_m": root_xy_span(np.asarray(prediction["root_pos"][:frame_count])),
        "dataset_root_xy_span_m": root_xy_span(robot_root["root_pos"][:frame_count]),
        "accepted_visualization_contract": accepted_contract(args),
    }
    if not args.render:
        return report
    if source_bvh is None:
        report.update({"status": "blocked", "message": "source BVH could not be resolved"})
        return report

    final_video = clip_dir / "source_somamesh_inference_g1_isaac_with_axes.mp4"
    source_video = clip_dir / "source_somabvh_somamesh.mp4"
    source_report_path = clip_dir / "source_somabvh_somamesh.json"
    target_video = clip_dir / "target_g1_isaaclab.mp4"
    target_report_path = target_video.with_suffix(".json")
    render_log = clip_dir / "accepted_render.log"

    source_command = [
        str(args.soma_python),
        str(ROOT / "scripts" / "render_somamesh_source.py"),
        "--bvh",
        str(source_bvh),
        "--output",
        str(source_video),
        "--report",
        str(source_report_path),
        "--retargeter-root",
        str(args.soma_retargeter_root),
        "--soma-usd",
        str(args.somamesh_usd),
        "--fps",
        str(fps),
        "--frame-count",
        str(frame_count),
        "--width",
        str(int(args.width if args.width is not None else visual_cfg.get("width", 960))),
        "--height",
        str(int(args.height if args.height is not None else visual_cfg.get("height", 540))),
        "--stride-triangles",
        str(int(args.somamesh_triangle_stride)),
        "--title",
        clip_name,
    ]
    source_env = os.environ.copy()
    source_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(args.soma_retargeter_root),
            str(ROOT),
            str(SRC),
            source_env.get("PYTHONPATH", ""),
        ]
    )
    source_result = run_logged(source_command, render_log, cwd=ROOT, env=source_env)
    if source_result.returncode != 0:
        report.update(
            {
                "status": "failed",
                "render_stage": "source_somamesh",
                "render_log": str(render_log),
                "source_command": source_command,
                "source_returncode": source_result.returncode,
            }
        )
        return report

    target_command = [
        str(args.isaac_python),
        str(ROOT / "scripts" / "render_g1_isaac_pair.py"),
        "--g1-motion",
        str(inference_npz),
        "--format",
        "npz",
        "--output",
        str(target_video),
        "--robot-usd",
        str(args.robot_usd),
        "--duration-sec",
        str(duration_sec),
        "--width",
        str(int(args.width if args.width is not None else visual_cfg.get("width", 960))),
        "--height",
        str(int(args.height if args.height is not None else visual_cfg.get("height", 540))),
        "--source-renderer",
        "somamesh",
        "--camera-mode",
        str(args.camera_mode),
        "--camera-offset",
        *[str(value) for value in args.camera_offset],
        "--ground-size",
        str(args.ground_size),
        "--ground-color",
        *[str(value) for value in args.ground_color],
        "--camera-follow-smoothing",
        str(args.camera_follow_smoothing),
        "--camera-framing-margin",
        str(args.camera_framing_margin),
        "--preserve-world-root" if args.preserve_world_root else "",
        "--draw-orientation-labels" if args.draw_orientation_labels else "",
        "--root-rot-format",
        "wxyz",
        "--fast-exit-after-report",
    ]
    target_command = [item for item in target_command if item]
    target_result = run_logged(target_command, render_log, cwd=ROOT)
    if target_result.returncode != 0:
        report.update(
            {
                "status": "failed",
                "render_stage": "target_isaaclab",
                "render_log": str(render_log),
                "source_command": source_command,
                "source_returncode": source_result.returncode,
                "target_command": target_command,
                "target_returncode": target_result.returncode,
            }
        )
        return report

    combine_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video),
        "-i",
        str(target_video),
        "-filter_complex",
        f"[0:v][1:v]hstack=inputs=2,fps={int(round(fps))}[v]",
        "-map",
        "[v]",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(final_video),
    ]
    combine_result = run_logged(combine_command, render_log, cwd=ROOT)
    metadata_path = final_video.with_suffix(".json")
    source_report = load_json_if_exists(source_report_path)
    target_report = load_json_if_exists(target_report_path)
    final_report = {
        "status": "ok" if combine_result.returncode == 0 and final_video.exists() else "failed",
        "sample_name": clip_name,
        "source_bvh": str(source_bvh),
        "inference_npz": str(inference_npz),
        "final_video": str(final_video),
        "source_video": str(source_video),
        "target_video": str(target_video),
        "source_report": str(source_report_path),
        "target_report": str(target_report_path),
        "render_log": str(render_log),
        "source_display_conversion": "(x, y, z)_display = (x, -z, y)_soma",
        "source_renderer": source_report.get("renderer", ""),
        "target_backend": target_report.get("backend", ""),
        "g1_asset_path": target_report.get("robot_asset", str(args.robot_usd)),
        "target_root_xy_span_m": target_report.get("target_root_xy_span_m"),
        "changed_frames_source": source_report.get("changed_frames"),
        "changed_frames_target": target_report.get("changed_frames"),
        "orientation_overlay": target_report.get("orientation_overlay"),
        "ground_size": target_report.get("ground_size"),
        "ground_color": target_report.get("ground_color"),
        "camera_policy": target_report.get("camera_policy"),
        "ffprobe": ffprobe_json(final_video) if final_video.exists() else {},
    }
    metadata_path.write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report.update(
        {
            "status": final_report["status"],
            "source_command": source_command,
            "source_returncode": source_result.returncode,
            "target_command": target_command,
            "target_returncode": target_result.returncode,
            "combine_command": combine_command,
            "combine_returncode": combine_result.returncode,
            "render_log": str(render_log),
            "accepted_video": str(final_video),
            "accepted_metadata": str(metadata_path),
            "source_video": str(source_video),
            "target_video": str(target_video),
            "source_metadata": str(source_report_path),
            "target_metadata": str(target_report_path),
        }
    )
    report["accepted_renderer_report"] = final_report
    return report


def run_logged(
    command: Sequence[str],
    log_path: Path,
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(result.stdout)
        log.write(f"\nreturncode={result.returncode}\n")
    return result


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ffprobe_json(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate,duration",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        return {"status": "failed", "output": result.stdout[-2000:]}
    return json.loads(result.stdout)


def euler_xyz_to_quat_wxyz_batch(euler: np.ndarray) -> np.ndarray:
    return np.asarray([euler_xyz_to_quat_wxyz(row) for row in euler], dtype=np.float32)


def euler_xyz_to_quat_wxyz(euler: Sequence[float]) -> list[float]:
    x, y, z = [float(value) for value in euler]
    cx, sx = math.cos(x * 0.5), math.sin(x * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    cz, sz = math.cos(z * 0.5), math.sin(z * 0.5)
    return [
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ]


def root_xy_span(root_pos: np.ndarray) -> float:
    if root_pos.shape[0] <= 1:
        return 0.0
    xy = root_pos[:, :2]
    return float(np.linalg.norm(np.max(xy, axis=0) - np.min(xy, axis=0)))


def accepted_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source": "SomaMesh/global-SOMA source panel",
        "source_display_conversion": "(x, y, z)_display = (x, -z, y)_soma",
        "target": "IsaacLab G1 kinematic playback",
        "semantic_lr_markers": True,
        "world_axes": True,
        "root_axes": True,
        "source_renderer": "somamesh",
        "preserve_world_root": bool(args.preserve_world_root),
        "draw_orientation_labels": bool(args.draw_orientation_labels),
        "ground_size": float(args.ground_size),
        "ground_color": [float(value) for value in args.ground_color],
        "camera_mode": args.camera_mode,
        "camera_offset": [float(value) for value in args.camera_offset],
        "camera_follow_smoothing": int(args.camera_follow_smoothing),
        "camera_framing_margin": float(args.camera_framing_margin),
        "robot_usd": str(args.robot_usd),
        "scope": "kinematic visualization only; not policy tracking or dynamics proof",
    }


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# LR-177 Accepted Clean Validation",
        "",
        f"- Status: {summary['status']}",
        f"- Checkpoint: `{summary['checkpoint']}`",
        "- Source: SomaMesh/global-SOMA with `(x, y, z)_display = (x, -z, y)_soma`.",
        "- Target: IsaacLab G1 kinematic playback from exported checkpoint inference.",
        "- Overlays: world axes, root axes, semantic L/R markers.",
        "- Scope: kinematic visualization only; not policy/dynamics/tracking proof.",
        "",
        "| # | sample | status | frames | root XY span | video |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in summary["reports"]:
        lines.append(
            f"| {row.get('index', '')} | `{row.get('filename', '')}` | {row.get('status', '')} | "
            f"{row.get('frames', '')} | {row.get('root_xy_span_m', '')} | `{row.get('accepted_video', '')}` |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
