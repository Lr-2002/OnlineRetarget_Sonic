#!/usr/bin/env python3
"""Render source SOMA BVH / paired G1 dataset clips for data auditing."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import train_sonic_kin_skeleton_ae as kin


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sonic_kin_soma_motionlib_a1_concat_1gpu.json")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="OnlineRetarget")
    parser.add_argument("--wandb-entity", default="")
    return parser.parse_args()


def category_from_motion_name(name: str) -> str:
    stem = Path(name).stem
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    stem = re.sub(r"__A\d+(_M)?$", "", stem)
    stem = re.sub(r"_\d+$", "", stem)
    return stem or Path(name).stem


def rows_from_cached_bvhs(config: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    input_cfg = config["input_data"]
    visual_cfg = config.get("visual_validation", {})
    robot_dir = Path(input_cfg["robot_motion_dir"])
    soma_dir = Path(input_cfg["soma_motion_dir"])
    cache_root = Path(str(visual_cfg.get("source_bvh_cache", "")))
    rows: list[dict[str, Any]] = []
    if not cache_root.exists():
        return rows

    for bvh_path in sorted(cache_root.rglob("*.bvh")):
        try:
            rel = bvh_path.relative_to(cache_root).as_posix()
        except ValueError:
            continue
        prefix = "soma_proportional/bvh/"
        if not rel.startswith(prefix):
            continue
        tail = rel[len(prefix) :]
        if "/" not in tail:
            continue
        date, bvh_name = tail.split("/", 1)
        motion_stem = Path(bvh_name).stem
        robot_name = f"{date}__{motion_stem}.pkl"
        robot_path = robot_dir / robot_name
        soma_path = soma_dir / robot_name
        if not robot_path.exists() or not soma_path.exists():
            continue
        rows.append(
            {
                "path": str(robot_path),
                "relative_path": robot_name,
                "robot_relative_path": robot_name,
                "soma_relative_path": robot_name,
                "filename": Path(robot_name).stem,
                "category": category_from_motion_name(robot_name),
                "source_soma_proportional_path": rel,
                "source_bvh": str(bvh_path),
            }
        )

    selected: list[dict[str, Any]] = []
    seen_categories: set[str] = set()
    for row in rows:
        category = str(row["category"])
        if category in seen_categories:
            continue
        selected.append(row)
        seen_categories.add(category)
        if len(selected) >= count:
            return selected
    for row in rows:
        if len(selected) >= count:
            break
        if row not in selected:
            selected.append(row)
    return selected


def combine_two_videos(inputs: Sequence[Path], output: Path, fps: int) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return {"status": "blocked", "message": "ffmpeg is required"}
    missing = [str(path) for path in inputs if not path.exists() or path.stat().st_size == 0]
    if missing:
        return {"status": "failed", "message": "missing input panel", "missing": missing}
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


def render_pair(
    *,
    row: Mapping[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    index: int,
    render_deps: Mapping[str, Any],
    g1_model: Any,
    duration_sec: float,
) -> dict[str, Any]:
    cfg = dict(config.get("visual_validation", {}))
    clip_name = kin._safe_filename(str(row.get("filename") or Path(str(row["relative_path"])).stem))
    clip_dir = output_dir / f"{index:02d}_{clip_name}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    arrays = kin.load_soma_motionlib_arrays(row, config)
    robot_root = kin._load_motionlib_robot_root(row, config)
    _, soma_entry = kin._single_motionlib_entry(
        Path(config["input_data"]["soma_motion_dir"]) / str(row["soma_relative_path"])
    )
    fps = float(arrays["fps"])
    total_frames = min(
        arrays["joint_pos"].shape[0],
        robot_root["root_pos"].shape[0],
        robot_root["root_quat"].shape[0],
    )
    frame_count = min(total_frames, max(1, int(round(duration_sec * fps))))
    render_config = render_deps["ReviewClipExportConfig"](
        render_max_frames=frame_count,
        render_width=int(cfg.get("width", 640)),
        render_height=int(cfg.get("height", 360)),
        fps=fps,
        source_position_scale=float(cfg.get("source_position_scale", 0.01)),
        model_xml=Path(str(cfg.get("g1_model_xml", ""))),
    )

    source_video = clip_dir / "source_bvh_capsules.mp4"
    dataset_video = clip_dir / "dataset_g1_capsules.mp4"
    combined_video = clip_dir / "source_bvh_dataset_g1.mp4"
    metadata_path = clip_dir / "metadata.json"

    source_bvh = kin._resolve_source_bvh(row, config, output_dir)
    if source_bvh is not None:
        source_report = kin._render_time_aligned_source_bvh(
            source_bvh,
            source_video,
            render_config=render_config,
            target_fps=fps,
            frame_count=frame_count,
            render_deps=render_deps,
        )
    else:
        source_report = render_deps["_render_capsule_3d_video"](
            frames=kin._soma_motionlib_source_frames(arrays["soma_joints"][:frame_count], arrays.get("joint_names")),
            edges=kin._soma_edges(arrays.get("joint_names")),
            video_path=source_video,
            config=render_config,
            label="source soma motionlib fallback",
            up_axis=2,
            capsule_color=(48, 132, 83),
            key_color=(132, 103, 34),
        )

    root_pos = robot_root["root_pos"][:frame_count]
    root_quat = robot_root["root_quat"][:frame_count]
    dataset_frames = kin._g1_prediction_frames(
        arrays["joint_pos"][:frame_count],
        root_pos=root_pos,
        root_quat=root_quat,
        g1_model=g1_model,
        render_deps=render_deps,
    )
    dataset_report = render_deps["_render_capsule_3d_video"](
        frames=dataset_frames,
        edges=kin._g1_edges(g1_model, render_deps["G1_CAPSULE_IGNORE_BODIES"]),
        video_path=dataset_video,
        config=render_config,
        label="dataset g1 fk",
        up_axis=2,
        capsule_color=(61, 107, 160),
        key_color=(139, 91, 41),
    )
    combine_report = combine_two_videos((source_video, dataset_video), combined_video, fps=int(round(fps)))

    soma_transl = np.asarray(soma_entry.get("soma_transl", np.zeros((frame_count, 3))), dtype=np.float32)
    soma_fps = float(soma_entry.get("fps") or config["input_data"].get("source_fps") or fps)
    soma_transl = kin.resample_soma_array(soma_transl, soma_fps, fps, target_len=total_frames)
    source_span = float(np.linalg.norm(soma_transl[:frame_count][-1, :2] - soma_transl[:frame_count][0, :2])) if frame_count > 1 else 0.0
    target_span = float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2])) if frame_count > 1 else 0.0
    joint_std = float(np.std(arrays["joint_pos"][:frame_count]))
    metadata = {
        "index": index,
        "filename": row.get("filename", ""),
        "category": row.get("category", ""),
        "relative_path": row.get("relative_path", ""),
        "source_bvh": str(source_bvh) if source_bvh is not None else "",
        "fps": fps,
        "frames": frame_count,
        "duration_sec": frame_count / fps if fps > 0 else 0.0,
        "soma_source_xy_span": source_span,
        "g1_target_root_xy_span": target_span,
        "g1_joint_pos_std": joint_std,
        "source_render": source_report,
        "dataset_render": dataset_report,
        "combine": combine_report,
        "combined_video": str(combined_video),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "index": index,
        "filename": row.get("filename", ""),
        "category": row.get("category", ""),
        "combined_status": combine_report.get("status"),
        "combined_video": str(combined_video),
        "metadata": str(metadata_path),
        "source_status": source_report.get("status"),
        "dataset_status": dataset_report.get("status"),
        "source_bvh": str(source_bvh) if source_bvh is not None else "",
        "soma_source_xy_span": source_span,
        "g1_target_root_xy_span": target_span,
        "g1_joint_pos_std": joint_std,
    }


def main() -> None:
    args = parse_args()
    config = kin.read_config(Path(args.config))
    timestamp = kin.timestamp_compact()
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/dataset_pair_visual_audit") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = rows_from_cached_bvhs(config, args.count)
    if not rows:
        raise SystemExit("no cached BVH-backed paired motionlib rows were found")

    render_deps = kin._load_visual_render_deps()
    g1_model = render_deps["load_g1_kinematic_model"](Path(config["visual_validation"]["g1_model_xml"]))
    reports = []
    started = time.perf_counter()
    for index, row in enumerate(rows[: args.count]):
        report = render_pair(
            row=row,
            config=config,
            output_dir=output_dir,
            index=index,
            render_deps=render_deps,
            g1_model=g1_model,
            duration_sec=args.duration_sec,
        )
        reports.append(report)
        print(json.dumps({"event": "rendered", **report}, sort_keys=True), flush=True)

    summary = {
        "status": "ok",
        "output_dir": str(output_dir),
        "requested_count": args.count,
        "rendered_count": len(reports),
        "ok_count": sum(1 for row in reports if row.get("combined_status") == "ok"),
        "elapsed_sec": time.perf_counter() - started,
        "reports": reports,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.wandb:
        import wandb

        run = wandb.init(
            entity=args.wandb_entity or None,
            project=args.wandb_project,
            name=f"dataset_pair_visual_audit_{timestamp}",
            config={
                "config": args.config,
                "count": args.count,
                "duration_sec": args.duration_sec,
                "output_dir": str(output_dir),
            },
            tags=["dataset-audit", "bvh-g1-dataset", "soma-motionlib"],
        )
        payload: dict[str, Any] = {}
        table = wandb.Table(
            columns=[
                "index",
                "category",
                "filename",
                "source_status",
                "dataset_status",
                "source_xy_span",
                "target_root_xy_span",
                "joint_std",
                "video",
            ]
        )
        for report in reports:
            video_path = Path(str(report["combined_video"]))
            video = wandb.Video(str(video_path), fps=50, format="mp4") if video_path.exists() else None
            key = f"bvh_dataset_pair/{int(report['index']):02d}_{kin._safe_metric_name(str(report['category']))}"
            if video is not None:
                payload[key] = video
            table.add_data(
                report["index"],
                report["category"],
                report["filename"],
                report["source_status"],
                report["dataset_status"],
                report["soma_source_xy_span"],
                report["g1_target_root_xy_span"],
                report["g1_joint_pos_std"],
                video,
            )
        payload["dataset_pair_summary"] = table
        run.log(payload)
        run.summary.update(
            {
                "rendered_count": summary["rendered_count"],
                "ok_count": summary["ok_count"],
                "output_dir": str(output_dir),
            }
        )
        run.finish()
        summary["wandb_url"] = run.url
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "summary", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
