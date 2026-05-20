#!/usr/bin/env python3
"""Export paired SOMA BVH and BONES-SONIC G1 capsule videos for visual checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_retarget.data.review_clips import ReviewClipExportConfig, _SourceCapsuleRenderer
from online_retarget.data.review_clips import _render_capsule_3d_video
from online_retarget.data.sonic_review_clips import SONIC_PRUNED_CAPSULE_EDGES
from online_retarget.data.sonic_review_clips import SONIC_PRUNED_BODY_NAMES
from online_retarget.data.bones_sonic import SONIC_BODY_NAMES, SONIC_ORDER_NOTE


DEFAULT_FILENAMES = ("reach_jump_R_001__A419",)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("runs/indices/seed_clean_pair_mapping_v0/seed_clean_pair_mapping.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/vis_pair_check"))
    parser.add_argument("--run-name", default="seed_clean_pair_visual_check_v0")
    parser.add_argument("--filename", action="append", default=[])
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--source-position-scale", type=float, default=0.01)
    args = parser.parse_args()

    np = _require_numpy()
    rows = _select_rows(
        _read_mapping(args.mapping_csv),
        args.filename or list(DEFAULT_FILENAMES),
        args.limit,
    )
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        clip_dir = output_dir / f"{index:02d}_{_safe_name(row['filename'])}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        source_video = clip_dir / "source_soma_bvh_capsules.mp4"
        sonic_video = clip_dir / "target_g1_sonic_capsules.mp4"
        metadata_path = clip_dir / "metadata.json"

        bvh_fps = _bvh_fps(Path(row["clean_bvh_path"]))
        source_config = ReviewClipExportConfig(
            render_max_frames=0,
            render_width=args.width,
            render_height=args.height,
            fps=bvh_fps,
            source_position_scale=args.source_position_scale,
        )
        source_report = _SourceCapsuleRenderer(source_config).render_bvh(
            Path(row["clean_bvh_path"]),
            source_video,
        )

        sonic_report = _render_sonic_capsules(
            Path(row["bones_sonic_path"]),
            sonic_video,
            width=args.width,
            height=args.height,
            np=np,
        )

        metadata = {
            "filename": row["filename"],
            "actor_uid": row.get("actor_uid", ""),
            "split": row.get("split", ""),
            "curation_action": row.get("curation_action", ""),
            "quality_flags": row.get("quality_flags", ""),
            "merged_quality_action": row.get("merged_quality_action", ""),
            "merged_quality_flags": row.get("merged_quality_flags", ""),
            "content_natural_desc_1": row.get("content_natural_desc_1", ""),
            "content_technical_description": row.get("content_technical_description", ""),
            "source_bvh": row["clean_bvh_path"],
            "source_bvh_fps": bvh_fps,
            "source_bvh_frame_count": row.get("move_duration_frames", ""),
            "source_bvh_capsule_video": str(source_video),
            "target_g1_sonic_npz": row["bones_sonic_path"],
            "target_g1_sonic_fps": row.get("sonic_fps", ""),
            "target_g1_sonic_frame_count": row.get("sonic_frame_count", ""),
            "target_g1_sonic_capsule_video": str(sonic_video),
            "render_source": source_report,
            "render_sonic_capsules": sonic_report,
            "note": (
                "Source video is BVH FK rendered as 3D capsules from clean SOMA proportional BVH. "
                "Target capsule video is BONES-SONIC G1 body_pos_w rendered as pruned 3D capsules. "
                + SONIC_ORDER_NOTE
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary_rows.append(
            {
                "filename": row["filename"],
                "actor_uid": row.get("actor_uid", ""),
                "split": row.get("split", ""),
                "merged_quality_action": row.get("merged_quality_action", ""),
                "merged_quality_flags": row.get("merged_quality_flags", ""),
                "source_frames": row.get("move_duration_frames", ""),
                "source_fps": bvh_fps,
                "sonic_frames": row.get("sonic_frame_count", ""),
                "sonic_fps": row.get("sonic_fps", ""),
                "source_status": source_report.get("status", ""),
                "source_changed_frames": source_report.get("changed_frames", ""),
                "sonic_status": sonic_report.get("status", ""),
                "sonic_changed_frames": sonic_report.get("changed_frames", ""),
                "source_video": str(source_video),
                "sonic_capsule_video": str(sonic_video),
                "metadata": str(metadata_path),
            }
        )

    _write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "mapping_csv": str(args.mapping_csv),
                "output_dir": str(output_dir),
                "rows": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(_readme(summary_rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": summary_rows}, indent=2, sort_keys=True))


def _read_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_rows(rows: Sequence[dict[str, str]], names: Sequence[str], limit: int) -> list[dict[str, str]]:
    by_name = {row["filename"]: row for row in rows}
    selected = [by_name[name] for name in names if name in by_name]
    if len(selected) < limit:
        for row in rows:
            if len(selected) >= limit:
                break
            if row in selected:
                continue
            if (
                row.get("merged_quality_action") == "keep"
                and row.get("curation_action") == "keep"
                and row.get("clean_bvh_exists") == "True"
                and row.get("bones_sonic_exists") == "True"
            ):
                selected.append(row)
    return selected[:limit]


def _render_sonic_capsules(
    npz_path: Path,
    video_path: Path,
    *,
    width: int,
    height: int,
    np: Any,
) -> dict[str, object]:
    try:
        with np.load(npz_path) as data:
            body_pos = np.asarray(data["body_pos_w"], dtype=float)
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    except Exception as exc:
        return {"status": "failed", "message": f"Could not load SONIC NPZ: {exc}"}
    frames = _body_pos_frames(body_pos)
    config = ReviewClipExportConfig(render_max_frames=0, render_width=width, render_height=height, fps=fps)
    return _render_capsule_3d_video(
        frames=frames,
        edges=_sonic_edges(),
        video_path=video_path,
        config=config,
        label="target g1 sonic body_pos_w capsules",
        up_axis=2,
        capsule_color=(61, 107, 160),
        key_color=(139, 91, 41),
    )


def _body_pos_frames(body_pos: Any) -> list[dict[str, tuple[float, float, float]]]:
    selected = [
        (index, name)
        for index, name in enumerate(SONIC_BODY_NAMES)
        if name in set(SONIC_PRUNED_BODY_NAMES)
    ]
    frames: list[dict[str, tuple[float, float, float]]] = []
    for frame in body_pos:
        frames.append(
            {
                name: (
                    float(frame[index, 0]),
                    float(frame[index, 1]),
                    float(frame[index, 2]),
                )
                for index, name in selected
            }
        )
    return frames


def _sonic_edges() -> tuple[tuple[str, str], ...]:
    available = set(SONIC_BODY_NAMES)
    return tuple((start, end) for start, end in SONIC_PRUNED_CAPSULE_EDGES if start in available and end in available)


def _bvh_fps(path: Path) -> float:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip().startswith("Frame Time:"):
                frame_time = float(line.split(":", 1)[1].strip())
                return 1.0 / frame_time
    return 120.0


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _readme(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Pair Visual Check",
        "",
        "原始 SOMA proportional BVH 使用 3D capsule FK 渲染；目标侧 G1 使用 BONES-SONIC NPZ `body_pos_w` 渲染为 pruned 3D capsule。",
        "",
        "| filename | source | G1 capsules | metadata |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("filename", "")),
                    str(row.get("source_video", "")),
                    str(row.get("sonic_capsule_video", "")),
                    str(row.get("metadata", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value).strip("._")
    return safe[:120] or "clip"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("This script needs numpy; use /home/user/venvs/isaaclab-210/bin/python.") from exc
    return np


if __name__ == "__main__":
    main()
