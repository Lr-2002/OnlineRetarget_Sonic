#!/usr/bin/env python3
"""Upload already-rendered accepted_vertical_v2 MP4s to W&B.

This is a no-restart backfill helper for LR-342-style runs where SomaMesh/G1
accepted-v2 MP4s exist locally but only JSON/HTML summaries reached W&B.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


VIDEO_FIELDS = (
    ("primary", "combined_video"),
    ("row1_soma_somamesh", "row1_soma_somamesh_video"),
    ("row2_g1_target", "row2_g1_target_isaaclab_video"),
    ("row3_g1_kinematics", "row3_g1_kinematics_isaaclab_video"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output directory")
    parser.add_argument("--entity", default=None, help="W&B entity")
    parser.add_argument("--project", required=True, help="W&B project")
    parser.add_argument("--resume-run-id", default=None, help="Existing W&B run id to append to")
    parser.add_argument("--run-name", default="accepted-v2-backfill", help="New W&B run name")
    parser.add_argument("--step", type=int, action="append", help="Specific step to upload; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Print upload plan without importing W&B")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def accepted_summary_paths(output_dir: Path, steps: set[int] | None = None) -> list[Path]:
    base = output_dir / "periodic_eval"
    if not base.exists():
        return []
    paths: list[Path] = []
    for step_dir in sorted(base.glob("step_*")):
        if not step_dir.is_dir():
            continue
        step = _step_from_name(step_dir.name)
        if steps is not None and step not in steps:
            continue
        summary = (
            step_dir
            / "visualization"
            / "accepted_vertical_v2"
            / "lr310_dp_visual_validation_summary.json"
        )
        if summary.exists():
            paths.append(summary)
    return paths


def upload_records(output_dir: Path, steps: set[int] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in accepted_summary_paths(output_dir, steps):
        summary = load_json(summary_path)
        step = int(summary.get("step", _step_from_periodic_path(summary_path)))
        for clip in summary.get("clips", []):
            if not isinstance(clip, dict) or not bool(clip.get("acceptance_ok", False)):
                continue
            record: dict[str, Any] = {
                "step": step,
                "summary_json": str(summary_path),
                "metadata": str(clip.get("metadata", "")),
                "sample_id": str(clip.get("sample_id", "")),
            }
            for _label, field in VIDEO_FIELDS:
                record[field] = str(clip.get(field, "") or "")
            record = _fill_media_from_metadata(record)
            if _record_paths_exist(record):
                records.append(record)
    return records


def upload_to_wandb(
    records: Iterable[dict[str, Any]],
    *,
    entity: str | None,
    project: str,
    resume_run_id: str | None,
    run_name: str,
) -> int:
    import wandb

    run = wandb.init(
        entity=entity,
        project=project,
        id=resume_run_id,
        resume="allow" if resume_run_id else None,
        name=run_name if not resume_run_id else None,
        job_type="accepted_vertical_v2_mp4_backfill",
    )
    uploaded = 0
    try:
        for record in records:
            step = int(record["step"])
            payload: dict[str, Any] = {
                "periodic_eval/visualization/accepted_vertical_v2/backfill_step": step,
                "periodic_eval/visualization/accepted_vertical_v2/backfill_sample_id": record[
                    "sample_id"
                ],
                "periodic_eval/visualization/accepted_vertical_v2/backfill_summary_json": record[
                    "summary_json"
                ],
                "periodic_eval/visualization/accepted_vertical_v2/backfill_metadata": record[
                    "metadata"
                ],
            }
            for label, field in VIDEO_FIELDS:
                path = Path(str(record[field]))
                key = f"periodic_eval/visualization/accepted_vertical_v2/{label}"
                payload[key] = wandb.Video(str(path))
                run.save(str(path))
            for field in ("summary_json", "metadata"):
                path = Path(str(record[field]))
                if path.exists():
                    run.save(str(path))
            run.log(payload, step=step)
            uploaded += 1
    finally:
        run.finish()
    return uploaded


def _record_paths_exist(record: dict[str, Any]) -> bool:
    required = ["metadata", *(field for _label, field in VIDEO_FIELDS)]
    return all(Path(str(record.get(field, ""))).exists() for field in required)


def _fill_media_from_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata_path = Path(str(record.get("metadata", "")))
    if not metadata_path.exists():
        return record
    try:
        metadata = load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return record
    if not record.get("combined_video"):
        record["combined_video"] = str(metadata.get("combined_video", ""))
    contract = metadata.get("accepted_visual_contract", {})
    if not isinstance(contract, dict):
        return record
    panels = contract.get("panels", [])
    if not isinstance(panels, list):
        return record
    panel_fields = (
        ("row1_soma_somamesh_video", 0),
        ("row2_g1_target_isaaclab_video", 1),
        ("row3_g1_kinematics_isaaclab_video", 2),
    )
    for field, index in panel_fields:
        if record.get(field):
            continue
        if index < len(panels) and isinstance(panels[index], dict):
            record[field] = str(panels[index].get("artifact", "") or "")
    return record


def _step_from_name(name: str) -> int:
    try:
        return int(name.removeprefix("step_"))
    except ValueError:
        return -1


def _step_from_periodic_path(path: Path) -> int:
    for parent in path.parents:
        if parent.name.startswith("step_"):
            return _step_from_name(parent.name)
    return -1


def main() -> int:
    args = parse_args()
    steps = set(args.step) if args.step else None
    records = upload_records(args.output_dir, steps)
    if args.dry_run:
        print(json.dumps({"count": len(records), "records": records}, indent=2, sort_keys=True))
        return 0
    uploaded = upload_to_wandb(
        records,
        entity=args.entity,
        project=args.project,
        resume_run_id=args.resume_run_id,
        run_name=args.run_name,
    )
    print(json.dumps({"uploaded": uploaded}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
