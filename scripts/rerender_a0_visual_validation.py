#!/usr/bin/env python3
"""Rerender A0 visual validation from a fixed checkpoint.

This is a read-only inference path. It loads the frozen A0 model checkpoint,
the existing rows cache, and normalization stats, then calls the reusable
visual-validation boundary in acceptance mode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch  # noqa: E402

import scripts.train_sonic_kin_skeleton_ae as kin  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows-cache", type=Path, default=None)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--num-videos", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--acceptance-backend", action="store_true")
    parser.add_argument("--isaac-python-bin", default="/workspace/isaaclab/_isaac_sim/python.sh")
    parser.add_argument("--isaac-render-script", type=Path, default=ROOT / "scripts" / "render_g1_isaac_pair.py")
    parser.add_argument("--no-execute-isaaclab", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = kin.read_config(args.config)
    visual_cfg = config.setdefault("visual_validation", {})
    if args.num_videos is not None:
        visual_cfg["num_videos"] = int(args.num_videos)
    if args.duration_sec is not None:
        visual_cfg["duration_sec"] = float(args.duration_sec)
    if args.width is not None:
        visual_cfg["width"] = int(args.width)
    if args.height is not None:
        visual_cfg["height"] = int(args.height)

    training_output_dir = args.checkpoint.parent.parent
    rows_cache = args.rows_cache or kin.rows_from_index_cache_path(training_output_dir)
    stats_path = args.stats or training_output_dir / "stats" / "normalization.pt"
    if not rows_cache.exists():
        raise FileNotFoundError(f"rows cache is required for read-only rerender: {rows_cache}")
    if not stats_path.exists():
        raise FileNotFoundError(f"normalization stats are required for read-only rerender: {stats_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint is missing: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    rows, skipped, rows_cache_report = _load_or_repair_rows_cache_split(
        rows_cache,
        output_dir=args.output_dir,
        config=config,
    )
    skeleton_lookup = kin.build_skeleton_ae_feature_lookup(config, device)
    if skeleton_lookup is not None:
        skeleton_lookup.validate_and_annotate_rows(rows)
    validation_dataset = kin.KinWindowDataset(rows, "validation", config, skeleton_lookup)

    loaded_stats = torch.load(stats_path, map_location=device, weights_only=False)
    stats = {key: value.to(device) for key, value in loaded_stats.items() if torch.is_tensor(value)}
    kin.require_normalization_keys(stats, config)
    motion_dim = int(stats["motion_mean"].numel())
    skeleton_dim = kin.skeleton_feature_dim(stats, config)
    target_dim = int(stats["target_mean"].numel())
    window = int(config["features"]["future_window_frames"])
    root_pose_dim = kin.root_pose_target_dim(config, window)
    command_dim = kin.target_command_dim(target_dim, window, config)
    if command_dim <= 0 or command_dim % (window * 2) != 0:
        raise ValueError(f"target_dim={target_dim} is incompatible with window={window} and root pose dim={root_pose_dim}")
    joint_dim = command_dim // (window * 2)

    model = kin.make_model(motion_dim, skeleton_dim, target_dim, config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state)
    step = int(args.step if args.step is not None else checkpoint.get("step", 0))

    metrics = kin.run_visual_validation(
        model=model,
        validation_rows=validation_dataset.rows,
        stats=stats,
        device=device,
        config=config,
        output_dir=args.output_dir,
        step=step,
        joint_dim=joint_dim,
        wandb_run=None,
        skeleton_feature_lookup=skeleton_lookup,
        acceptance_backend=bool(args.acceptance_backend),
        isaac_python_bin=args.isaac_python_bin,
        isaac_render_script=args.isaac_render_script,
        execute_isaaclab=not bool(args.no_execute_isaaclab),
    )
    summary_path = args.output_dir / "visual_validation" / f"step_{step:08d}" / "summary.json"
    rerender_inputs = {
        "rows_cache_original": str(rows_cache),
        "rows_cache_effective": str(rows_cache_report["effective_path"]),
        "rows_cache_repair": rows_cache_report,
        "stats": str(stats_path),
        "checkpoint": str(args.checkpoint),
    }
    _annotate_summary(summary_path, rerender_inputs=rerender_inputs)
    result: dict[str, Any] = {
        "event": "a0_visual_rerender",
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "rows_cache": str(rows_cache_report["effective_path"]),
        "rows_cache_original": str(rows_cache),
        "rows_cache_repair": rows_cache_report,
        "rows_cache_skipped": int(skipped),
        "stats": str(stats_path),
        "output_dir": str(args.output_dir),
        "summary": str(summary_path),
        "acceptance_backend": bool(args.acceptance_backend),
        "metrics": metrics,
    }
    print(json.dumps(result, sort_keys=True), flush=True)


def _load_or_repair_rows_cache_split(
    rows_cache: Path,
    *,
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    rows, skipped = kin.read_rows_from_index_cache(rows_cache)
    missing_split = sum(1 for row in rows if "split" not in row)
    counts_before = _split_counts(rows)
    if missing_split == 0:
        return rows, skipped, {
            "status": "unchanged",
            "original_path": str(rows_cache),
            "effective_path": str(rows_cache),
            "row_count": len(rows),
            "skipped_count": int(skipped),
            "missing_split_count": 0,
            "split_counts": counts_before,
        }

    split_cfg = config.get("split", {})
    repaired_rows = [dict(row) for row in rows]
    kin.split_rows(
        repaired_rows,
        float(split_cfg["validation_ratio"]),
        str(split_cfg["hash_salt"]),
    )
    repaired_path = output_dir / "rerender_inputs" / "rows_from_index_cache.with_split.json"
    payload = json.loads(rows_cache.read_text(encoding="utf-8"))
    payload["rows"] = repaired_rows
    payload["row_count"] = len(repaired_rows)
    payload["skipped_count"] = int(skipped)
    payload["split_repair"] = {
        "source_cache": str(rows_cache),
        "validation_ratio": float(split_cfg["validation_ratio"]),
        "hash_salt": str(split_cfg["hash_salt"]),
        "missing_split_count": int(missing_split),
    }
    kin.write_rows_from_index_cache(repaired_path, payload)
    report = {
        "status": "repaired",
        "original_path": str(rows_cache),
        "effective_path": str(repaired_path),
        "row_count": len(repaired_rows),
        "skipped_count": int(skipped),
        "missing_split_count": int(missing_split),
        "split_counts_before": counts_before,
        "split_counts": _split_counts(repaired_rows),
        "validation_ratio": float(split_cfg["validation_ratio"]),
        "hash_salt": str(split_cfg["hash_salt"]),
    }
    return repaired_rows, skipped, report


def _split_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get("split", "missing"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _annotate_summary(summary_path: Path, *, rerender_inputs: dict[str, Any]) -> None:
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["rerender_inputs"] = rerender_inputs
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
