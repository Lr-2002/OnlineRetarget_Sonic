"""Command line helpers for repository initialization and inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_retarget.config import Paths
from online_retarget.data.bvh_quality import BVHQualityConfig, scan_bvh_quality_from_index
from online_retarget.data.curation import QualityPolicy, SplitConfig, build_split_index
from online_retarget.data.g1_quality import G1QualityConfig, scan_g1_quality_from_index
from online_retarget.data.bones_seed import actor_skeletons, summarize_metadata
from online_retarget.data.quality_merge import merge_quality_stats
from online_retarget.data.supervised_builder import SupervisedBuildConfig, build_supervised_jsonl
from online_retarget.data.thresholds import write_threshold_proposals
from online_retarget.data.windowed_builder import WindowedBuildConfig, build_windowed_jsonl
from online_retarget.evaluation import EvaluationConfig, evaluate_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(prog="online-retarget")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Summarize BONES-SEED metadata")
    inventory.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    inventory.add_argument("--sample-actors", type=int, default=3)

    split_index = subparsers.add_parser(
        "split-index",
        help="Build actor-heldout split index and metadata curation report",
    )
    split_index.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    split_index.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    split_index.add_argument("--train-ratio", type=float, default=0.8)
    split_index.add_argument("--val-ratio", type=float, default=0.1)
    split_index.add_argument("--seed", type=int, default=17)
    split_index.add_argument("--policy-name", default="metadata_only")
    split_index.add_argument("--min-duration-frames", type=int, default=0)
    split_index.add_argument(
        "--keep-mirrors-as-normal",
        action="store_true",
        help="Do not downweight mirrored rows; actor split still prevents leakage.",
    )
    split_index.add_argument(
        "--allow-missing-optional-metadata",
        action="store_true",
        help="Do not quarantine rows missing optional actor metadata fields.",
    )

    quality = subparsers.add_parser(
        "scan-g1-quality",
        help="Scan G1 CSV targets referenced by a split index for clip-level quality stats",
    )
    quality.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    quality.add_argument("--index-csv", type=Path, required=True)
    quality.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    quality.add_argument("--limit", type=int, default=100)
    quality.add_argument("--full", action="store_true", help="Scan all matching rows")
    quality.add_argument("--split", action="append", default=[])
    quality.add_argument("--curation-action", action="append", default=[])
    quality.add_argument("--fps", type=float, default=30.0)
    quality.add_argument("--max-joint-velocity", type=float, default=20.0)
    quality.add_argument("--max-root-speed", type=float, default=8.0)

    source_quality = subparsers.add_parser(
        "scan-source-quality",
        help="Scan source SOMA BVH motions referenced by a split index",
    )
    source_quality.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    source_quality.add_argument("--index-csv", type=Path, required=True)
    source_quality.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    source_quality.add_argument("--limit", type=int, default=100)
    source_quality.add_argument("--full", action="store_true", help="Scan all matching rows")
    source_quality.add_argument("--split", action="append", default=[])
    source_quality.add_argument("--curation-action", action="append", default=[])
    source_quality.add_argument("--max-channel-velocity", type=float, default=3000.0)
    source_quality.add_argument("--max-root-speed", type=float, default=500.0)
    source_quality.add_argument("--expected-frame-time", type=float)
    source_quality.add_argument("--frame-time-tolerance", type=float, default=1e-4)

    thresholds = subparsers.add_parser(
        "propose-thresholds",
        help="Propose percentile-based thresholds from quality stats JSONL",
    )
    thresholds.add_argument("--stats-jsonl", type=Path, required=True)
    thresholds.add_argument("--output-json", type=Path, required=True)
    thresholds.add_argument("--metric", action="append", required=True)
    thresholds.add_argument("--percentile", type=float, default=0.99)
    thresholds.add_argument("--action", default="quarantine")

    supervised = subparsers.add_parser(
        "build-supervised-jsonl",
        help="Build small source-BVH to G1-joint JSONL samples for baseline debugging",
    )
    supervised.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    supervised.add_argument("--index-csv", type=Path, required=True)
    supervised.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    supervised.add_argument("--split", default="train")
    supervised.add_argument("--curation-action", action="append", default=[])
    supervised.add_argument("--action-column", default="curation_action")
    supervised.add_argument("--limit", type=int, default=16)
    supervised.add_argument("--history-frames", type=int, default=8)
    supervised.add_argument("--target-frame-offset", type=int, default=0)

    windowed = subparsers.add_parser(
        "build-windowed-jsonl",
        help="Build fixed 30-body source-position windows and G1-joint JSONL samples",
    )
    windowed.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    windowed.add_argument("--index-csv", type=Path, required=True)
    windowed.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    windowed.add_argument("--split", default="train")
    windowed.add_argument("--curation-action", action="append", default=[])
    windowed.add_argument("--action-column", default="merged_quality_action")
    windowed.add_argument("--limit", type=int, default=16)
    windowed.add_argument("--history-frames", type=int, default=8)
    windowed.add_argument("--target-frame-offset", type=int, default=0)
    windowed.add_argument("--position-scale", type=float, default=0.01)

    merge_quality = subparsers.add_parser(
        "merge-quality",
        help="Merge split index, source stats, and G1 stats into a curated index",
    )
    merge_quality.add_argument("--split-index-csv", type=Path, required=True)
    merge_quality.add_argument("--source-stats-jsonl", type=Path)
    merge_quality.add_argument("--g1-stats-jsonl", type=Path)
    merge_quality.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    merge_quality.add_argument("--run-name", default="merged_quality")

    offline_eval = subparsers.add_parser(
        "offline-eval",
        help="Evaluate prediction/target JSONL and write offline metric reports",
    )
    offline_eval.add_argument("--input-jsonl", type=Path, required=True)
    offline_eval.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    offline_eval.add_argument("--run-name", default="offline_eval")
    offline_eval.add_argument("--fps", type=float, default=30.0)
    offline_eval.add_argument("--joint-jump-velocity", type=float, default=20.0)
    offline_eval.add_argument("--failure-metric", default="joint_rmse")
    offline_eval.add_argument("--max-failures", type=int, default=50)

    args = parser.parse_args()
    if args.command == "inventory":
        _inventory(args.data_root, args.sample_actors)
    elif args.command == "split-index":
        _split_index(args)
    elif args.command == "scan-g1-quality":
        _scan_g1_quality(args)
    elif args.command == "scan-source-quality":
        _scan_source_quality(args)
    elif args.command == "propose-thresholds":
        _propose_thresholds(args)
    elif args.command == "build-supervised-jsonl":
        _build_supervised_jsonl(args)
    elif args.command == "build-windowed-jsonl":
        _build_windowed_jsonl(args)
    elif args.command == "merge-quality":
        _merge_quality(args)
    elif args.command == "offline-eval":
        _offline_eval(args)


def _inventory(data_root: Path, sample_actors: int) -> None:
    summary = summarize_metadata(data_root)
    actors = actor_skeletons(data_root)
    payload = {
        "summary": summary.to_dict(),
        "sample_actors": [actor.to_dict() for actor in actors[:sample_actors]],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _split_index(args: argparse.Namespace) -> None:
    result = build_split_index(
        data_root=args.data_root,
        output_root=args.output_root,
        split_config=SplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        ),
        quality_policy=QualityPolicy(
            name=args.policy_name,
            min_duration_frames=args.min_duration_frames,
            downweight_mirrors=not args.keep_mirrors_as_normal,
            quarantine_missing_optional_metadata=not args.allow_missing_optional_metadata,
        ),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _scan_g1_quality(args: argparse.Namespace) -> None:
    limit = None if args.full else args.limit
    result = scan_g1_quality_from_index(
        data_root=args.data_root,
        index_csv=args.index_csv,
        output_root=args.output_root,
        config=G1QualityConfig(
            fps=args.fps,
            max_joint_velocity=args.max_joint_velocity,
            max_root_speed=args.max_root_speed,
        ),
        limit=limit,
        splits=tuple(args.split),
        actions=tuple(args.curation_action) or ("keep", "downweight", "quarantine"),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _scan_source_quality(args: argparse.Namespace) -> None:
    limit = None if args.full else args.limit
    result = scan_bvh_quality_from_index(
        data_root=args.data_root,
        index_csv=args.index_csv,
        output_root=args.output_root,
        config=BVHQualityConfig(
            max_channel_velocity=args.max_channel_velocity,
            max_root_speed=args.max_root_speed,
            expected_frame_time=args.expected_frame_time,
            frame_time_tolerance=args.frame_time_tolerance,
        ),
        limit=limit,
        splits=tuple(args.split),
        actions=tuple(args.curation_action) or ("keep", "downweight", "quarantine"),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _propose_thresholds(args: argparse.Namespace) -> None:
    payload = write_threshold_proposals(
        stats_jsonl=args.stats_jsonl,
        output_json=args.output_json,
        metrics=tuple(args.metric),
        percentile=args.percentile,
        action=args.action,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_supervised_jsonl(args: argparse.Namespace) -> None:
    result = build_supervised_jsonl(
        data_root=args.data_root,
        index_csv=args.index_csv,
        output_root=args.output_root,
        config=SupervisedBuildConfig(
            split=args.split,
            actions=tuple(args.curation_action) or ("keep", "downweight"),
            action_column=args.action_column,
            limit=args.limit,
            history_frames=args.history_frames,
            target_frame_offset=args.target_frame_offset,
        ),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _build_windowed_jsonl(args: argparse.Namespace) -> None:
    result = build_windowed_jsonl(
        data_root=args.data_root,
        index_csv=args.index_csv,
        output_root=args.output_root,
        config=WindowedBuildConfig(
            split=args.split,
            actions=tuple(args.curation_action) or ("keep", "downweight"),
            action_column=args.action_column,
            limit=args.limit,
            history_frames=args.history_frames,
            target_frame_offset=args.target_frame_offset,
            position_scale=args.position_scale,
        ),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _merge_quality(args: argparse.Namespace) -> None:
    result = merge_quality_stats(
        split_index_csv=args.split_index_csv,
        source_stats_jsonl=args.source_stats_jsonl,
        g1_stats_jsonl=args.g1_stats_jsonl,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _offline_eval(args: argparse.Namespace) -> None:
    result = evaluate_jsonl(
        input_jsonl=args.input_jsonl,
        output_root=args.output_root,
        config=EvaluationConfig(
            fps=args.fps,
            joint_jump_velocity=args.joint_jump_velocity,
            failure_metric=args.failure_metric,
            max_failures=args.max_failures,
            run_name=args.run_name,
        ),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
