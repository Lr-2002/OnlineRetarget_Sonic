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
from online_retarget.data.pair_quality import PairQualityConfig, scan_pair_quality_from_index
from online_retarget.data.policy_audit import (
    CurationPolicyAuditConfig,
    audit_curation_policy,
)
from online_retarget.data.quality_merge import merge_quality_stats
from online_retarget.data.review_manifest import build_review_manifest, merge_review_decisions
from online_retarget.data.source_fk_quality import (
    SourceFKQualityConfig,
    scan_source_fk_quality_from_index,
)
from online_retarget.data.supervised_builder import SupervisedBuildConfig, build_supervised_jsonl
from online_retarget.data.thresholds import (
    write_accepted_threshold_policy,
    write_threshold_proposals,
)
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
    quality.add_argument(
        "--sample-by",
        action="append",
        default=[],
        help="When --limit is active, stratify scan rows by this index field.",
    )
    quality.add_argument("--fps", type=float, default=120.0)
    quality.add_argument("--max-joint-velocity", type=float, default=20.0)
    quality.add_argument("--max-root-speed", type=float, default=8.0)
    quality.add_argument(
        "--max-joint-acceleration",
        type=float,
        help="Optional threshold for G1 joint acceleration flags. Defaults to metric-only.",
    )
    quality.add_argument(
        "--max-root-acceleration",
        type=float,
        help="Optional threshold for G1 root acceleration flags. Defaults to metric-only.",
    )
    quality.add_argument(
        "--max-root-jerk",
        type=float,
        help="Optional threshold for G1 root jerk flags. Defaults to metric-only.",
    )
    quality.add_argument("--root-position-scale", type=float, default=0.01)
    quality.add_argument("--joint-angle-scale", type=float, default=0.017453292519943295)
    quality.add_argument("--root-rotation-scale", type=float, default=0.017453292519943295)
    quality.add_argument("--frame-stride", type=int, default=1)
    quality.add_argument("--max-frames", type=int)
    quality.add_argument("--model-xml", type=Path)
    quality.add_argument("--ground-height", type=float, default=0.0)
    quality.add_argument("--contact-height-threshold", type=float, default=0.04)
    quality.add_argument("--max-contact-slide-speed", type=float, default=0.25)
    quality.add_argument("--max-mean-foot-clearance", type=float, default=0.10)
    quality.add_argument("--max-penetration-depth", type=float, default=0.03)
    quality.add_argument(
        "--max-contact-correction-offset",
        type=float,
        default=0.15,
        help="Metric-only cap for vertical contact-mask correction candidate offsets.",
    )
    quality.add_argument("--min-contact-frame-ratio", type=float, default=0.05)
    quality.add_argument("--max-joint-limit-violation-rate", type=float, default=0.0)
    quality.add_argument("--start-end-frames", type=int, default=10)
    quality.add_argument("--max-start-end-root-speed", type=float, default=0.20)
    quality.add_argument("--min-self-collision-distance", type=float, default=0.015)
    quality.add_argument("--max-self-collision-proxy-rate", type=float, default=0.0)
    quality.add_argument("--min-self-collision-kinematic-hops", type=int, default=4)

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
    source_quality.add_argument(
        "--sample-by",
        action="append",
        default=[],
        help="When --limit is active, stratify scan rows by this index field.",
    )
    source_quality.add_argument("--max-channel-velocity", type=float, default=3000.0)
    source_quality.add_argument("--max-root-speed", type=float, default=500.0)
    source_quality.add_argument(
        "--max-channel-acceleration",
        type=float,
        help="Optional threshold for source channel acceleration flags. Defaults to metric-only.",
    )
    source_quality.add_argument(
        "--max-root-acceleration",
        type=float,
        help="Optional threshold for source root acceleration flags. Defaults to metric-only.",
    )
    source_quality.add_argument(
        "--max-root-jerk",
        type=float,
        help="Optional threshold for source root jerk flags. Defaults to metric-only.",
    )
    source_quality.add_argument("--expected-frame-time", type=float)
    source_quality.add_argument("--frame-time-tolerance", type=float, default=1e-4)

    source_fk_quality = subparsers.add_parser(
        "scan-source-fk-quality",
        help="Scan source SOMA BVH motions with FK foot/contact geometry metrics",
    )
    source_fk_quality.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    source_fk_quality.add_argument("--index-csv", type=Path, required=True)
    source_fk_quality.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    source_fk_quality.add_argument("--limit", type=int, default=100)
    source_fk_quality.add_argument("--full", action="store_true", help="Scan all matching rows")
    source_fk_quality.add_argument("--split", action="append", default=[])
    source_fk_quality.add_argument("--curation-action", action="append", default=[])
    source_fk_quality.add_argument("--action-column", default="curation_action")
    source_fk_quality.add_argument(
        "--sample-by",
        action="append",
        default=[],
        help="When --limit is active, stratify scan rows by this index field.",
    )
    source_fk_quality.add_argument(
        "--fps",
        type=float,
        help="Override source BVH FPS for contact/slide speeds. Defaults to BVH Frame Time.",
    )
    source_fk_quality.add_argument("--position-scale", type=float, default=0.01)
    source_fk_quality.add_argument("--frame-stride", type=int, default=1)
    source_fk_quality.add_argument("--max-frames", type=int)
    source_fk_quality.add_argument("--ground-height", type=float)
    source_fk_quality.add_argument("--ground-percentile", type=float, default=0.05)
    source_fk_quality.add_argument("--contact-height-threshold", type=float, default=0.04)
    source_fk_quality.add_argument("--max-contact-slide-speed", type=float, default=0.25)
    source_fk_quality.add_argument("--max-mean-foot-clearance", type=float, default=0.10)
    source_fk_quality.add_argument("--max-penetration-depth", type=float, default=0.03)
    source_fk_quality.add_argument(
        "--max-contact-correction-offset",
        type=float,
        default=0.15,
        help="Metric-only cap for vertical contact-mask correction candidate offsets.",
    )
    source_fk_quality.add_argument("--min-contact-frame-ratio", type=float, default=0.05)

    pair_quality = subparsers.add_parser(
        "scan-pair-quality",
        help="Scan source/G1 pair consistency and provenance from a split index",
    )
    pair_quality.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    pair_quality.add_argument("--index-csv", type=Path, required=True)
    pair_quality.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    pair_quality.add_argument("--limit", type=int, default=100)
    pair_quality.add_argument("--full", action="store_true", help="Scan all matching rows")
    pair_quality.add_argument("--split", action="append", default=[])
    pair_quality.add_argument("--curation-action", action="append", default=[])
    pair_quality.add_argument(
        "--sample-by",
        action="append",
        default=[],
        help="When --limit is active, stratify scan rows by this index field.",
    )
    pair_quality.add_argument("--expected-source-frame-time", type=float)
    pair_quality.add_argument("--g1-fps", type=float, default=120.0)
    pair_quality.add_argument("--frame-time-tolerance", type=float, default=1e-4)
    pair_quality.add_argument("--max-frame-count-delta", type=int, default=0)
    pair_quality.add_argument("--max-duration-delta-sec", type=float, default=1e-3)
    pair_quality.add_argument("--target-provenance", default="kinematic_g1_csv")

    thresholds = subparsers.add_parser(
        "propose-thresholds",
        help="Propose percentile-based thresholds from quality stats JSONL",
    )
    thresholds.add_argument("--stats-jsonl", type=Path, required=True)
    thresholds.add_argument("--output-json", type=Path, required=True)
    thresholds.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric where high values are bad; propose an upper-tail threshold.",
    )
    thresholds.add_argument(
        "--lower-metric",
        action="append",
        default=[],
        help="Metric where low values are bad; propose a lower-tail threshold.",
    )
    thresholds.add_argument("--percentile", type=float, default=0.99)
    thresholds.add_argument("--action", default="quarantine")
    thresholds.add_argument(
        "--group-by",
        action="append",
        default=[],
        help="Also propose thresholds for each value of this stats field, e.g. category.",
    )
    thresholds.add_argument(
        "--min-group-size",
        type=int,
        default=1,
        help="Skip grouped threshold proposals with fewer rows than this.",
    )

    threshold_policy = subparsers.add_parser(
        "accept-threshold-policy",
        help="Write a named accepted threshold policy artifact from proposal JSON files",
    )
    threshold_policy.add_argument("--policy-id", required=True)
    threshold_policy.add_argument(
        "--threshold-proposal-json",
        type=Path,
        action="append",
        default=[],
        help="Threshold proposal artifact to accept into the policy.",
    )
    threshold_policy.add_argument("--output-json", type=Path, required=True)
    threshold_policy.add_argument("--accepted-by", required=True)
    threshold_policy.add_argument("--rationale", required=True)
    threshold_policy.add_argument("--representative", action="store_true")

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
    merge_quality.add_argument("--source-fk-stats-jsonl", type=Path)
    merge_quality.add_argument("--g1-stats-jsonl", type=Path)
    merge_quality.add_argument("--pair-stats-jsonl", type=Path)
    merge_quality.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    merge_quality.add_argument("--run-name", default="merged_quality")

    review_manifest = subparsers.add_parser(
        "build-review-manifest",
        help="Build JSONL/Markdown manifests for manual review from worst_clips.csv",
    )
    review_manifest.add_argument("--worst-clips-csv", type=Path, required=True)
    review_manifest.add_argument("--output-root", type=Path)
    review_manifest.add_argument("--run-name", default="manual_review")
    review_manifest.add_argument("--max-per-family", type=int, default=5)

    review_decisions = subparsers.add_parser(
        "merge-review-decisions",
        help="Merge manual review decisions into a new review manifest JSONL",
    )
    review_decisions.add_argument("--review-manifest-jsonl", type=Path, required=True)
    review_decisions.add_argument(
        "--decisions-file",
        type=Path,
        required=True,
        help="CSV or JSONL rows keyed by review_id with decision and recommended_action.",
    )
    review_decisions.add_argument("--output-jsonl", type=Path)
    review_decisions.add_argument("--output-report-json", type=Path)

    policy_audit = subparsers.add_parser(
        "audit-curation-policy",
        help="Audit whether a merged quality policy is ready for formal training",
    )
    policy_audit.add_argument("--policy-id", required=True)
    policy_audit.add_argument("--curated-report-json", type=Path, required=True)
    policy_audit.add_argument("--threshold-policy-json", type=Path)
    policy_audit.add_argument(
        "--threshold-proposal-json",
        type=Path,
        action="append",
        default=[],
        help="Threshold proposal artifact. Pass once per source/G1/pair proposal file.",
    )
    policy_audit.add_argument("--review-report-json", type=Path)
    policy_audit.add_argument("--review-manifest-jsonl", type=Path)
    policy_audit.add_argument("--review-decision-report-json", type=Path)
    policy_audit.add_argument("--output-json", type=Path)
    policy_audit.add_argument(
        "--allow-representative",
        action="store_true",
        help="Allow partial scan coverage as a warning instead of a policy blocker.",
    )
    policy_audit.add_argument(
        "--thresholds-accepted",
        action="store_true",
        help="Declare threshold proposals accepted as the named policy.",
    )
    policy_audit.add_argument(
        "--skip-review-decisions",
        action="store_true",
        help="Do not require filled manual review decisions in the manifest.",
    )
    policy_audit.add_argument(
        "--required-group-by",
        action="append",
        default=["category", "split"],
        help="Required threshold grouping field. Defaults to category and split.",
    )
    policy_audit.add_argument(
        "--diversity-dimension",
        action="append",
        default=["actor_uid", "source_skeleton", "category", "split"],
        help="Diversity-loss dimension that must retain at least one keep/downweight row.",
    )
    policy_audit.add_argument(
        "--allow-dirty-report",
        action="store_true",
        help="Do not block promotion when reports were generated from a dirty git tree.",
    )

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
    elif args.command == "scan-source-fk-quality":
        _scan_source_fk_quality(args)
    elif args.command == "scan-pair-quality":
        _scan_pair_quality(args)
    elif args.command == "propose-thresholds":
        _propose_thresholds(args)
    elif args.command == "accept-threshold-policy":
        _accept_threshold_policy(args)
    elif args.command == "build-supervised-jsonl":
        _build_supervised_jsonl(args)
    elif args.command == "build-windowed-jsonl":
        _build_windowed_jsonl(args)
    elif args.command == "merge-quality":
        _merge_quality(args)
    elif args.command == "build-review-manifest":
        _build_review_manifest(args)
    elif args.command == "merge-review-decisions":
        _merge_review_decisions(args)
    elif args.command == "audit-curation-policy":
        _audit_curation_policy(args)
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
            max_joint_acceleration=args.max_joint_acceleration,
            max_root_acceleration=args.max_root_acceleration,
            max_root_jerk=args.max_root_jerk,
            root_position_scale=args.root_position_scale,
            joint_angle_scale=args.joint_angle_scale,
            root_rotation_scale=args.root_rotation_scale,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            model_xml=args.model_xml,
            ground_height=args.ground_height,
            contact_height_threshold=args.contact_height_threshold,
            max_contact_slide_speed=args.max_contact_slide_speed,
            max_mean_foot_clearance=args.max_mean_foot_clearance,
            max_penetration_depth=args.max_penetration_depth,
            max_contact_correction_offset=args.max_contact_correction_offset,
            min_contact_frame_ratio=args.min_contact_frame_ratio,
            max_joint_limit_violation_rate=args.max_joint_limit_violation_rate,
            start_end_frames=args.start_end_frames,
            max_start_end_root_speed=args.max_start_end_root_speed,
            min_self_collision_distance=args.min_self_collision_distance,
            max_self_collision_proxy_rate=args.max_self_collision_proxy_rate,
            min_self_collision_kinematic_hops=args.min_self_collision_kinematic_hops,
        ),
        limit=limit,
        splits=tuple(args.split),
        actions=tuple(args.curation_action) or ("keep", "downweight", "quarantine"),
        sample_by=tuple(args.sample_by),
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
            max_channel_acceleration=args.max_channel_acceleration,
            max_root_acceleration=args.max_root_acceleration,
            max_root_jerk=args.max_root_jerk,
            expected_frame_time=args.expected_frame_time,
            frame_time_tolerance=args.frame_time_tolerance,
        ),
        limit=limit,
        splits=tuple(args.split),
        actions=tuple(args.curation_action) or ("keep", "downweight", "quarantine"),
        sample_by=tuple(args.sample_by),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _scan_source_fk_quality(args: argparse.Namespace) -> None:
    limit = None if args.full else args.limit
    result = scan_source_fk_quality_from_index(
        data_root=args.data_root,
        index_csv=args.index_csv,
        output_root=args.output_root,
        config=SourceFKQualityConfig(
            fps=args.fps,
            position_scale=args.position_scale,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            ground_height=args.ground_height,
            ground_percentile=args.ground_percentile,
            contact_height_threshold=args.contact_height_threshold,
            max_contact_slide_speed=args.max_contact_slide_speed,
            max_mean_foot_clearance=args.max_mean_foot_clearance,
            max_penetration_depth=args.max_penetration_depth,
            max_contact_correction_offset=args.max_contact_correction_offset,
            min_contact_frame_ratio=args.min_contact_frame_ratio,
        ),
        limit=limit,
        splits=tuple(args.split),
        actions=tuple(args.curation_action) or ("keep", "downweight", "quarantine"),
        action_column=args.action_column,
        sample_by=tuple(args.sample_by),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _scan_pair_quality(args: argparse.Namespace) -> None:
    limit = None if args.full else args.limit
    result = scan_pair_quality_from_index(
        data_root=args.data_root,
        index_csv=args.index_csv,
        output_root=args.output_root,
        config=PairQualityConfig(
            expected_source_frame_time=args.expected_source_frame_time,
            g1_fps=args.g1_fps,
            frame_time_tolerance=args.frame_time_tolerance,
            max_frame_count_delta=args.max_frame_count_delta,
            max_duration_delta_sec=args.max_duration_delta_sec,
            target_provenance=args.target_provenance,
        ),
        limit=limit,
        splits=tuple(args.split),
        actions=tuple(args.curation_action) or ("keep", "downweight", "quarantine"),
        sample_by=tuple(args.sample_by),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _propose_thresholds(args: argparse.Namespace) -> None:
    if not args.metric and not args.lower_metric:
        raise SystemExit("propose-thresholds requires at least one --metric or --lower-metric")

    payload = write_threshold_proposals(
        stats_jsonl=args.stats_jsonl,
        output_json=args.output_json,
        metrics=tuple(args.metric),
        percentile=args.percentile,
        action=args.action,
        group_by=tuple(args.group_by),
        min_group_size=args.min_group_size,
        lower_metrics=tuple(args.lower_metric),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _accept_threshold_policy(args: argparse.Namespace) -> None:
    if not args.threshold_proposal_json:
        raise SystemExit("accept-threshold-policy requires at least one --threshold-proposal-json")
    payload = write_accepted_threshold_policy(
        proposal_jsons=tuple(args.threshold_proposal_json),
        output_json=args.output_json,
        policy_id=args.policy_id,
        accepted_by=args.accepted_by,
        rationale=args.rationale,
        representative=args.representative,
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
        source_fk_stats_jsonl=args.source_fk_stats_jsonl,
        g1_stats_jsonl=args.g1_stats_jsonl,
        pair_stats_jsonl=args.pair_stats_jsonl,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _build_review_manifest(args: argparse.Namespace) -> None:
    result = build_review_manifest(
        worst_clips_csv=args.worst_clips_csv,
        output_root=args.output_root,
        run_name=args.run_name,
        max_per_family=args.max_per_family,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _merge_review_decisions(args: argparse.Namespace) -> None:
    result = merge_review_decisions(
        review_manifest_jsonl=args.review_manifest_jsonl,
        decisions_file=args.decisions_file,
        output_jsonl=args.output_jsonl,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _audit_curation_policy(args: argparse.Namespace) -> None:
    result = audit_curation_policy(
        curated_report_json=args.curated_report_json,
        threshold_proposal_jsons=tuple(args.threshold_proposal_json),
        threshold_policy_json=args.threshold_policy_json,
        review_report_json=args.review_report_json,
        review_manifest_jsonl=args.review_manifest_jsonl,
        review_decision_report_json=args.review_decision_report_json,
        output_json=args.output_json,
        config=CurationPolicyAuditConfig(
            policy_id=args.policy_id,
            allow_representative=args.allow_representative,
            thresholds_accepted=args.thresholds_accepted,
            require_review_decisions=not args.skip_review_decisions,
            required_group_by=tuple(args.required_group_by),
            diversity_dimensions=tuple(args.diversity_dimension),
            require_clean_report_git=not args.allow_dirty_report,
        ),
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
