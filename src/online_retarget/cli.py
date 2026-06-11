"""Command line helpers for repository initialization and inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from online_retarget.config import Paths
from online_retarget.config_presets import apply_config_preset as _apply_config_preset
from online_retarget.data.bones_sonic import build_sonic_index
from online_retarget.data.bvh_quality import BVHQualityConfig, scan_bvh_quality_from_index
from online_retarget.data.curation import QualityPolicy, SplitConfig, build_split_index
from online_retarget.data.g1_quality import G1QualityConfig, scan_g1_quality_from_index
from online_retarget.data.bones_seed import actor_skeletons, summarize_metadata
from online_retarget.data.pair_quality import PairQualityConfig, scan_pair_quality_from_index
from online_retarget.data.policy_audit import (
    CurationPolicyAuditConfig,
    audit_curation_policy,
    discover_threshold_proposals_for_run,
    preflight_curation_policy,
)
from online_retarget.data.quality_merge import merge_quality_stats
from online_retarget.data.quality_readiness import (
    DEFAULT_REQUIRED_LANES,
    QualityLaneInput,
    check_quality_lane_readiness,
)
from online_retarget.data.quality_review_exports import export_balanced_quality_review_csv
from online_retarget.data.quality_smoke import QualitySmokeConfig, run_quality_smoke
from online_retarget.data.quality_summary import summarize_quality_jsonl
from online_retarget.data.review_manifest import (
    build_review_decision_template,
    build_review_manifest,
    merge_review_decisions,
)
from online_retarget.data.review_clips import ReviewClipExportConfig, export_review_clips
from online_retarget.data.sonic_quality import SonicQualityConfig, scan_sonic_quality_from_index
from online_retarget.data.sonic_review_clips import (
    SonicReviewClipExportConfig,
    export_sonic_review_clips,
)
from online_retarget.data.sonic_windowed_builder import (
    SonicWindowedBuildConfig,
    _run_name,
    build_sonic_windowed_jsonl,
)
from online_retarget.data.skeleton_ae_registry import build_all_skeleton_ae_registry
from online_retarget.data.skeleton_registry import build_skeleton_registry
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

    sonic_index = subparsers.add_parser(
        "build-sonic-index",
        help="Build a read-only BONES-SONIC NPZ index and schema report",
    )
    sonic_index.add_argument(
        "--sonic-root",
        type=Path,
        default=Paths.from_env().data_root / "bones_sonic",
    )
    sonic_index.add_argument(
        "--metadata-csv",
        type=Path,
        default=Paths.from_env().data_root / "metadata" / "seed_metadata_v003.csv",
    )
    sonic_index.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    sonic_index.add_argument("--run-name", default="bones_sonic_index_v0")
    sonic_index.add_argument("--limit", type=int)

    skeleton_registry = subparsers.add_parser(
        "build-skeleton-registry",
        help="Aggregate curated rows into actor/proportional-skeleton encoder ids",
    )
    skeleton_registry.add_argument("--index-csv", type=Path, required=True)
    skeleton_registry.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    skeleton_registry.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    skeleton_registry.add_argument(
        "--run-name",
        default="bones_sonic_txt_filtered_skeleton_registry_v0",
    )
    skeleton_registry.add_argument("--action-column", default="merged_quality_action")
    skeleton_registry.add_argument(
        "--allowed-action",
        action="append",
        default=[],
        help="Allowed row action. Repeat for multiple values. Defaults to keep/downweight.",
    )

    skeleton_ae_registry = subparsers.add_parser(
        "build-skeleton-ae-registry",
        help="Build all-identity 104D skeleton geometry rows for Skeleton AE pretraining",
    )
    skeleton_ae_registry.add_argument("--index-csv", type=Path, required=True)
    skeleton_ae_registry.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    skeleton_ae_registry.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    skeleton_ae_registry.add_argument(
        "--run-name",
        default="bones_sonic_all_skeleton_ae_registry_v0",
    )
    skeleton_ae_registry.add_argument("--source-tar", type=Path)
    skeleton_ae_registry.add_argument("--skeleton-id-column", default="actor_uid")
    skeleton_ae_registry.add_argument("--validation-ratio", type=float, default=0.1)
    skeleton_ae_registry.add_argument("--seed", type=int, default=2026053001)
    skeleton_ae_registry.add_argument("--position-scale", type=float, default=0.01)
    skeleton_ae_registry.add_argument("--limit", type=int)

    sonic_quality = subparsers.add_parser(
        "scan-sonic-quality",
        help="Scan BONES-SONIC NPZ targets for provisional quality stats",
    )
    sonic_quality.add_argument("--index-csv", type=Path, required=True)
    sonic_quality.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    sonic_quality.add_argument("--limit", type=int, default=100)
    sonic_quality.add_argument("--full", action="store_true", help="Scan all SONIC index rows")
    sonic_quality.add_argument(
        "--sample-by",
        action="append",
        default=[],
        help="When --limit is active, stratify scan rows by this SONIC index field.",
    )
    sonic_quality.add_argument("--max-joint-velocity", type=float, default=20.0)
    sonic_quality.add_argument("--max-joint-step-velocity", type=float, default=20.0)
    sonic_quality.add_argument("--max-joint-acceleration", type=float)
    sonic_quality.add_argument("--max-root-speed", type=float, default=8.0)
    sonic_quality.add_argument("--max-root-step-speed", type=float, default=8.0)
    sonic_quality.add_argument("--max-root-acceleration", type=float)
    sonic_quality.add_argument("--frame-stride", type=int, default=1)
    sonic_quality.add_argument("--max-frames", type=int)
    sonic_quality.add_argument("--model-xml", type=Path)
    sonic_quality.add_argument("--ground-height", type=float, default=0.0)
    sonic_quality.add_argument("--contact-height-threshold", type=float, default=0.04)
    sonic_quality.add_argument("--max-contact-slide-speed", type=float, default=0.25)
    sonic_quality.add_argument("--max-contact-skate-distance", type=float, default=0.02)
    sonic_quality.add_argument("--max-mean-foot-clearance", type=float, default=0.10)
    sonic_quality.add_argument("--max-penetration-depth", type=float, default=0.03)
    sonic_quality.add_argument("--min-contact-frame-ratio", type=float, default=0.05)
    sonic_quality.add_argument("--max-joint-limit-violation-rate", type=float, default=0.0)
    sonic_quality.add_argument(
        "--enable-joint-limit-flags",
        action="store_true",
        help="Let XML joint-limit metrics trigger SONIC quality flags.",
    )
    sonic_quality.add_argument("--start-end-frames", type=int, default=10)
    sonic_quality.add_argument("--max-start-end-root-speed", type=float, default=0.20)
    sonic_quality.add_argument("--min-self-collision-distance", type=float, default=0.015)
    sonic_quality.add_argument("--max-self-collision-proxy-rate", type=float, default=0.0)
    sonic_quality.add_argument("--min-self-collision-kinematic-hops", type=int, default=4)
    sonic_quality.add_argument(
        "--enable-body-origin-contact-flags",
        action="store_true",
        help="Let SONIC body-origin foot/contact proxies trigger quality flags.",
    )
    sonic_quality.add_argument(
        "--enable-body-origin-self-collision-flags",
        action="store_true",
        help="Let SONIC body-origin self-collision proxies trigger quality flags.",
    )

    sonic_review = subparsers.add_parser(
        "export-sonic-review-clips",
        help="Render full-length BONES-SONIC NPZ body_pos_w 3D capsule review videos",
    )
    sonic_review.add_argument("--stats-jsonl", type=Path, required=True)
    sonic_review.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    sonic_review.add_argument("--run-name", default="sonic_review_clips_v0")
    sonic_review.add_argument("--max-per-family", type=int, default=1)
    sonic_review.add_argument("--keep-examples", type=int, default=2)
    sonic_review.add_argument(
        "--render-max-frames",
        type=int,
        default=0,
        help="Maximum frames to render per clip; 0 means full length.",
    )
    sonic_review.add_argument("--render-width", type=int, default=640)
    sonic_review.add_argument("--render-height", type=int, default=360)
    sonic_review.add_argument("--fps", type=float)

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
    quality.add_argument("--max-contact-skate-distance", type=float, default=0.02)
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
    source_fk_quality.add_argument("--max-contact-skate-distance", type=float, default=0.02)
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

    quality_smoke = subparsers.add_parser(
        "quality-smoke",
        help="Run a bounded non-promotable M2Q data-filtering smoke pipeline",
    )
    quality_smoke.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    quality_smoke.add_argument("--output-root", type=Path, default=Paths.from_env().output_root)
    quality_smoke.add_argument("--run-name", default="quality_smoke_limit24_by_category_split")
    quality_smoke.add_argument("--limit", type=int, default=24)
    quality_smoke.add_argument("--seed", type=int, default=17)
    quality_smoke.add_argument("--train-ratio", type=float, default=0.8)
    quality_smoke.add_argument("--val-ratio", type=float, default=0.1)
    quality_smoke.add_argument(
        "--sample-by",
        action="append",
        default=[],
        help="Stratify bounded source/G1/pair scans by this split-index field. Defaults to category and split.",
    )
    quality_smoke.add_argument("--threshold-percentile", type=float, default=0.95)
    quality_smoke.add_argument("--threshold-min-group-size", type=int, default=1)
    quality_smoke.add_argument("--frame-stride", type=int, default=2)
    quality_smoke.add_argument("--max-frames", type=int, default=256)
    quality_smoke.add_argument("--model-xml", type=Path)
    quality_smoke.add_argument("--review-max-per-family", type=int, default=3)

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

    run_threshold_policy = subparsers.add_parser(
        "accept-curation-threshold-policy",
        help="Write threshold_policy.json for a curated run from discovered proposal files",
    )
    run_threshold_policy.add_argument("--curated-run-dir", type=Path, required=True)
    run_threshold_policy.add_argument("--policy-id")
    run_threshold_policy.add_argument("--output-json", type=Path)
    run_threshold_policy.add_argument("--accepted-by", required=True)
    run_threshold_policy.add_argument("--rationale", required=True)
    run_threshold_policy.add_argument("--representative", action="store_true")
    run_threshold_policy.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing threshold_policy.json.",
    )

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
    windowed.add_argument("--window-stride", type=int, default=10)
    windowed.add_argument("--max-windows-per-clip", type=int, default=1)
    windowed.add_argument("--position-scale", type=float, default=0.01)

    sonic_windowed = subparsers.add_parser(
        "build-sonic-windowed-jsonl",
        help="Build walk source-BVH to BONES-SONIC joint-position JSONL samples",
    )
    sonic_windowed.add_argument(
        "--config",
        type=Path,
        help="Optional training/debug YAML config; data/build keys provide builder defaults.",
    )
    sonic_windowed.add_argument("--data-root", type=Path)
    sonic_windowed.add_argument("--index-csv", type=Path)
    sonic_windowed.add_argument("--output-root", type=Path)
    sonic_windowed.add_argument("--split", choices=("train", "val", "test"))
    sonic_windowed.add_argument("--task-query")
    sonic_windowed.add_argument(
        "--source-mode",
        choices=("sonic_body_pos", "soma_bvh"),
    )
    sonic_windowed.add_argument("--include-mirrors", action="store_true", default=None)
    sonic_windowed.add_argument("--limit", type=int)
    sonic_windowed.add_argument("--clip-limit", type=int)
    sonic_windowed.add_argument("--history-frames", type=int)
    sonic_windowed.add_argument("--target-frame-offset", type=int)
    sonic_windowed.add_argument("--target-horizon-frames", type=int)
    sonic_windowed.add_argument("--target-future-step", type=int)
    sonic_windowed.add_argument("--source-rotation", choices=("rot6d",))
    sonic_windowed.add_argument(
        "--source-bvh-tar",
        type=Path,
        help="Explicit SOMA-proportional BVH tar path; defaults to data_root/soma_proportional.tar.",
    )
    sonic_windowed.add_argument(
        "--sonic-npz-root",
        type=Path,
        help="Root for sonic_relative_path NPZ lookup, e.g. /mnt/data_cpfs/bones_sonic.",
    )
    sonic_windowed.add_argument(
        "--sonic-path-prefix-from",
        help="Prefix in index sonic_path to rewrite when sonic_relative_path root is not used.",
    )
    sonic_windowed.add_argument(
        "--sonic-path-prefix-to",
        help="Replacement prefix for --sonic-path-prefix-from.",
    )
    sonic_windowed.add_argument(
        "--no-source-angular-velocity",
        action="store_true",
        default=None,
        help="Reserve angular-velocity token slots but fill them with zeros.",
    )
    sonic_windowed.add_argument("--window-stride", type=int)
    sonic_windowed.add_argument("--max-windows-per-clip", type=int)
    sonic_windowed.add_argument("--split-seed", type=int)
    sonic_windowed.add_argument("--train-ratio", type=float)
    sonic_windowed.add_argument("--val-ratio", type=float)
    sonic_windowed.add_argument("--position-scale", type=float)

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

    review_template = subparsers.add_parser(
        "build-review-decision-template",
        help="Build a reviewer-fillable CSV decision template from review_manifest.jsonl",
    )
    review_template.add_argument("--review-manifest-jsonl", type=Path, required=True)
    review_template.add_argument("--output-csv", type=Path)
    review_template.add_argument("--output-report-json", type=Path)
    review_template.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing template/report file.",
    )

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

    review_clips = subparsers.add_parser(
        "export-review-clips",
        help="Export source BVH, G1 CSV, metadata, and optional G1 preview MP4s from a review CSV",
    )
    review_clips.add_argument("--data-root", type=Path, default=Paths.from_env().data_root)
    review_clips.add_argument("--input-csv", type=Path, required=True)
    review_clips.add_argument("--output-root", type=Path, default=Paths.from_env().output_root / "review_clips")
    review_clips.add_argument("--run-name", default="review_clips")
    review_clips.add_argument("--label", default="review")
    review_clips.add_argument("--limit", type=int, default=8)
    review_clips.add_argument("--render-g1", action="store_true")
    review_clips.add_argument("--render-source-capsules", action="store_true")
    review_clips.add_argument("--render-g1-capsules", action="store_true")
    review_clips.add_argument("--model-xml", type=Path)
    review_clips.add_argument(
        "--render-max-frames",
        type=int,
        default=120,
        help="Maximum frames to render per clip; use 0 for full length.",
    )
    review_clips.add_argument("--render-width", type=int, default=640)
    review_clips.add_argument("--render-height", type=int, default=360)
    review_clips.add_argument("--fps", type=float, default=120.0)
    review_clips.add_argument("--root-position-scale", type=float, default=0.01)
    review_clips.add_argument("--source-position-scale", type=float, default=0.01)
    review_clips.add_argument("--angle-scale", type=float, default=0.017453292519943295)
    review_clips.add_argument(
        "--hide-render-frames",
        action="store_true",
        help="Do not overlay MuJoCo body frames in rendered videos.",
    )

    balanced_review = subparsers.add_parser(
        "export-balanced-quality-review",
        help="Build a flag-balanced review CSV from a quality stats JSONL file",
    )
    balanced_review.add_argument("--stats-jsonl", type=Path, required=True)
    balanced_review.add_argument(
        "--split-index-csv",
        type=Path,
        help="Optional split index used to backfill source BVH paths for target-only stats.",
    )
    balanced_review.add_argument("--output-csv", type=Path, required=True)
    balanced_review.add_argument("--output-report-json", type=Path)
    balanced_review.add_argument(
        "--flag",
        action="append",
        default=[],
        help="Quality flag to sample. Defaults to all flags found in descending frequency.",
    )
    balanced_review.add_argument("--max-per-flag", type=int, default=2)
    balanced_review.add_argument("--action-min-rank", default="quarantine")
    balanced_review.add_argument(
        "--include-downweight",
        action="store_true",
        help="Allow downweight rows in addition to quarantine/exclude rows.",
    )

    quality_summary = subparsers.add_parser(
        "summarize-quality-jsonl",
        help="Summarize a quality stats JSONL file for progress and final checkpoints",
    )
    quality_summary.add_argument("--stats-jsonl", type=Path, required=True)
    quality_summary.add_argument("--output-json", type=Path, required=True)
    quality_summary.add_argument(
        "--metric",
        action="append",
        default=[],
        help="Metric column to summarize. Defaults to common G1/source quality metrics.",
    )
    quality_summary.add_argument(
        "--group-by",
        action="append",
        default=[],
        help="Categorical field to count. Defaults to split and category.",
    )
    quality_summary.add_argument(
        "--quantile",
        action="append",
        type=float,
        default=[],
        help="Quantile to report in [0, 1]. Defaults to 0.5, 0.9, 0.95, and 0.99.",
    )

    quality_readiness = subparsers.add_parser(
        "check-quality-readiness",
        help="Check whether source/source-FK/G1/pair quality lanes are full-scan ready",
    )
    quality_readiness.add_argument("--index-csv", type=Path, required=True)
    quality_readiness.add_argument("--output-json", type=Path, required=True)
    quality_readiness.add_argument(
        "--curation-action",
        action="append",
        default=[],
        help="Index curation action to count as expected. Defaults to keep/downweight/quarantine.",
    )
    quality_readiness.add_argument(
        "--required-lane",
        action="append",
        default=[],
        choices=DEFAULT_REQUIRED_LANES,
        help="Required lane. Defaults to source, source_fk, g1, and pair.",
    )
    quality_readiness.add_argument("--source-stats-jsonl", type=Path)
    quality_readiness.add_argument("--source-report-json", type=Path)
    quality_readiness.add_argument("--source-fk-stats-jsonl", type=Path)
    quality_readiness.add_argument("--source-fk-report-json", type=Path)
    quality_readiness.add_argument("--g1-stats-jsonl", type=Path)
    quality_readiness.add_argument("--g1-report-json", type=Path)
    quality_readiness.add_argument("--pair-stats-jsonl", type=Path)
    quality_readiness.add_argument("--pair-report-json", type=Path)

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

    policy_preflight = subparsers.add_parser(
        "preflight-curation-policy",
        help="Run the standard curation-policy preflight from a curated run directory",
    )
    policy_preflight.add_argument("--curated-run-dir", type=Path, required=True)
    policy_preflight.add_argument("--policy-id")
    policy_preflight.add_argument("--threshold-policy-json", type=Path)
    policy_preflight.add_argument("--review-decision-report-json", type=Path)
    policy_preflight.add_argument("--output-json", type=Path)
    policy_preflight.add_argument(
        "--allow-representative",
        action="store_true",
        help="Allow partial scan coverage as a warning instead of a policy blocker.",
    )
    policy_preflight.add_argument(
        "--thresholds-accepted",
        action="store_true",
        help="Declare discovered threshold proposals accepted as the named policy.",
    )
    policy_preflight.add_argument(
        "--skip-review-decisions",
        action="store_true",
        help="Do not require filled manual review decisions in the manifest.",
    )
    policy_preflight.add_argument(
        "--required-group-by",
        action="append",
        default=["category", "split"],
        help="Required threshold grouping field. Defaults to category and split.",
    )
    policy_preflight.add_argument(
        "--diversity-dimension",
        action="append",
        default=["actor_uid", "source_skeleton", "category", "split"],
        help="Diversity-loss dimension that must retain at least one keep/downweight row.",
    )
    policy_preflight.add_argument(
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
    offline_eval.add_argument("--ground-height", type=float, default=0.0)
    offline_eval.add_argument("--up-axis", default=2)
    offline_eval.add_argument("--contact-height-threshold", type=float, default=0.04)
    offline_eval.add_argument("--max-contact-slide-speed", type=float, default=0.25)
    offline_eval.add_argument("--max-contact-skate-distance", type=float, default=0.02)
    offline_eval.add_argument("--failure-metric", default="joint_rmse")
    offline_eval.add_argument("--max-failures", type=int, default=50)

    args = parser.parse_args()
    if args.command == "inventory":
        _inventory(args.data_root, args.sample_actors)
    elif args.command == "build-sonic-index":
        _build_sonic_index(args)
    elif args.command == "build-skeleton-registry":
        _build_skeleton_registry(args)
    elif args.command == "build-skeleton-ae-registry":
        _build_skeleton_ae_registry(args)
    elif args.command == "scan-sonic-quality":
        _scan_sonic_quality(args)
    elif args.command == "export-sonic-review-clips":
        _export_sonic_review_clips(args)
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
    elif args.command == "quality-smoke":
        _quality_smoke(args)
    elif args.command == "propose-thresholds":
        _propose_thresholds(args)
    elif args.command == "accept-threshold-policy":
        _accept_threshold_policy(args)
    elif args.command == "accept-curation-threshold-policy":
        _accept_curation_threshold_policy(args)
    elif args.command == "build-supervised-jsonl":
        _build_supervised_jsonl(args)
    elif args.command == "build-windowed-jsonl":
        _build_windowed_jsonl(args)
    elif args.command == "build-sonic-windowed-jsonl":
        _build_sonic_windowed_jsonl(args)
    elif args.command == "merge-quality":
        _merge_quality(args)
    elif args.command == "build-review-manifest":
        _build_review_manifest(args)
    elif args.command == "build-review-decision-template":
        _build_review_decision_template(args)
    elif args.command == "merge-review-decisions":
        _merge_review_decisions(args)
    elif args.command == "export-review-clips":
        _export_review_clips(args)
    elif args.command == "export-balanced-quality-review":
        _export_balanced_quality_review(args)
    elif args.command == "summarize-quality-jsonl":
        _summarize_quality_jsonl(args)
    elif args.command == "check-quality-readiness":
        _check_quality_readiness(args)
    elif args.command == "audit-curation-policy":
        _audit_curation_policy(args)
    elif args.command == "preflight-curation-policy":
        _preflight_curation_policy(args)
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


def _build_sonic_index(args: argparse.Namespace) -> None:
    result = build_sonic_index(
        sonic_root=args.sonic_root,
        metadata_csv=args.metadata_csv,
        output_root=args.output_root,
        run_name=args.run_name,
        limit=args.limit,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _build_skeleton_registry(args: argparse.Namespace) -> None:
    result = build_skeleton_registry(
        index_csv=args.index_csv,
        data_root=args.data_root,
        output_root=args.output_root,
        run_name=args.run_name,
        action_column=args.action_column,
        allowed_actions=tuple(args.allowed_action or ["keep", "downweight"]),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _build_skeleton_ae_registry(args: argparse.Namespace) -> None:
    result = build_all_skeleton_ae_registry(
        index_csv=args.index_csv,
        data_root=args.data_root,
        output_root=args.output_root,
        run_name=args.run_name,
        skeleton_id_column=args.skeleton_id_column,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        position_scale=args.position_scale,
        source_tar=args.source_tar,
        limit=args.limit,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _scan_sonic_quality(args: argparse.Namespace) -> None:
    result = scan_sonic_quality_from_index(
        index_csv=args.index_csv,
        output_root=args.output_root,
        config=SonicQualityConfig(
            max_joint_velocity=args.max_joint_velocity,
            max_joint_step_velocity=args.max_joint_step_velocity,
            max_joint_acceleration=args.max_joint_acceleration,
            max_root_speed=args.max_root_speed,
            max_root_step_speed=args.max_root_step_speed,
            max_root_acceleration=args.max_root_acceleration,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            model_xml=args.model_xml,
            ground_height=args.ground_height,
            contact_height_threshold=args.contact_height_threshold,
            max_contact_slide_speed=args.max_contact_slide_speed,
            max_contact_skate_distance=args.max_contact_skate_distance,
            max_mean_foot_clearance=args.max_mean_foot_clearance,
            max_penetration_depth=args.max_penetration_depth,
            min_contact_frame_ratio=args.min_contact_frame_ratio,
            max_joint_limit_violation_rate=args.max_joint_limit_violation_rate,
            enable_joint_limit_flags=args.enable_joint_limit_flags,
            start_end_frames=args.start_end_frames,
            max_start_end_root_speed=args.max_start_end_root_speed,
            min_self_collision_distance=args.min_self_collision_distance,
            max_self_collision_proxy_rate=args.max_self_collision_proxy_rate,
            min_self_collision_kinematic_hops=args.min_self_collision_kinematic_hops,
            enable_body_origin_contact_flags=args.enable_body_origin_contact_flags,
            enable_body_origin_self_collision_flags=args.enable_body_origin_self_collision_flags,
        ),
        limit=None if args.full else args.limit,
        sample_by=args.sample_by,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _export_sonic_review_clips(args: argparse.Namespace) -> None:
    result = export_sonic_review_clips(
        SonicReviewClipExportConfig(
            stats_jsonl=args.stats_jsonl,
            output_root=args.output_root,
            run_name=args.run_name,
            max_per_family=args.max_per_family,
            keep_examples=args.keep_examples,
            render_max_frames=args.render_max_frames,
            render_width=args.render_width,
            render_height=args.render_height,
            fps=args.fps,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _build_sonic_windowed_jsonl(args: argparse.Namespace) -> None:
    payload = _apply_config_preset(_load_mapping_config(args.config)) if args.config else {}
    data_cfg = _mapping_section(payload, "data")
    build_cfg = _mapping_section(data_cfg, "build")
    data_root = args.data_root or _path_from_config(data_cfg.get("root")) or Paths.from_env().data_root
    index_csv = args.index_csv or _path_from_config(data_cfg.get("index_csv"))
    if index_csv is None:
        raise SystemExit("build-sonic-windowed-jsonl requires --index-csv or data.index_csv in --config")
    build_config = _sonic_windowed_build_config_from_args(args, data_cfg=data_cfg, build_cfg=build_cfg)
    output_root = _sonic_windowed_output_root_from_config(
        args,
        payload=payload,
        data_cfg=data_cfg,
        build_cfg=build_cfg,
        build_config=build_config,
    )
    result = build_sonic_windowed_jsonl(
        data_root=data_root,
        index_csv=index_csv,
        output_root=output_root,
        config=build_config,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _sonic_windowed_output_root_from_config(
    args: argparse.Namespace,
    *,
    payload: dict[str, object],
    data_cfg: dict[str, object],
    build_cfg: dict[str, object],
    build_config: SonicWindowedBuildConfig,
) -> Path:
    if args.output_root is not None:
        return args.output_root
    samples_jsonl = _path_from_config(data_cfg.get("samples_jsonl"))
    if samples_jsonl is not None:
        return _output_root_from_configured_samples(samples_jsonl, build_config)
    experiment_cfg = _mapping_section(payload, "experiment")
    return (
        _path_from_config(build_cfg.get("output_root"))
        or _path_from_config(experiment_cfg.get("output_root"))
        or Paths.from_env().output_root
    )


def _output_root_from_configured_samples(
    samples_jsonl: Path,
    build_config: SonicWindowedBuildConfig,
) -> Path:
    expected_run_name = _run_name(build_config)
    if (
        samples_jsonl.name != "samples.jsonl"
        or samples_jsonl.parent.name != expected_run_name
        or samples_jsonl.parent.parent.name != "supervised"
    ):
        raise SystemExit(
            "data.samples_jsonl must match the build-sonic-windowed-jsonl run name for the "
            f"selected policy_preset: expected */supervised/{expected_run_name}/samples.jsonl, "
            f"got {samples_jsonl}"
        )
    return samples_jsonl.parent.parent.parent


def _sonic_windowed_build_config_from_args(
    args: argparse.Namespace,
    *,
    data_cfg: dict[str, object],
    build_cfg: dict[str, object],
) -> SonicWindowedBuildConfig:
    defaults = SonicWindowedBuildConfig()

    def pick(name: str, default: object) -> object:
        cli_value = getattr(args, name)
        if cli_value is not None:
            return cli_value
        if name in build_cfg and build_cfg[name] is not None:
            return build_cfg[name]
        if name in data_cfg and data_cfg[name] is not None:
            return data_cfg[name]
        return default

    no_source_angular_velocity = getattr(args, "no_source_angular_velocity")
    if no_source_angular_velocity is None:
        include_source_angular_velocity = bool(
            build_cfg.get(
                "include_source_angular_velocity",
                data_cfg.get("include_source_angular_velocity", defaults.include_source_angular_velocity),
            )
        )
    else:
        include_source_angular_velocity = not bool(no_source_angular_velocity)

    return SonicWindowedBuildConfig(
        split=str(pick("split", defaults.split)),
        task_query=str(pick("task_query", data_cfg.get("task", defaults.task_query))),
        source_mode=str(pick("source_mode", defaults.source_mode)),
        include_mirrors=bool(pick("include_mirrors", defaults.include_mirrors)),
        limit=int(pick("limit", defaults.limit)),
        clip_limit=_optional_int(pick("clip_limit", defaults.clip_limit)),
        history_frames=int(pick("history_frames", defaults.history_frames)),
        target_frame_offset=int(pick("target_frame_offset", defaults.target_frame_offset)),
        target_horizon_frames=int(pick("target_horizon_frames", defaults.target_horizon_frames)),
        target_future_step=int(pick("target_future_step", defaults.target_future_step)),
        source_rotation=str(pick("source_rotation", defaults.source_rotation)),
        include_source_angular_velocity=include_source_angular_velocity,
        source_bvh_tar=_optional_str(pick("source_bvh_tar", defaults.source_bvh_tar)),
        sonic_npz_root=_optional_str(pick("sonic_npz_root", defaults.sonic_npz_root)),
        sonic_path_prefix_from=_optional_str(
            pick("sonic_path_prefix_from", defaults.sonic_path_prefix_from)
        ),
        sonic_path_prefix_to=_optional_str(
            pick("sonic_path_prefix_to", defaults.sonic_path_prefix_to)
        ),
        window_stride=int(pick("window_stride", defaults.window_stride)),
        max_windows_per_clip=int(pick("max_windows_per_clip", defaults.max_windows_per_clip)),
        split_seed=int(pick("split_seed", defaults.split_seed)),
        train_ratio=float(pick("train_ratio", defaults.train_ratio)),
        val_ratio=float(pick("val_ratio", defaults.val_ratio)),
        position_scale=float(pick("position_scale", defaults.position_scale)),
    )


def _load_mapping_config(path: Path) -> dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(f"PyYAML is required to read --config for build-sonic-windowed-jsonl: {path}") from exc
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return payload


def _mapping_section(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _path_from_config(value: object) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


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
            max_contact_skate_distance=args.max_contact_skate_distance,
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
            max_contact_skate_distance=args.max_contact_skate_distance,
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


def _quality_smoke(args: argparse.Namespace) -> None:
    max_frames = args.max_frames if args.max_frames > 0 else None
    result = run_quality_smoke(
        data_root=args.data_root,
        output_root=args.output_root,
        config=QualitySmokeConfig(
            run_name=args.run_name,
            limit=args.limit,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            sample_by=tuple(args.sample_by) or ("category", "split"),
            threshold_percentile=args.threshold_percentile,
            threshold_min_group_size=args.threshold_min_group_size,
            frame_stride=args.frame_stride,
            max_frames=max_frames,
            model_xml=args.model_xml,
            review_max_per_family=args.review_max_per_family,
        ),
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


def _accept_curation_threshold_policy(args: argparse.Namespace) -> None:
    output_json = args.output_json or args.curated_run_dir / "threshold_policy.json"
    if output_json.exists() and not args.overwrite:
        raise SystemExit(f"threshold policy already exists; pass --overwrite to replace: {output_json}")
    proposal_jsons = discover_threshold_proposals_for_run(args.curated_run_dir)
    if not proposal_jsons:
        raise SystemExit(f"no threshold proposals found for curated run: {args.curated_run_dir}")
    payload = write_accepted_threshold_policy(
        proposal_jsons=tuple(proposal_jsons),
        output_json=output_json,
        policy_id=args.policy_id or args.curated_run_dir.name,
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
            window_stride=args.window_stride,
            max_windows_per_clip=args.max_windows_per_clip,
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


def _build_review_decision_template(args: argparse.Namespace) -> None:
    result = build_review_decision_template(
        review_manifest_jsonl=args.review_manifest_jsonl,
        output_csv=args.output_csv,
        output_report_json=args.output_report_json,
        overwrite=args.overwrite,
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


def _export_review_clips(args: argparse.Namespace) -> None:
    result = export_review_clips(
        data_root=args.data_root,
        input_csv=args.input_csv,
        output_root=args.output_root,
        run_name=args.run_name,
        label=args.label,
        config=ReviewClipExportConfig(
            limit=args.limit,
            render_g1=args.render_g1,
            render_source_capsules=args.render_source_capsules,
            render_g1_capsules=args.render_g1_capsules,
            model_xml=args.model_xml,
            render_max_frames=args.render_max_frames,
            render_width=args.render_width,
            render_height=args.render_height,
            fps=args.fps,
            root_position_scale=args.root_position_scale,
            source_position_scale=args.source_position_scale,
            angle_scale=args.angle_scale,
            render_frames=not args.hide_render_frames,
        ),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _export_balanced_quality_review(args: argparse.Namespace) -> None:
    result = export_balanced_quality_review_csv(
        stats_jsonl=args.stats_jsonl,
        output_csv=args.output_csv,
        split_index_csv=args.split_index_csv,
        output_report_json=args.output_report_json,
        flags=tuple(args.flag),
        max_per_flag=args.max_per_flag,
        action_min_rank=args.action_min_rank,
        include_downweight=args.include_downweight,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _summarize_quality_jsonl(args: argparse.Namespace) -> None:
    result = summarize_quality_jsonl(
        stats_jsonl=args.stats_jsonl,
        output_json=args.output_json,
        metrics=tuple(args.metric) if args.metric else (),
        group_by=tuple(args.group_by) if args.group_by else (),
        quantiles=tuple(args.quantile) if args.quantile else (),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _check_quality_readiness(args: argparse.Namespace) -> None:
    required_lanes = set(args.required_lane or DEFAULT_REQUIRED_LANES)
    result = check_quality_lane_readiness(
        index_csv=args.index_csv,
        output_json=args.output_json,
        lanes=(
            QualityLaneInput(
                "source",
                stats_jsonl=args.source_stats_jsonl,
                report_json=args.source_report_json,
                required="source" in required_lanes,
            ),
            QualityLaneInput(
                "source_fk",
                stats_jsonl=args.source_fk_stats_jsonl,
                report_json=args.source_fk_report_json,
                required="source_fk" in required_lanes,
            ),
            QualityLaneInput(
                "g1",
                stats_jsonl=args.g1_stats_jsonl,
                report_json=args.g1_report_json,
                required="g1" in required_lanes,
            ),
            QualityLaneInput(
                "pair",
                stats_jsonl=args.pair_stats_jsonl,
                report_json=args.pair_report_json,
                required="pair" in required_lanes,
            ),
        ),
        actions=tuple(args.curation_action) or ("keep", "downweight", "quarantine"),
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


def _preflight_curation_policy(args: argparse.Namespace) -> None:
    result = preflight_curation_policy(
        curated_run_dir=args.curated_run_dir,
        policy_id=args.policy_id,
        output_json=args.output_json,
        threshold_policy_json=args.threshold_policy_json,
        review_decision_report_json=args.review_decision_report_json,
        config=CurationPolicyAuditConfig(
            policy_id=args.policy_id or args.curated_run_dir.name,
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
            ground_height=args.ground_height,
            up_axis=args.up_axis,
            contact_height_threshold=args.contact_height_threshold,
            max_contact_slide_speed=args.max_contact_slide_speed,
            max_contact_skate_distance=args.max_contact_skate_distance,
            failure_metric=args.failure_metric,
            max_failures=args.max_failures,
            run_name=args.run_name,
        ),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
